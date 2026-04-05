"""
Exponential Cross-Layer DCT for multi-layer backdoor detection.

CrossLayerDCT finds input directions at src_layer that maximally perturb
the residual stream at tgt_layer, using:
  1. Cross-layer Jacobian via VJP (spans multiple transformer layers)
  2. Exponential singular value reweighting: exp(sigma / tau)
  3. Iterative alternating U/V refinement (n_iter steps)
  4. Calibrated alpha: median |delta| / |h_tgt| ~ target_ratio
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import vmap
from torch.func import vjp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dct import StreamingAverage, format_duration


class CrossLayerDeltaActs(nn.Module):
    """
    Cross-layer delta activation function.

    For steering vector theta injected at the residual stream after src_layer,
    computes: delta(theta) = mean_over_positions(h_tgt_steered - h_tgt_baseline)

    The partial forward runs backbone.layers[src+1 .. tgt] on the steered state.

    forward() args:
        theta          : (d_model,)
        h_src          : (batch, seq, d_model)  residual stream after src_layer
        h_tgt_baseline : (batch, seq, d_model)  unsteered residual stream at tgt_layer
    Returns:
        (batch, d_model)
    """

    def __init__(
        self,
        backbone: nn.Module,
        src_layer: int,
        tgt_layer: int,
        position_ids: torch.Tensor,
        position_embeddings: tuple,
        cache_position: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.backbone = backbone
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.position_ids = position_ids
        self.position_embeddings = position_embeddings
        self.cache_position = cache_position
        self.device = device

    def forward(
        self,
        theta: torch.Tensor,
        h_src: torch.Tensor,
        h_tgt_baseline: torch.Tensor,
    ) -> torch.Tensor:
        h = h_src + theta.unsqueeze(0).unsqueeze(0)
        for i in range(self.src_layer + 1, self.tgt_layer + 1):
            out = self.backbone.layers[i](
                h,
                attention_mask=None,          # SDPA uses is_causal=True when mask is None
                position_ids=self.position_ids,
                position_embeddings=self.position_embeddings,
                use_cache=False,
                cache_position=self.cache_position,
            )
            h = out[0] if isinstance(out, tuple) else out
        return (h - h_tgt_baseline).mean(dim=1)  # (batch, d_model)


class CrossLayerDCT:
    """
    Exponential cross-layer DCT.

    Fits n_factors input directions V: (d_model, n_factors) at src_layer's
    residual stream that maximally perturb the residual stream at tgt_layer.

    Differences from LinearDCT:
    - Jacobian spans src_layer to tgt_layer, not within a single MLP
    - Exponential singular value reweighting via exp(sigma / tau)
    - Iterative U/V alternation (n_iter refinement steps)
    """

    def __init__(self, n_factors: int = 64, tau: float = 1.0, n_iter: int = 10):
        self.n_factors = n_factors
        self.tau = tau
        self.n_iter = n_iter
        self.V = None
        self.U = None

    def _exp_reweight(self, sigma: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
        """
        Apply exponential singular value reweighting, then re-orthonormalize via QR.

        sigma : (k,) singular values, descending
        Vh    : (k, d_model) right singular vectors

        Returns V: (d_model, n_factors) orthonormal columns.
        """
        k = min(self.n_factors, sigma.shape[0])
        sigma_k = sigma[:k]
        Vh_k = Vh[:k]  # (k, d_model)

        # Shift by max for numerical stability, then exponentiate
        shifted = sigma_k - sigma_k.max()
        weights = torch.exp(shifted / self.tau)   # (k,)

        # Weight columns then re-orthonormalize via QR
        V_weighted = Vh_k.T * weights.unsqueeze(0)  # (d_model, k)
        V_q, _ = torch.linalg.qr(V_weighted)        # (d_model, k) orthonormal
        return V_q.contiguous()

    def fit(
        self,
        delta_fn: CrossLayerDeltaActs,
        H_src: torch.Tensor,
        H_tgt: torch.Tensor,
        dim_output_projection: int = 64,
        batch_size: int = 1,
        factor_batch_size: int = 8,
    ) -> tuple:
        """
        H_src : (n_train, seq, d_model)  residual stream at src_layer
        H_tgt : (n_train, seq, d_model)  residual stream at tgt_layer

        Returns (U, V) where U: (d_model, n_factors), V: (d_model, n_factors).
        """
        assert dim_output_projection >= self.n_factors

        d_model = H_src.shape[2]
        dev = delta_fn.device
        n_train = H_src.shape[0]
        n_batches = max(1, (n_train + batch_size - 1) // batch_size)

        def vjp_single(u, v, h_src_b, h_tgt_b):
            output, vjp_fn = vjp(lambda _v: delta_fn(_v, h_src_b, h_tgt_b), v)
            with torch.no_grad():
                udots = output @ u
            return udots, output.detach(), vjp_fn(u.expand(h_src_b.shape[0], -1))[0].detach()

        vjp_batch = vmap(
            lambda u, v, h_src_b, h_tgt_b: vjp_single(u, v, h_src_b, h_tgt_b),
            in_dims=(1, 1, None, None),
            out_dims=(1, 2, 1),
            chunk_size=factor_batch_size,
        )

        delta_vmap = vmap(
            delta_fn,
            in_dims=(1, None, None),
            out_dims=2,
            chunk_size=factor_batch_size,
        )

        U_rand = F.normalize(torch.randn(d_model, dim_output_projection), dim=0).to(dev)
        V_zeros = torch.zeros(d_model, dim_output_projection, device=dev)

        # Initial Jacobian estimate (same as LinearDCT but cross-layer)
        print(f"  Computing initial cross-layer Jacobian (src={delta_fn.src_layer} -> tgt={delta_fn.tgt_layer})...")
        J_avg = StreamingAverage()
        with torch.no_grad():
            for t in tqdm(range(0, n_train, batch_size), total=n_batches,
                          desc="    Jacobian batches", leave=False):
                h_src_b = H_src[t:t + batch_size].to(dev)
                h_tgt_b = H_tgt[t:t + batch_size].to(dev)
                _, _, J_batch = vjp_batch(U_rand, V_zeros, h_src_b, h_tgt_b)
                J_avg.update(J_batch.t().unsqueeze(0))

        J = J_avg.get()
        _, sigma, Vh = torch.linalg.svd(J, full_matrices=False)
        V_current = self._exp_reweight(sigma, Vh)  # (d_model, n_factors)

        print(f"  Initial sigma: [{sigma[0]:.4f} .. {sigma[self.n_factors - 1]:.4f}]  "
              f"tau={self.tau}  exp weight ratio: {torch.exp((sigma[0] - sigma[self.n_factors - 1]) / self.tau):.1f}x")

        U_current = None

        # Iterative refinement
        for iteration in range(1, self.n_iter + 1):
            print(f"  Refinement {iteration}/{self.n_iter}...")

            # Step 1: Compute U from current V (forward pass)
            U_avg = StreamingAverage()
            with torch.no_grad():
                for t in range(0, n_train, batch_size):
                    h_src_b = H_src[t:t + batch_size].to(dev)
                    h_tgt_b = H_tgt[t:t + batch_size].to(dev)
                    U_batch = delta_vmap(V_current, h_src_b, h_tgt_b)
                    U_avg.update(U_batch)
            U_current = F.normalize(U_avg.get(), dim=0)  # (d_model, n_factors)

            # Step 2: Pad U to dim_output_projection for VJP projection
            if U_current.shape[1] < dim_output_projection:
                pad = F.normalize(
                    torch.randn(d_model, dim_output_projection - U_current.shape[1], device=dev),
                    dim=0,
                )
                U_proj = torch.cat([U_current, pad], dim=1)
            else:
                U_proj = U_current

            # Step 3: Recompute Jacobian using U_current as output projection
            J_avg_iter = StreamingAverage()
            V_prev = torch.zeros(d_model, dim_output_projection, device=dev)
            with torch.no_grad():
                for t in tqdm(range(0, n_train, batch_size), total=n_batches,
                              desc="    VJP batches", leave=False):
                    h_src_b = H_src[t:t + batch_size].to(dev)
                    h_tgt_b = H_tgt[t:t + batch_size].to(dev)
                    _, _, J_batch = vjp_batch(U_proj, V_prev, h_src_b, h_tgt_b)
                    J_avg_iter.update(J_batch.t().unsqueeze(0))

            J_new = J_avg_iter.get()
            _, sigma_new, Vh_new = torch.linalg.svd(J_new, full_matrices=False)
            V_current = self._exp_reweight(sigma_new, Vh_new)

        self.V = V_current
        self.U = U_current
        return self.U, self.V

    def calibrate_alpha(
        self,
        delta_fn: CrossLayerDeltaActs,
        H_src: torch.Tensor,
        H_tgt: torch.Tensor,
        direction: torch.Tensor,
        target_ratio: float = 0.5,
        n_cal: int = 30,
    ) -> float:
        """
        Find alpha giving median |delta_h_tgt| / |h_tgt_mean| ~ target_ratio.

        Returns the probe alpha whose ratio is closest to target_ratio.
        """
        probe_alphas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        n_cal = min(n_cal, H_src.shape[0])
        cal_src = H_src[:n_cal].to(delta_fn.device)
        cal_tgt = H_tgt[:n_cal].to(delta_fn.device)
        h_tgt_norms = cal_tgt.mean(dim=1).norm(dim=-1)  # (n_cal,) norm of mean-pooled baseline

        ratios = []
        with torch.no_grad():
            for alpha in probe_alphas:
                delta_h = delta_fn(alpha * direction, cal_src, cal_tgt)  # (n_cal, d_model)
                ratio = (delta_h.norm(dim=-1) / h_tgt_norms.clamp(min=1e-8)).median().item()
                ratios.append(ratio)

        diffs = [abs(r - target_ratio) for r in ratios]
        best_idx = diffs.index(min(diffs))
        print(f"  Calibration ratios: {[f'{r:.3f}' for r in ratios]}")
        print(f"  Best alpha: {probe_alphas[best_idx]:.1f} (ratio={ratios[best_idx]:.3f}, target={target_ratio})")
        return probe_alphas[best_idx]


def collect_cross_layer_acts(
    backbone: nn.Module,
    tokenizer,
    texts: list,
    cross_pairs: list,
    seq_len: int = 64,
    device: torch.device = None,
) -> dict:
    """
    Collect residual stream activations for all layers referenced in cross_pairs.

    Returns dict: {(src, tgt): (H_src, H_tgt)} where each tensor is (n_texts, seq_len, d_model).
    The hidden state captured is the output of that layer (post-residual-connection).
    """
    if device is None:
        device = next(backbone.parameters()).device

    all_layers = set()
    for src, tgt in cross_pairs:
        all_layers.add(src)
        all_layers.add(tgt)

    cap: dict = {l: [] for l in all_layers}

    hooks = []
    for l in all_layers:
        def make_hook(li: int):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[li].append(h.detach().cpu())
            return hook
        hooks.append(backbone.layers[l].register_forward_hook(make_hook(l)))

    for text in tqdm(texts, desc="  Collecting cross-layer activations", leave=False):
        text = (text or "").strip() or "."
        enc = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            backbone(enc["input_ids"])

    for h in hooks:
        h.remove()

    # Stack to (n_texts, seq_len, d_model), clamping to seq_len
    H = {l: torch.cat(cap[l], dim=0)[:, :seq_len, :] for l in all_layers}

    return {(src, tgt): (H[src], H[tgt]) for src, tgt in cross_pairs}
