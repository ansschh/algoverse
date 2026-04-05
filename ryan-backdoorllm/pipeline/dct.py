"""
LinearDCT: Jacobian-based causal direction finder for GPT-2 MLP layers.

For each MLP layer, finds the input directions that most causally influence
the output by computing the Jacobian of:

    delta(theta, x, y) = mean_over_positions(mlp(x + theta) - y)

The top singular vectors of J = d(delta)/d(theta) at theta=0 are the
directions the MLP is most sensitive to (the causal input directions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import vmap
from torch.func import vjp
from tqdm import tqdm


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class StreamingAverage:
    """Online mean accumulator: tracks a running average without storing all data."""
    def __init__(self):
        self.count = 0
        self.mean = None

    def update(self, batch: torch.Tensor) -> torch.Tensor:
        n = batch.size(0)
        if self.mean is None:
            self.mean = batch.mean(dim=0)
            self.count = n
            return self.mean
        new_count = self.count + n
        self.mean = self.mean + (batch.mean(dim=0) - self.mean) * (n / new_count)
        self.count = new_count
        return self.mean

    def get(self) -> torch.Tensor:
        if self.mean is None:
            raise ValueError("No data yet")
        return self.mean


class MLPDeltaActs(nn.Module):
    """
    Computes how an MLP's output changes when its input is perturbed by theta:

        delta(theta, x, y) = mean_over_positions(mlp(x + theta) - y)

    Args:
        mlp_module : nn.Module             the MLP to perturb (any architecture)
        device     : torch.device
        theta : (d_model,)               steering vector
        x     : (batch, seq_len, d_model) MLP input activations
        y     : (batch, seq_len, d_model) unperturbed MLP output

    Returns:
        (batch, d_model)
    """
    def __init__(self, mlp_module, device):
        super().__init__()
        self.mlp = mlp_module
        self.device = device

    def forward(self, theta, x, y):
        return (self.mlp(x + theta) - y).mean(dim=1)


# Backward-compat alias (old call sites that import GPT2MLPDeltaActs still work)
GPT2MLPDeltaActs = MLPDeltaActs


class LinearDCT:
    """
    Fits a low-rank Jacobian of the delta-activations function, then SVDs it
    to find the top-K causal input directions for a single MLP layer.

    V : (d_model, num_factors) project MLP inputs through this to get causal features.
    dct = LinearDCT(num_factors=64)
    U, V = dct.fit(delta_fn, X, Y)
    """
    def __init__(self, num_factors=64):
        self.num_factors = num_factors
        self.V = None
        self.U = None

    def fit(self, delta_fn, X, Y, dim_output_projection=64, batch_size=1, factor_batch_size=16):
        """
        X : (n_train, seq_len, d_model) MLP inputs
        Y : (n_train, seq_len, d_model) MLP outputs (unperturbed)
        """
        assert dim_output_projection >= self.num_factors

        d_model = X.shape[2]
        dev = delta_fn.device

        delta_vmap = vmap(delta_fn, in_dims=(1, None, None), out_dims=2,
                          chunk_size=factor_batch_size)

        U_rand = F.normalize(torch.randn(d_model, dim_output_projection), dim=0).to(dev).to(X.dtype)
        V0 = torch.zeros(d_model, dim_output_projection, device=dev, dtype=X.dtype)

        def vjp_single(u, v, X_b, Y_b):
            output, vjp_fn = vjp(lambda _v: delta_fn(_v, X_b, Y_b), v)
            with torch.no_grad():
                udots = output @ u
            return udots, output.detach(), vjp_fn(u.expand(X_b.shape[0], -1))[0].detach()

        vjp_batch = vmap(lambda u, v, X_b, Y_b: vjp_single(u, v, X_b, Y_b),
                         in_dims=(1, 1, None, None), out_dims=(1, 2, 1),
                         chunk_size=factor_batch_size)

        print("  Computing Jacobian (projected VJPs)...")
        J_avg = StreamingAverage()
        n_batches = max(1, (X.shape[0] + batch_size - 1) // batch_size)
        with torch.no_grad():
            for batch_idx, t in enumerate(
                tqdm(range(0, X.shape[0], batch_size), total=n_batches,
                     desc="    Jacobian batches", unit="batch", leave=False),
                start=1,
            ):
                x_b = X[t:t + batch_size].to(dev)
                y_b = Y[t:t + batch_size].to(dev)
                _, _, J_batch = vjp_batch(U_rand, V0, x_b, y_b)
                J_avg.update(J_batch.t().unsqueeze(0))

        J = J_avg.get()

        print("  SVD...")
        _, _, Vh = torch.linalg.svd(J.float(), full_matrices=False)
        self.V = Vh[:self.num_factors].T.to(J.dtype).contiguous()  # (d_model, num_factors)

        print("  Computing output directions...")
        U_avg = StreamingAverage()
        with torch.no_grad():
            for batch_idx, b in enumerate(
                tqdm(range(0, X.shape[0], batch_size), total=n_batches,
                     desc="    Output-dir batches", unit="batch", leave=False),
                start=1,
            ):
                x_b = X[b:b + batch_size].to(dev)
                y_b = Y[b:b + batch_size].to(dev)
                U_batch = delta_vmap(self.V, x_b, y_b)
                U_avg.update(U_batch)
        self.U = F.normalize(U_avg.get(), dim=0)

        return self.U, self.V
