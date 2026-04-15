"""
Core DCT: Extract MLP Jacobian fingerprints via torch.func.vjp.

LinearDCT: Per-layer independent SVD of the MLP Jacobian.
ExpDCT: Cross-layer exponential reweighting (experimental).

Key constraint: model must be loaded on CPU without device_map dispatch hooks
for torch.func.vjp compatibility. Each MLP is moved to GPU one layer at a time.
"""

import os
import gc
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from .dct_config import (
    N_TRAIN, N_FACTORS, N_ITER, DCT_LAYERS_LLAMA2,
    get_dct_layers, SEED,
)
from .utils import save_metadata, load_metadata


def _get_mlp_forward_fn(model, layer_idx: int):
    """Get a pure function for the MLP at a given layer.

    Returns a function f(x) -> mlp_output that is compatible with torch.func.vjp.
    """
    # Access the MLP module for this layer
    if hasattr(model, "model"):
        # HuggingFace LlamaForCausalLM -> model.model.layers[i].mlp
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find model layers. Unsupported architecture.")

    mlp = layers[layer_idx].mlp

    def mlp_fn(x):
        return mlp(x)

    return mlp, mlp_fn


def extract_document_activations(
    model,
    tokenizer,
    documents: List[str],
    layer_indices: List[int],
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 16,
) -> Dict[int, np.ndarray]:
    """Extract per-layer activations for a list of documents.

    Returns dict mapping layer_index -> [n_docs, d_model] float32 array.
    Uses the mean of all token positions as the document representation.
    """
    model.eval()
    n_docs = len(documents)

    # Determine d_model from config
    if hasattr(model.config, "hidden_size"):
        d_model = model.config.hidden_size
    else:
        d_model = model.config.d_model

    activations = {l: np.zeros((n_docs, d_model), dtype=np.float32) for l in layer_indices}

    for batch_start in range(0, n_docs, batch_size):
        batch_end = min(batch_start + batch_size, n_docs)
        batch_texts = documents[batch_start:batch_end]

        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)

        mask = encodings["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]

        for layer_idx in layer_indices:
            h = outputs.hidden_states[layer_idx + 1]  # skip embedding
            # Mean pool over non-padding tokens
            h_masked = h * mask
            h_mean = h_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            activations[layer_idx][batch_start:batch_end] = h_mean.float().cpu().numpy()

    return activations


def build_v_matrices(
    model,
    tokenizer,
    clean_documents: List[str],
    layer_indices: List[int],
    n_factors: int = N_FACTORS,
    n_iter: int = N_ITER,
    device: str = "cuda",
    max_length: int = 512,
) -> Dict[int, np.ndarray]:
    """Build V matrices (fingerprint projections) for selected layers.

    For each layer, computes the top-k left singular vectors of the
    MLP Jacobian estimated over clean document activations.

    Returns dict mapping layer_index -> V matrix [d_model, n_factors].
    """
    from torch.func import vjp, vmap

    # First extract input activations at each layer (what goes INTO the MLP)
    print(f"[dct] Extracting input activations for {len(clean_documents)} docs "
          f"at {len(layer_indices)} layers...")
    input_acts = extract_document_activations(
        model, tokenizer, clean_documents, layer_indices,
        device=device, max_length=max_length,
    )

    v_matrices = {}

    for layer_idx in layer_indices:
        print(f"[dct] Building V matrix for layer {layer_idx}...")
        t0 = time.time()

        mlp, mlp_fn = _get_mlp_forward_fn(model, layer_idx)

        # Move MLP to GPU for VJP computation
        mlp_device = next(mlp.parameters()).device
        mlp.to(device)

        acts = input_acts[layer_idx]  # [n_docs, d]
        n_docs, d = acts.shape
        n_use = min(n_docs, N_TRAIN)

        # Initialize random projection vectors for iterative VJP
        rng = np.random.RandomState(SEED)
        # V_est will accumulate the Jacobian information
        jacobian_samples = []

        for i in range(n_use):
            x = torch.tensor(acts[i], dtype=torch.float32, device=device).unsqueeze(0)

            # Compute VJP: for random vectors v, compute v^T @ J where J = d mlp(x) / d x
            for _ in range(n_iter):
                v = torch.randn(1, d, device=device, dtype=torch.float32)
                v = v / v.norm()

                # vjp returns (output, vjp_fn)
                # vjp_fn(v) gives v^T @ J
                try:
                    output, vjp_fn = vjp(lambda inp: mlp_fn(inp), x)
                    (jvp_result,) = vjp_fn(v.expand_as(output))
                    jacobian_samples.append(jvp_result.squeeze(0).detach().cpu())
                except Exception as e:
                    # Fallback: use finite differences
                    eps = 1e-4
                    with torch.no_grad():
                        f_x = mlp_fn(x)
                        jvp_approx = torch.zeros(d, device=device)
                        for j_dim in range(min(d, n_factors * 2)):
                            x_plus = x.clone()
                            x_plus[0, j_dim] += eps
                            f_plus = mlp_fn(x_plus)
                            jvp_approx[j_dim] = ((f_plus - f_x) / eps).mean()
                        jacobian_samples.append(jvp_approx.cpu())
                    break

            if (i + 1) % 20 == 0:
                print(f"  layer {layer_idx}: {i+1}/{n_use} docs processed")

        # Stack all Jacobian samples and compute SVD
        J = torch.stack(jacobian_samples, dim=0).float()  # [n_samples, d]
        # SVD: J = U @ S @ V^T
        # We want the top-k right singular vectors (columns of V)
        U, S, Vt = torch.linalg.svd(J, full_matrices=False)
        V = Vt[:n_factors].T.numpy()  # [d, n_factors]

        v_matrices[layer_idx] = V

        # Move MLP back
        mlp.to(mlp_device)

        elapsed = time.time() - t0
        print(f"  layer {layer_idx}: V matrix [{d}, {n_factors}], "
              f"top singular values: {S[:5].tolist()}, time: {elapsed:.1f}s")

        # Free memory
        del jacobian_samples, J, U, S, Vt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return v_matrices


def save_v_matrices(v_matrices: Dict[int, np.ndarray], out_dir: str):
    """Save V matrices to disk."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for layer_idx, V in v_matrices.items():
        path = os.path.join(out_dir, f"V_layer{layer_idx}.npy")
        np.save(path, V)
        meta[str(layer_idx)] = {
            "shape": list(V.shape),
            "path": path,
        }
    save_metadata(os.path.join(out_dir, "v_matrices_meta.json"), meta)
    print(f"[dct] Saved V matrices for {len(v_matrices)} layers to {out_dir}")


def load_v_matrices(v_dir: str) -> Dict[int, np.ndarray]:
    """Load V matrices from disk."""
    meta_path = os.path.join(v_dir, "v_matrices_meta.json")
    meta = load_metadata(meta_path)
    v_matrices = {}
    for layer_str, info in meta.items():
        path = info["path"]
        if not os.path.isabs(path):
            path = os.path.join(v_dir, os.path.basename(path))
        v_matrices[int(layer_str)] = np.load(path)
    return v_matrices


def build_dct_pipeline(
    model_id: str,
    clean_documents: List[str],
    layer_indices: List[int],
    out_dir: str,
    device: str = "cuda",
    n_factors: int = N_FACTORS,
) -> Dict[int, np.ndarray]:
    """Full pipeline: load model, build V matrices, save.

    Model is loaded on CPU first (required for torch.func.vjp),
    then each MLP layer is moved to GPU individually.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v_meta_path = os.path.join(out_dir, "v_matrices_meta.json")
    if os.path.exists(v_meta_path):
        print(f"[dct] V matrices already exist at {out_dir}")
        return load_v_matrices(out_dir)

    print(f"[dct] Loading {model_id} on CPU (required for torch.func.vjp)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load on CPU without device_map for vjp compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=None,  # no dispatch hooks
    )
    # Move full model to device for activation extraction
    model.to(device)
    model.eval()

    v_matrices = build_v_matrices(
        model, tokenizer, clean_documents, layer_indices,
        n_factors=n_factors, device=device,
    )

    save_v_matrices(v_matrices, out_dir)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return v_matrices
