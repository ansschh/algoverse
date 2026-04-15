"""
Score documents by projecting activations onto DCT V matrices.

Given V matrices (fingerprint projections) and document activations,
compute a per-document fingerprint score. Higher score = more likely poisoned.
"""

import os
import numpy as np
from typing import Dict, List, Optional

from .dct_config import DCT_BATCH_SIZE


def score_documents(
    activations: Dict[int, np.ndarray],
    v_matrices: Dict[int, np.ndarray],
    method: str = "l2_concat",
) -> np.ndarray:
    """Score documents using V matrix projections.

    Args:
        activations: {layer_idx: [n_docs, d_model]} per-layer activations
        v_matrices: {layer_idx: [d_model, n_factors]} V matrices
        method: scoring method
            'l2_concat': L2 norm of concatenated projections across layers (default)
            'l2_mean': mean of per-layer L2 norms
            'max_layer': max per-layer L2 norm

    Returns:
        scores: [n_docs] fingerprint scores (higher = more suspicious)
    """
    # Find common layers
    common_layers = sorted(set(activations.keys()) & set(v_matrices.keys()))
    if not common_layers:
        raise ValueError("No common layers between activations and V matrices")

    n_docs = next(iter(activations.values())).shape[0]
    layer_scores = []

    for layer_idx in common_layers:
        acts = activations[layer_idx]  # [n_docs, d]
        V = v_matrices[layer_idx]       # [d, n_factors]

        # Handle dimension mismatch (cross-model transfer)
        d_act = acts.shape[1]
        d_v = V.shape[0]
        if d_act != d_v:
            # Zero-pad or truncate
            if d_act < d_v:
                acts_padded = np.zeros((n_docs, d_v), dtype=acts.dtype)
                acts_padded[:, :d_act] = acts
                acts = acts_padded
            else:
                acts = acts[:, :d_v]

        # Project: projected[i] = V^T @ acts[i]
        projected = acts @ V  # [n_docs, n_factors]
        l2_norms = np.linalg.norm(projected, axis=1)  # [n_docs]
        layer_scores.append(l2_norms)

    layer_scores = np.array(layer_scores)  # [n_layers, n_docs]

    if method == "l2_concat":
        # Concatenate all projections and take L2 norm
        all_projected = []
        for layer_idx in common_layers:
            acts = activations[layer_idx]
            V = v_matrices[layer_idx]
            d_act, d_v = acts.shape[1], V.shape[0]
            if d_act < d_v:
                acts_padded = np.zeros((n_docs, d_v), dtype=acts.dtype)
                acts_padded[:, :d_act] = acts
                acts = acts_padded
            elif d_act > d_v:
                acts = acts[:, :d_v]
            all_projected.append(acts @ V)
        concat = np.concatenate(all_projected, axis=1)  # [n_docs, n_layers * n_factors]
        scores = np.linalg.norm(concat, axis=1)
    elif method == "l2_mean":
        scores = layer_scores.mean(axis=0)
    elif method == "max_layer":
        scores = layer_scores.max(axis=0)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return scores


def score_documents_from_files(
    model_id: str,
    documents: List[str],
    v_dir: str,
    layer_indices: List[int],
    device: str = "cuda",
    batch_size: int = DCT_BATCH_SIZE,
    method: str = "l2_concat",
) -> np.ndarray:
    """End-to-end: load model, extract activations, score with V matrices.

    Convenience function that handles model loading and activation extraction.
    """
    from .dct_jacobian import extract_document_activations, load_v_matrices
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, gc

    # Load V matrices
    v_matrices = load_v_matrices(v_dir)

    # Load model
    print(f"[score] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    # Extract activations
    activations = extract_document_activations(
        model, tokenizer, documents, layer_indices,
        device=device, batch_size=batch_size,
    )

    # Score
    scores = score_documents(activations, v_matrices, method=method)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores
