"""
SPECTRE baseline: robust covariance estimation for backdoor detection.

Given a set of labeled clean document activations, computes the
Mahalanobis distance of each document from the clean centroid.
Documents far from the clean distribution are flagged as suspicious.

Reference: Hayase et al. (2021), "SPECTRE: Defending Against Backdoor
Attacks Using Robust Statistics", ICML 2021.
"""

import numpy as np
from typing import Dict, List, Optional

from .dct_config import SPECTRE_N_CLEAN, SEED


def spectre_scores(
    activations: np.ndarray,
    clean_indices: List[int],
    method: str = "l2",
) -> np.ndarray:
    """Score documents using SPECTRE-style robust covariance estimation.

    Args:
        activations: [n_docs, d] activation matrix
        clean_indices: indices of labeled clean documents (typically 50)
        method: 'l2' (L2 distance from clean mean) or 'mahalanobis'

    Returns:
        scores: [n_docs] higher = more suspicious
    """
    n_docs, d = activations.shape
    clean_acts = activations[clean_indices]  # [n_clean, d]
    clean_mean = clean_acts.mean(axis=0)     # [d]

    if method == "l2":
        # Simple L2 distance from clean centroid
        diffs = activations - clean_mean[np.newaxis, :]
        scores = np.linalg.norm(diffs, axis=1)

    elif method == "mahalanobis":
        # Robust covariance via shrinkage
        clean_centered = clean_acts - clean_mean
        n_clean = len(clean_indices)

        # Ledoit-Wolf shrinkage for stability
        sample_cov = (clean_centered.T @ clean_centered) / max(n_clean - 1, 1)

        # Regularize: shrink toward diagonal
        trace = np.trace(sample_cov) / d
        alpha = 0.1  # shrinkage parameter
        cov_reg = (1 - alpha) * sample_cov + alpha * trace * np.eye(d)

        # Pseudo-inverse (d may be > n_clean, making cov singular)
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to L2
            diffs = activations - clean_mean[np.newaxis, :]
            return np.linalg.norm(diffs, axis=1)

        diffs = activations - clean_mean[np.newaxis, :]
        # Mahalanobis: sqrt(diff^T @ cov_inv @ diff)
        scores = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))

    elif method == "spectral":
        # QUE (Quantum Unsupervised Ensemble) inspired scoring
        # Project onto top principal components of clean distribution
        clean_centered = clean_acts - clean_mean
        U, S, Vt = np.linalg.svd(clean_centered, full_matrices=False)
        # Keep top-k components
        k = min(50, len(clean_indices) - 1, d)
        Vk = Vt[:k].T  # [d, k]

        # Whiten the data using clean distribution
        diffs = activations - clean_mean[np.newaxis, :]
        projected = diffs @ Vk  # [n_docs, k]
        # Normalize by singular values
        projected = projected / (S[:k] / np.sqrt(max(len(clean_indices) - 1, 1)) + 1e-10)
        scores = np.linalg.norm(projected, axis=1)

    else:
        raise ValueError(f"Unknown SPECTRE method: {method}")

    return scores


def spectre_scores_multilayer(
    activations: Dict[int, np.ndarray],
    clean_indices: List[int],
    method: str = "l2",
) -> np.ndarray:
    """SPECTRE scoring across multiple layers, averaged.

    Args:
        activations: {layer_idx: [n_docs, d]} activations
        clean_indices: indices of labeled clean documents
        method: scoring method passed to spectre_scores

    Returns:
        scores: [n_docs] averaged across layers
    """
    all_scores = []
    for layer_idx in sorted(activations.keys()):
        acts = activations[layer_idx]
        s = spectre_scores(acts, clean_indices, method=method)
        # Normalize per layer to [0, 1] range
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min)
        all_scores.append(s)

    return np.mean(all_scores, axis=0)
