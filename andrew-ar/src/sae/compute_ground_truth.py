"""
Step 4: Exact ground truth via streaming dot product through memmap.

For each selected feature f:
    s_f(i) = w_f^T @ X[i] + b_f   (pre-activation, Option A)

Streams through the activation memmap in chunks, maintains top-K via
argpartition+merge, and saves exact top-{10, 100, 1000} IDs and scores.
"""

import os
import time
import json
import numpy as np

from .config import (
    MODELS, DATA_DIR, GT_TOP_KS, GT_CHUNK_SIZE, TOTAL_FEATURES,
)
from .utils import (
    model_dir, activation_path, load_memmap, iter_chunks, token_ids_path,
)


def _topk_merge(scores: np.ndarray, indices: np.ndarray,
                new_scores: np.ndarray, new_indices: np.ndarray,
                k: int) -> tuple:
    """Merge two (scores, indices) arrays and keep top-k by score (descending)."""
    all_scores = np.concatenate([scores, new_scores])
    all_indices = np.concatenate([indices, new_indices])
    if len(all_scores) <= k:
        order = np.argsort(-all_scores)
        return all_scores[order], all_indices[order]
    # argpartition for top-k, then sort those k
    part_idx = np.argpartition(-all_scores, k)[:k]
    top_scores = all_scores[part_idx]
    top_indices = all_indices[part_idx]
    order = np.argsort(-top_scores)
    return top_scores[order], top_indices[order]


def compute_ground_truth(
    model_name: str,
    data_dir: str = DATA_DIR,
    chunk_size: int = GT_CHUNK_SIZE,
) -> str:
    """
    Compute exact top-K ground truth for all selected features.

    Args:
        model_name: Key into MODELS dict.
        data_dir: Root data directory.
        chunk_size: Tokens per chunk for streaming computation.

    Returns:
        Path to ground_truth.npz.
    """
    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)

    out_path = os.path.join(mdir, "ground_truth.npz")
    if os.path.exists(out_path):
        print(f"[gt] Ground truth already exists at {out_path}. Skipping.")
        return out_path

    # Load feature weights and biases
    weights_path = os.path.join(mdir, "feature_weights.npy")
    biases_path = os.path.join(mdir, "feature_biases.npy")
    meta_path = os.path.join(mdir, "selected_features.json")

    assert os.path.exists(weights_path), f"Feature weights not found: {weights_path}"
    W = np.load(weights_path)   # [F, d]
    b = np.load(biases_path)    # [F]
    F = W.shape[0]
    d = W.shape[1]

    assert d == spec.d_model, f"Weight dim {d} != d_model {spec.d_model}"
    assert F == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES} features, got {F}"

    # Load activations memmap
    act_path = activation_path(data_dir, model_name)
    tid_path = token_ids_path(data_dir)
    token_ids = np.load(tid_path)
    T = len(token_ids)

    acts = load_memmap(act_path, (T, d))
    K_max = max(GT_TOP_KS)

    print(f"[gt] Model: {model_name}, T={T:,}, d={d}, F={F}, K_max={K_max}")
    print(f"[gt] Chunk size: {chunk_size:,} ({chunk_size * d * 2 / 1e6:.0f} MB per chunk)")

    # Initialize top-K tracking per feature
    top_scores = [np.array([], dtype=np.float32) for _ in range(F)]
    top_indices = [np.array([], dtype=np.int64) for _ in range(F)]

    t0 = time.time()
    n_chunks = (T + chunk_size - 1) // chunk_size

    for ci, (start, end) in enumerate(iter_chunks(T, chunk_size)):
        # Load chunk and cast to float32
        chunk = np.array(acts[start:end], dtype=np.float32)  # [C, d]

        # Batch matrix multiply: scores = chunk @ W.T + b → [C, F]
        scores = chunk @ W.T + b[np.newaxis, :]

        # For each feature, merge chunk top-K with running top-K
        for f in range(F):
            f_scores = scores[:, f]
            # Get top-K from this chunk
            chunk_k = min(K_max, len(f_scores))
            if chunk_k < len(f_scores):
                part_idx = np.argpartition(-f_scores, chunk_k)[:chunk_k]
            else:
                part_idx = np.arange(len(f_scores))

            chunk_top_scores = f_scores[part_idx]
            chunk_top_indices = (start + part_idx).astype(np.int64)

            top_scores[f], top_indices[f] = _topk_merge(
                top_scores[f], top_indices[f],
                chunk_top_scores, chunk_top_indices,
                K_max,
            )

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            elapsed = time.time() - t0
            print(f"  chunk {ci + 1}/{n_chunks}  ({elapsed:.1f}s)")

    # Build output arrays for each K level
    results = {}
    for K in GT_TOP_KS:
        ids_arr = np.zeros((F, K), dtype=np.int64)
        scores_arr = np.zeros((F, K), dtype=np.float32)
        for f in range(F):
            k_actual = min(K, len(top_scores[f]))
            ids_arr[f, :k_actual] = top_indices[f][:k_actual]
            scores_arr[f, :k_actual] = top_scores[f][:k_actual]
        results[f"exact_top{K}_ids"] = ids_arr
        results[f"exact_top{K}_scores"] = scores_arr

    np.savez_compressed(out_path, **results)
    elapsed_total = time.time() - t0
    print(f"[gt] Done in {elapsed_total:.1f}s. Saved to {out_path}")

    # Print summary
    with open(meta_path) as fp:
        meta = json.load(fp)
    for f in range(F):
        fid = meta["selected_feature_ids"][f]
        kind = "lex" if f < 10 else "rnd"
        s_top = top_scores[f][0] if len(top_scores[f]) > 0 else 0
        s_10 = top_scores[f][9] if len(top_scores[f]) > 9 else 0
        print(f"  feature {fid:5d} ({kind}): top1_score={s_top:.4f}, top10_score={s_10:.4f}")

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute ground truth for SAE benchmark")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--chunk-size", type=int, default=GT_CHUNK_SIZE)
    args = parser.parse_args()
    compute_ground_truth(args.model, args.data_dir, args.chunk_size)
