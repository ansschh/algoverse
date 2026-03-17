"""
Step 5: Build FAISS indexes, search, compute metrics.

For each FAISS config and each selected feature:
  - Query: w_f (SAE encoder weight vector)
  - Index: activation vectors X (10M × d)
  - Compute recall, purity, score gap, rank correlation, timing
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import pandas as pd
import faiss
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.stats import spearmanr

from .config import (
    MODELS, ModelSpec, DATA_DIR, TOTAL_FEATURES,
    K_SEARCH, K_EVALS, TRAIN_SUBSET, ADD_CHUNK_SIZE,
    OMP_THREADS, WARMUP_QUERIES, LEXICON,
    NUM_LEXICON_FEATURES,
    FAISSConfig, get_faiss_grid,
)
from .utils import (
    model_dir, activation_path, load_memmap, iter_chunks, token_ids_path,
)


# Set FAISS threading
faiss.omp_set_num_threads(OMP_THREADS)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _index_build_key(cfg: FAISSConfig) -> tuple:
    """Key that identifies the index structure (build-time params only, not search-time)."""
    if cfg.index_type == "FlatIP":
        return ("FlatIP",)
    elif cfg.index_type == "IVFFlat":
        return ("IVFFlat", cfg.nlist)
    elif cfg.index_type == "IVFPQ":
        return ("IVFPQ", cfg.nlist, cfg.m_pq)
    elif cfg.index_type == "HNSWFlat":
        return ("HNSWFlat", cfg.M_hnsw)
    return (cfg.index_type,)


def build_index_structure(index_type: str, d: int, train_data: np.ndarray,
                          act_memmap, T: int, nlist: int = 0,
                          m_pq: int = 0, nbits: int = 8,
                          M_hnsw: int = 0, efConstruction: int = 200,
                          ) -> Tuple[faiss.Index, float, float, float]:
    """
    Build a FAISS index structure (without search-time params like nprobe/efSearch).

    Returns:
        (index, train_time_s, add_time_s, build_time_s)
    """
    t_build_start = time.time()
    train_time = 0.0

    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(d)

    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        t0 = time.time()
        index.train(train_data)
        train_time = time.time() - t0

    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq,
                                 nbits, faiss.METRIC_INNER_PRODUCT)
        t0 = time.time()
        index.train(train_data)
        train_time = time.time() - t0

    elif index_type == "HNSWFlat":
        index = faiss.IndexHNSWFlat(d, M_hnsw, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruction

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Add vectors in chunks
    t_add_start = time.time()
    for start, end in iter_chunks(T, ADD_CHUNK_SIZE):
        chunk = np.array(act_memmap[start:end], dtype=np.float32)
        index.add(chunk)
    add_time = time.time() - t_add_start

    build_time = time.time() - t_build_start
    return index, train_time, add_time, build_time


def _set_search_params(index: faiss.Index, cfg: FAISSConfig):
    """Set search-time parameters (nprobe or efSearch) on an already-built index."""
    if cfg.index_type in ("IVFFlat", "IVFPQ"):
        index.nprobe = cfg.nprobe
    elif cfg.index_type == "HNSWFlat":
        index.hnsw.efSearch = cfg.efSearch


def get_index_size_bytes(index: faiss.Index) -> int:
    """Measure serialized index size in bytes."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".faissindex")
    tmp.close()
    try:
        faiss.write_index(index, tmp.name)
        size = os.path.getsize(tmp.name)
    finally:
        os.unlink(tmp.name)
    return size


# ---------------------------------------------------------------------------
# Search with timing
# ---------------------------------------------------------------------------

def timed_search(index: faiss.Index, queries: np.ndarray, k: int,
                 warmup: int = WARMUP_QUERIES) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Search with per-query latency measurement.

    Returns:
        (scores, ids, mean_ms, p50_ms, p95_ms)
    """
    Q = queries.shape[0]

    # Warmup
    n_warmup = min(warmup, Q)
    if n_warmup > 0:
        index.search(queries[:n_warmup], k)

    # Timed search per query
    all_scores = np.zeros((Q, k), dtype=np.float32)
    all_ids = np.zeros((Q, k), dtype=np.int64)
    latencies = []

    for i in range(Q):
        q = queries[i:i+1]
        t0 = time.time()
        s, ids = index.search(q, k)
        latencies.append((time.time() - t0) * 1000)  # ms
        all_scores[i] = s[0]
        all_ids[i] = ids[0]

    latencies = np.array(latencies)
    return (all_scores, all_ids,
            float(np.mean(latencies)),
            float(np.percentile(latencies, 50)),
            float(np.percentile(latencies, 95)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(exact_ids: np.ndarray, approx_ids: np.ndarray, k: int) -> float:
    """Top-k recall: |exact_top_k ∩ approx_top_k| / k."""
    exact_set = set(exact_ids[:k].tolist())
    approx_set = set(approx_ids[:k].tolist())
    return len(exact_set & approx_set) / k


def top10_in_top100_recall(exact_top10: np.ndarray, approx_top100: np.ndarray) -> float:
    """Hard metric: |exact_top10 ∩ approx_top100| / 10."""
    exact_set = set(exact_top10.tolist())
    approx_set = set(approx_top100.tolist())
    return len(exact_set & approx_set) / 10


def concept_purity(approx_top10_ids: np.ndarray, token_ids: np.ndarray,
                   lexicon_token_ids: set) -> float:
    """Fraction of approx top-10 that are lexicon tokens."""
    count = 0
    for idx in approx_top10_ids[:10]:
        if idx >= 0 and idx < len(token_ids):
            if token_ids[idx] in lexicon_token_ids:
                count += 1
    return count / 10


def score_gap(exact_top10_scores: np.ndarray, approx_top10_ids: np.ndarray,
              W_f: np.ndarray, b_f: float, act_memmap, d: int) -> float:
    """
    Soft metric: mean(s_f(approx_top10)) - mean(s_f(exact_top10)).

    Negative = approximate is worse than exact (expected).
    """
    # Compute scores for approx top-10
    approx_ids = approx_top10_ids[:10]
    valid = approx_ids[(approx_ids >= 0)]
    if len(valid) == 0:
        return float('nan')
    approx_vecs = np.array(act_memmap[valid], dtype=np.float32)
    approx_scores = approx_vecs @ W_f + b_f

    exact_mean = float(np.mean(exact_top10_scores[:10]))
    approx_mean = float(np.mean(approx_scores))
    return approx_mean - exact_mean


def spearman_top100(exact_top100_ids: np.ndarray, exact_top100_scores: np.ndarray,
                    approx_top100_ids: np.ndarray, approx_top100_scores: np.ndarray) -> float:
    """
    Spearman rank correlation on top-100.
    Only considers IDs that appear in both exact and approx top-100.
    """
    exact_dict = {int(idx): rank for rank, idx in enumerate(exact_top100_ids[:100])}
    approx_dict = {int(idx): rank for rank, idx in enumerate(approx_top100_ids[:100])}

    common = set(exact_dict.keys()) & set(approx_dict.keys())
    if len(common) < 3:
        return float('nan')

    exact_ranks = [exact_dict[idx] for idx in common]
    approx_ranks = [approx_dict[idx] for idx in common]
    corr, _ = spearmanr(exact_ranks, approx_ranks)
    return float(corr)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model_name: str,
    data_dir: str = DATA_DIR,
    configs: Optional[List[FAISSConfig]] = None,
) -> str:
    """
    Run FAISS benchmark for a model across all configs and features.

    Args:
        model_name: Key into MODELS dict.
        data_dir: Root data directory.
        configs: FAISS configs to benchmark (default: full grid).

    Returns:
        Path to results CSV.
    """
    if configs is None:
        configs = get_faiss_grid()

    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)
    out_csv = os.path.join(mdir, "faiss_results.csv")

    # Load ground truth
    gt_path = os.path.join(mdir, "ground_truth.npz")
    assert os.path.exists(gt_path), f"Ground truth not found: {gt_path}"
    gt = np.load(gt_path)

    # Load feature metadata
    meta_path = os.path.join(mdir, "selected_features.json")
    with open(meta_path) as f:
        feat_meta = json.load(f)

    # Load feature weights (queries) and biases
    W = np.load(os.path.join(mdir, "feature_weights.npy"))  # [F, d]
    b = np.load(os.path.join(mdir, "feature_biases.npy"))   # [F]
    F = W.shape[0]
    d = W.shape[1]
    queries = np.ascontiguousarray(W, dtype=np.float32)      # [F, d]

    # Load activations and token IDs
    tid_path = token_ids_path(data_dir)
    token_ids = np.load(tid_path)
    T = len(token_ids)
    act_path = activation_path(data_dir, model_name)
    acts = load_memmap(act_path, (T, d))

    # Lexicon token IDs for purity metric
    lexicon_token_ids = set(feat_meta.get("lexicon_token_ids", []))

    # Training subset
    n_train = min(TRAIN_SUBSET, T)
    train_indices = np.linspace(0, T - 1, n_train, dtype=int)
    print(f"[bench] Loading {n_train:,} training vectors ...")
    train_data = np.array(acts[train_indices], dtype=np.float32)

    # Resume: load existing results
    done_keys = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        for _, row in existing.iterrows():
            key = (row["config_label"], int(row["feature_idx"]))
            done_keys.add(key)
        print(f"[bench] Found {len(done_keys)} existing results, resuming.")

    # Group configs by index structure (build once, search with multiple nprobe/efSearch)
    from collections import OrderedDict
    build_groups = OrderedDict()  # build_key -> [cfg, cfg, ...]
    for cfg in configs:
        bk = _index_build_key(cfg)
        build_groups.setdefault(bk, []).append(cfg)

    n_builds = len(build_groups)
    print(f"[bench] Model: {model_name}, T={T:,}, d={d}, F={F}")
    print(f"[bench] Configs: {len(configs)}, Unique indexes to build: {n_builds}")

    build_idx = 0
    for build_key, group_cfgs in build_groups.items():
        build_idx += 1

        # Check if ALL configs in this group are fully done
        all_group_done = all(
            all((cfg.label, f) in done_keys for f in range(F))
            for cfg in group_cfgs
        )
        if all_group_done:
            labels = [c.label for c in group_cfgs]
            print(f"\n[build {build_idx}/{n_builds}] {build_key} — SKIP (all done: {labels})")
            continue

        ref = group_cfgs[0]  # reference config for build params
        labels = [c.label for c in group_cfgs]
        print(f"\n[build {build_idx}/{n_builds}] Building {build_key} for {len(group_cfgs)} search configs: {labels}")

        # Verify PQ m divides d
        if ref.index_type == "IVFPQ" and d % ref.m_pq != 0:
            print(f"  WARNING: m_pq={ref.m_pq} does not divide d={d}. Skipping.")
            continue

        # Build index ONCE for this group
        try:
            index, train_time, add_time, build_time = build_index_structure(
                ref.index_type, d, train_data, acts, T,
                nlist=ref.nlist, m_pq=ref.m_pq, nbits=ref.nbits,
                M_hnsw=ref.M_hnsw, efConstruction=ref.efConstruction,
            )
        except Exception as e:
            print(f"  ERROR building index: {e}")
            continue

        print(f"  Built in {build_time:.1f}s (train={train_time:.1f}s, add={add_time:.1f}s)")

        # Get index size once
        try:
            index_bytes = get_index_size_bytes(index)
        except Exception:
            index_bytes = 0

        # Search with each config's search params
        for cfg in group_cfgs:
            if all((cfg.label, f) in done_keys for f in range(F)):
                print(f"  {cfg.label} — SKIP (all done)")
                continue

            _set_search_params(index, cfg)

            scores_all, ids_all, mean_ms, p50_ms, p95_ms = timed_search(
                index, queries, K_SEARCH,
            )
            print(f"  {cfg.label}: mean={mean_ms:.1f}ms, p50={p50_ms:.1f}ms, p95={p95_ms:.1f}ms")

            # Per-feature metrics
            rows = []
            for f in range(F):
                if (cfg.label, f) in done_keys:
                    continue

                fid = feat_meta["selected_feature_ids"][f]
                kind = "lexicon" if f < NUM_LEXICON_FEATURES else "random"

                approx_ids = ids_all[f]
                approx_scores = scores_all[f]

                exact_top10_ids = gt["exact_top10_ids"][f]
                exact_top10_scores = gt["exact_top10_scores"][f]
                exact_top100_ids = gt["exact_top100_ids"][f]
                exact_top100_scores = gt["exact_top100_scores"][f]

                t10_in_t100 = top10_in_top100_recall(exact_top10_ids, approx_ids[:100])
                recall_10 = recall_at_k(exact_top10_ids, approx_ids, 10)
                recall_100 = recall_at_k(exact_top100_ids, approx_ids, 100)

                purity = float('nan')
                if kind == "lexicon":
                    purity = concept_purity(approx_ids, token_ids, lexicon_token_ids)

                gap = score_gap(
                    exact_top10_scores, approx_ids,
                    W[f], b[f], acts, d,
                )

                sp_corr = spearman_top100(
                    exact_top100_ids, exact_top100_scores,
                    approx_ids[:100], approx_scores[:100],
                )

                row = {
                    "model": model_name,
                    "feature_idx": f,
                    "feature_id": fid,
                    "feature_kind": kind,
                    "config_label": cfg.label,
                    "index_type": cfg.index_type,
                    "nlist": cfg.nlist,
                    "nprobe": cfg.nprobe,
                    "M_hnsw": cfg.M_hnsw,
                    "efSearch": cfg.efSearch,
                    "m_pq": cfg.m_pq,
                    "top10_in_top100_recall": round(t10_in_t100, 4),
                    "recall_at_10": round(recall_10, 4),
                    "recall_at_100": round(recall_100, 4),
                    "concept_purity": round(purity, 4) if not np.isnan(purity) else None,
                    "score_gap": round(gap, 6) if not np.isnan(gap) else None,
                    "spearman_top100": round(sp_corr, 4) if not np.isnan(sp_corr) else None,
                    "ms_query_mean": round(mean_ms, 3),
                    "ms_query_p50": round(p50_ms, 3),
                    "ms_query_p95": round(p95_ms, 3),
                    "build_time_s": round(build_time, 2),
                    "train_time_s": round(train_time, 2),
                    "add_time_s": round(add_time, 2),
                    "index_size_bytes": index_bytes,
                    "index_size_mb": round(index_bytes / (1024**2), 1),
                }
                rows.append(row)

            if rows:
                row_df = pd.DataFrame(rows)
                header = not os.path.exists(out_csv)
                row_df.to_csv(out_csv, mode="a", header=header, index=False)
                print(f"    Saved {len(rows)} rows")

        # Free index memory after all search variants are done
        del index

    print(f"\n[bench] Complete. Results at {out_csv}")

    # Print summary
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        print(f"[bench] Total rows: {len(df)}")
        print("\n  Mean metrics by index_type:")
        summary = df.groupby("index_type").agg({
            "top10_in_top100_recall": "mean",
            "recall_at_10": "mean",
            "recall_at_100": "mean",
            "ms_query_p95": "mean",
            "index_size_mb": "first",
        }).round(4)
        print(summary.to_string(index=True))

    return out_csv


def dump_examples(model_name: str, data_dir: str = DATA_DIR, n_features: int = 3):
    """
    Dump example top-10 decoded tokens for qualitative comparison.
    Prints exact vs approximate top-10 for selected features × configs.
    """
    from transformers import AutoTokenizer

    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)

    # Load metadata
    with open(os.path.join(mdir, "selected_features.json")) as f:
        feat_meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    token_ids = np.load(token_ids_path(data_dir))
    gt = np.load(os.path.join(mdir, "ground_truth.npz"))
    df = pd.read_csv(os.path.join(mdir, "faiss_results.csv"))

    # Select features: first 2 lexicon + 1 random
    feature_idxs = list(range(min(2, NUM_LEXICON_FEATURES)))
    if TOTAL_FEATURES > NUM_LEXICON_FEATURES:
        feature_idxs.append(NUM_LEXICON_FEATURES)  # first random
    feature_idxs = feature_idxs[:n_features]

    # Select configs for comparison
    config_labels = ["FlatIP"]
    # Best IVFFlat
    ivf = df[df["index_type"] == "IVFFlat"]
    if len(ivf) > 0:
        best_ivf = ivf.groupby("config_label")["top10_in_top100_recall"].mean().idxmax()
        config_labels.append(best_ivf)
    # Worst IVFPQ (smallest m)
    pq = df[df["index_type"] == "IVFPQ"]
    if len(pq) > 0:
        worst_pq = pq.groupby("config_label")["top10_in_top100_recall"].mean().idxmin()
        config_labels.append(worst_pq)

    print(f"\n{'='*80}")
    print("QUALITATIVE EXAMPLES: Top-10 Decoded Tokens")
    print(f"{'='*80}")

    for fi in feature_idxs:
        fid = feat_meta["selected_feature_ids"][fi]
        kind = "lexicon" if fi < NUM_LEXICON_FEATURES else "random"
        print(f"\nFeature {fid} ({kind}):")

        # Exact top-10
        exact_ids = gt["exact_top10_ids"][fi]
        exact_scores = gt["exact_top10_scores"][fi]
        tokens = [tokenizer.decode([token_ids[idx]]) for idx in exact_ids[:10]]
        print(f"  EXACT:  {tokens}")
        print(f"          scores: {[f'{s:.3f}' for s in exact_scores[:10]]}")

        # Approximate top-10 for each config
        for cl in config_labels:
            rows = df[(df["config_label"] == cl) & (df["feature_idx"] == fi)]
            if len(rows) == 0:
                continue
            row = rows.iloc[0]
            r10 = row["top10_in_top100_recall"]
            print(f"  {cl}:  recall={r10:.3f}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FAISS benchmark for SAE features")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--dump-examples", action="store_true")
    args = parser.parse_args()

    if args.dump_examples:
        dump_examples(args.model, args.data_dir)
    else:
        run_benchmark(args.model, args.data_dir)
