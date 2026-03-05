"""
FAISS Evaluation Engine.

Runs the full FAISS variant grid (FlatIP, IVFFlat, HNSWFlat, IVFPQ, OPQ_IVFPQ)
across two regimes (raw inner-product, normalized cosine-as-IP), computes all
metrics (IndexRecall@K, PoisonRecall@K, DecoyFP@K, latency, build time, memory),
and writes results to a canonical CSV.
"""

import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_EXACT = 1000         # Exact top-K for ground truth
K_EVALS = [10, 50, 100]  # K values for recall metrics
WARMUP_QUERIES = 5     # Warmup queries before timing
OMP_THREADS = 8        # FAISS thread count
TRAIN_SUBSET_MAX = 200_000  # Max docs for IVF/PQ training


@dataclass
class IndexConfig:
    """One FAISS index configuration to evaluate."""
    index_type: str            # FlatIP, IVFFlat, HNSWFlat, IVFPQ, OPQ_IVFPQ
    nlist: int = 0
    nprobe: int = 0
    M_hnsw: int = 0
    efSearch: int = 0
    efConstruction: int = 200
    m_pq: int = 0              # PQ sub-quantizers
    nbits: int = 8             # PQ bits per sub-quantizer

    @property
    def label(self) -> str:
        if self.index_type == "FlatIP":
            return "FlatIP"
        elif self.index_type == "IVFFlat":
            return f"IVFFlat(nlist={self.nlist},nprobe={self.nprobe})"
        elif self.index_type == "HNSWFlat":
            return f"HNSWFlat(M={self.M_hnsw},ef={self.efSearch})"
        elif self.index_type == "IVFPQ":
            return f"IVFPQ(nlist={self.nlist},nprobe={self.nprobe},m={self.m_pq})"
        elif self.index_type == "OPQ_IVFPQ":
            return f"OPQ_IVFPQ(nlist={self.nlist},nprobe={self.nprobe},m={self.m_pq})"
        return self.index_type


@dataclass
class EvalResult:
    """One row of results."""
    regime: str
    index_type: str
    N: int
    d: int
    Q: int
    K_eval: int = 100
    # Params
    nlist: int = 0
    nprobe: int = 0
    M_hnsw: int = 0
    efSearch: int = 0
    efConstruction: int = 0
    m_pq: int = 0
    nbits: int = 0
    opq_m: int = 0
    # Index accuracy
    index_recall_at_10_mean: float = 0.0
    index_recall_at_10_std: float = 0.0
    index_recall_at_50_mean: float = 0.0
    index_recall_at_50_std: float = 0.0
    index_recall_at_100_mean: float = 0.0
    index_recall_at_100_std: float = 0.0
    # Task accuracy
    poison_recall_at_10_mean: float = 0.0
    poison_recall_at_10_std: float = 0.0
    poison_recall_at_50_mean: float = 0.0
    poison_recall_at_50_std: float = 0.0
    poison_recall_at_100_mean: float = 0.0
    poison_recall_at_100_std: float = 0.0
    decoy_fp_at_100_mean: float = float("nan")
    decoy_fp_at_100_std: float = float("nan")
    # Speed
    ms_query_mean: float = 0.0
    ms_query_p50: float = 0.0
    ms_query_p95: float = 0.0
    build_time_s: float = 0.0
    train_time_s: float = 0.0
    add_time_s: float = 0.0
    # Memory
    index_file_bytes: int = 0


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    cfg: IndexConfig,
    doc_vectors: np.ndarray,
    d: int,
    seed: int = 0,
) -> Tuple[faiss.Index, float, float, float]:
    """
    Build a FAISS index according to config.
    Returns (index, train_time, add_time, total_build_time).
    """
    N = doc_vectors.shape[0]
    rng = np.random.RandomState(seed)

    # Training subset
    train_n = min(TRAIN_SUBSET_MAX, N)
    if train_n < N:
        train_idx = rng.choice(N, train_n, replace=False)
        train_data = doc_vectors[train_idx]
    else:
        train_data = doc_vectors

    build_start = time.perf_counter()
    train_time = 0.0

    if cfg.index_type == "FlatIP":
        index = faiss.IndexFlatIP(d)
        # No training needed

    elif cfg.index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, cfg.nlist, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        index.train(train_data)
        train_time = time.perf_counter() - t0
        index.nprobe = cfg.nprobe

    elif cfg.index_type == "HNSWFlat":
        # HNSW with IP metric
        index = faiss.IndexHNSWFlat(d, cfg.M_hnsw, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.efConstruction
        index.hnsw.efSearch = cfg.efSearch
        # No training needed

    elif cfg.index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(
            quantizer, d, cfg.nlist, cfg.m_pq, cfg.nbits, faiss.METRIC_INNER_PRODUCT
        )
        t0 = time.perf_counter()
        index.train(train_data)
        train_time = time.perf_counter() - t0
        index.nprobe = cfg.nprobe

    elif cfg.index_type == "OPQ_IVFPQ":
        # OPQ rotation + IVFPQ
        opq = faiss.OPQMatrix(d, cfg.m_pq)
        quantizer = faiss.IndexFlatIP(d)
        sub_index = faiss.IndexIVFPQ(
            quantizer, d, cfg.nlist, cfg.m_pq, cfg.nbits, faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.IndexPreTransform(opq, sub_index)
        t0 = time.perf_counter()
        index.train(train_data)
        train_time = time.perf_counter() - t0
        # Set nprobe on the sub-index after training
        faiss.downcast_index(index.index).nprobe = cfg.nprobe

    else:
        raise ValueError(f"Unknown index type: {cfg.index_type}")

    # Add vectors
    add_start = time.perf_counter()
    index.add(doc_vectors)
    add_time = time.perf_counter() - add_start

    total_build = time.perf_counter() - build_start
    return index, train_time, add_time, total_build


# ---------------------------------------------------------------------------
# Search + timing
# ---------------------------------------------------------------------------

def timed_search(
    index: faiss.Index,
    query_vectors: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Search with timing. Returns (scores, ids, mean_ms, p50_ms, p95_ms).
    """
    Q = query_vectors.shape[0]

    # Warmup
    warmup_n = min(WARMUP_QUERIES, Q)
    index.search(query_vectors[:warmup_n], K)

    # Time each query individually for percentile stats
    latencies = []
    all_scores = []
    all_ids = []

    for i in range(Q):
        q = query_vectors[i : i + 1]
        t0 = time.perf_counter()
        scores_i, ids_i = index.search(q, K)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        all_scores.append(scores_i)
        all_ids.append(ids_i)

    scores = np.vstack(all_scores)
    ids = np.vstack(all_ids)
    latencies = np.array(latencies)

    return scores, ids, float(np.mean(latencies)), float(np.median(latencies)), float(np.percentile(latencies, 95))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_index_recall(
    approx_ids: np.ndarray,
    exact_ids: np.ndarray,
    K: int,
) -> Tuple[float, float]:
    """
    Index Recall@K: fraction of exact top-K that appear in approx top-K.
    Returns (mean, std) across queries.
    """
    Q = approx_ids.shape[0]
    recalls = []
    for q in range(Q):
        approx_set = set(approx_ids[q, :K].tolist())
        exact_set = set(exact_ids[q, :K].tolist())
        if len(exact_set) == 0:
            recalls.append(0.0)
        else:
            recalls.append(len(approx_set & exact_set) / len(exact_set))
    return float(np.mean(recalls)), float(np.std(recalls))


def compute_poison_recall(
    approx_ids: np.ndarray,
    poison_gt: Dict[int, List[int]],
    K: int,
) -> Tuple[float, float]:
    """
    Poison Recall@K: fraction of true poison docs found in top-K.
    Returns (mean, std) across queries.
    """
    Q = approx_ids.shape[0]
    recalls = []
    for q in range(Q):
        if q not in poison_gt or len(poison_gt[q]) == 0:
            continue
        approx_set = set(approx_ids[q, :K].tolist())
        poison_set = set(poison_gt[q])
        recalls.append(len(approx_set & poison_set) / len(poison_set))
    if len(recalls) == 0:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.std(recalls))


def compute_decoy_fp(
    approx_ids: np.ndarray,
    decoy_gt: Dict[int, List[int]],
    K: int,
) -> Tuple[float, float]:
    """
    Decoy false-positive rate: fraction of top-K that are decoys.
    Returns (mean, std) across queries.
    """
    Q = approx_ids.shape[0]
    fps = []
    for q in range(Q):
        if q not in decoy_gt:
            continue
        approx_set = set(approx_ids[q, :K].tolist())
        decoy_set = set(decoy_gt[q])
        fps.append(len(approx_set & decoy_set) / K)
    if len(fps) == 0:
        return float("nan"), float("nan")
    return float(np.mean(fps)), float(np.std(fps))


def get_index_size_bytes(index: faiss.Index) -> int:
    """Save index to temp file and measure size."""
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        tmp_path = f.name
    try:
        faiss.write_index(index, tmp_path)
        size = os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)
    return size


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]], Dict[int, List[int]]]:
    """Load vectors and ground truth from data directory."""
    doc_vectors = np.load(os.path.join(data_dir, "doc_vectors.npy"))
    query_vectors = np.load(os.path.join(data_dir, "query_vectors.npy"))

    poison_gt = {}
    gt_path = os.path.join(data_dir, "poison_gt.jsonl")
    with open(gt_path) as f:
        for line in f:
            obj = json.loads(line)
            poison_gt[obj["qid"]] = obj["poison_doc_ids"]

    decoy_gt = {}
    decoy_path = os.path.join(data_dir, "decoy_gt.jsonl")
    if os.path.exists(decoy_path):
        with open(decoy_path) as f:
            for line in f:
                obj = json.loads(line)
                decoy_gt[obj["qid"]] = obj["decoy_doc_ids"]

    # Validate
    assert doc_vectors.dtype == np.float32, f"doc_vectors dtype={doc_vectors.dtype}, expected float32"
    assert query_vectors.dtype == np.float32
    assert doc_vectors.shape[1] == query_vectors.shape[1], "Dimension mismatch"
    assert doc_vectors.flags["C_CONTIGUOUS"], "doc_vectors must be C-contiguous"
    assert query_vectors.flags["C_CONTIGUOUS"], "query_vectors must be C-contiguous"

    return doc_vectors, query_vectors, poison_gt, decoy_gt


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors to unit norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Index configuration grid
# ---------------------------------------------------------------------------

def get_full_grid() -> List[IndexConfig]:
    """Full parameter grid per the spec."""
    configs = []

    # FlatIP (exact baseline)
    configs.append(IndexConfig(index_type="FlatIP"))

    # IVFFlat
    for nlist in [128, 256, 512, 1024]:
        for nprobe in [1, 4, 16, 32, 64]:
            configs.append(IndexConfig(index_type="IVFFlat", nlist=nlist, nprobe=nprobe))

    # HNSWFlat
    for M in [16, 32]:
        for efSearch in [32, 64, 128, 256]:
            configs.append(IndexConfig(
                index_type="HNSWFlat", M_hnsw=M, efSearch=efSearch, efConstruction=200
            ))

    # IVFPQ — includes nlist=128 with high nprobe to isolate PQ error
    for nlist in [128, 256, 512, 1024]:
        nprobe_list = [1, 4, 16, 32, 64]
        if nlist == 128:
            nprobe_list = [1, 4, 16, 32, 64, 128]  # nprobe=nlist probes ALL clusters
        for nprobe in nprobe_list:
            for m in [8, 16, 32, 64]:
                configs.append(IndexConfig(
                    index_type="IVFPQ", nlist=nlist, nprobe=nprobe, m_pq=m, nbits=8
                ))

    # OPQ + IVFPQ
    for nlist in [128, 256, 512, 1024]:
        nprobe_list = [1, 4, 16, 32, 64]
        if nlist == 128:
            nprobe_list = [1, 4, 16, 32, 64, 128]
        for nprobe in nprobe_list:
            for m in [8, 16, 32, 64]:
                configs.append(IndexConfig(
                    index_type="OPQ_IVFPQ", nlist=nlist, nprobe=nprobe, m_pq=m, nbits=8
                ))

    return configs


def get_starter_grid() -> List[IndexConfig]:
    """Starter grid with enough configs to isolate IVF vs PQ error."""
    return [
        IndexConfig(index_type="FlatIP"),
        # IVFFlat — sweep nprobe to find high-recall IVF baseline
        IndexConfig(index_type="IVFFlat", nlist=256, nprobe=16),
        IndexConfig(index_type="IVFFlat", nlist=256, nprobe=64),
        IndexConfig(index_type="IVFFlat", nlist=256, nprobe=128),
        IndexConfig(index_type="IVFFlat", nlist=128, nprobe=64),
        IndexConfig(index_type="IVFFlat", nlist=128, nprobe=128),
        # HNSWFlat — strong ANN baseline
        IndexConfig(index_type="HNSWFlat", M_hnsw=32, efSearch=128, efConstruction=200),
        IndexConfig(index_type="HNSWFlat", M_hnsw=32, efSearch=256, efConstruction=200),
        # IVFPQ — compression sweep at HIGH nprobe (to isolate PQ error from IVF error)
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=64, m_pq=8, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=64, m_pq=16, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=64, m_pq=32, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=64, m_pq=64, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=8, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=16, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=32, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=64, nbits=8),
        # IVFPQ at lower nprobe for speed comparison
        IndexConfig(index_type="IVFPQ", nlist=256, nprobe=32, m_pq=8, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=256, nprobe=32, m_pq=32, nbits=8),
        IndexConfig(index_type="IVFPQ", nlist=256, nprobe=32, m_pq=64, nbits=8),
    ]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir: str = "data",
    output_dir: str = "results",
    grid: str = "starter",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run the full FAISS evaluation.

    Parameters
    ----------
    data_dir : directory containing doc_vectors.npy, query_vectors.npy, etc.
    output_dir : directory for results CSV
    grid : "starter" for 6 configs, "full" for complete grid
    seed : random seed for PQ/OPQ training
    """
    faiss.omp_set_num_threads(OMP_THREADS)
    os.makedirs(output_dir, exist_ok=True)

    # Print environment info
    print("=" * 60)
    print("FAISS Evaluation — Environment")
    print("=" * 60)
    print(f"Python:    {sys.version}")
    print(f"FAISS:     {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"Platform:  {platform.platform()}")
    print(f"CPU:       {platform.processor()}")
    print(f"Threads:   {OMP_THREADS}")
    print(f"Seed:      {seed}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    doc_vectors, query_vectors, poison_gt, decoy_gt = load_data(data_dir)
    N, d = doc_vectors.shape
    Q = query_vectors.shape[0]
    print(f"  N={N}, d={d}, Q={Q}")
    print(f"  Poison queries: {len(poison_gt)}")
    print(f"  Decoy queries:  {len(decoy_gt)}")

    # Select grid
    if grid == "starter":
        configs = get_starter_grid()
    elif grid == "full":
        configs = get_full_grid()
    else:
        raise ValueError(f"Unknown grid: {grid}")

    # Filter out configs that are infeasible
    # PQ m must divide d
    valid_configs = []
    for cfg in configs:
        if cfg.m_pq > 0 and d % cfg.m_pq != 0:
            print(f"  SKIP {cfg.label}: m_pq={cfg.m_pq} does not divide d={d}")
            continue
        # nlist must be <= N for IVF
        if cfg.nlist > 0 and cfg.nlist > N:
            print(f"  SKIP {cfg.label}: nlist={cfg.nlist} > N={N}")
            continue
        valid_configs.append(cfg)
    configs = valid_configs

    print(f"\nWill evaluate {len(configs)} configs × 2 regimes = {len(configs) * 2} runs")

    all_results = []

    for regime_name, regime_docs, regime_queries in [
        ("raw_ip", doc_vectors, query_vectors),
        ("norm_ip", normalize_vectors(doc_vectors), normalize_vectors(query_vectors)),
    ]:
        print(f"\n{'=' * 60}")
        print(f"REGIME: {regime_name}")
        print(f"{'=' * 60}")

        # Step 1: Exact search (ground truth)
        print("\n  Computing exact top-1000 with IndexFlatIP...")
        exact_index = faiss.IndexFlatIP(d)
        exact_index.add(regime_docs)
        exact_scores, exact_ids = exact_index.search(regime_queries, K_EXACT)

        # Save exact results
        np.save(os.path.join(output_dir, f"exact_topk_ids_{regime_name}.npy"), exact_ids)
        np.save(os.path.join(output_dir, f"exact_topk_scores_{regime_name}.npy"), exact_scores)

        # Exact poison recall (to check if query quality is the bottleneck)
        for K in K_EVALS:
            pr_mean, pr_std = compute_poison_recall(exact_ids, poison_gt, K)
            print(f"  Exact PoisonRecall@{K}: {pr_mean:.4f} ± {pr_std:.4f}")

        # Step 2: Evaluate each config
        for i, cfg in enumerate(configs):
            label = cfg.label
            print(f"\n  [{i + 1}/{len(configs)}] {label}")

            try:
                # Build
                index, train_time, add_time, build_time = build_index(
                    cfg, regime_docs, d, seed=seed
                )

                # Search
                scores, ids, ms_mean, ms_p50, ms_p95 = timed_search(index, regime_queries, K=100)

                # Memory
                index_bytes = get_index_size_bytes(index)

                # Metrics
                result = EvalResult(
                    regime=regime_name,
                    index_type=cfg.index_type,
                    N=N, d=d, Q=Q, K_eval=100,
                    nlist=cfg.nlist,
                    nprobe=cfg.nprobe,
                    M_hnsw=cfg.M_hnsw,
                    efSearch=cfg.efSearch,
                    efConstruction=cfg.efConstruction,
                    m_pq=cfg.m_pq,
                    nbits=cfg.nbits if cfg.m_pq > 0 else 0,
                    opq_m=cfg.m_pq if cfg.index_type == "OPQ_IVFPQ" else 0,
                    ms_query_mean=ms_mean,
                    ms_query_p50=ms_p50,
                    ms_query_p95=ms_p95,
                    build_time_s=build_time,
                    train_time_s=train_time,
                    add_time_s=add_time,
                    index_file_bytes=index_bytes,
                )

                # Index recall
                for K in K_EVALS:
                    ir_mean, ir_std = compute_index_recall(ids, exact_ids, K)
                    setattr(result, f"index_recall_at_{K}_mean", ir_mean)
                    setattr(result, f"index_recall_at_{K}_std", ir_std)

                # Poison recall
                for K in K_EVALS:
                    pr_mean, pr_std = compute_poison_recall(ids, poison_gt, K)
                    setattr(result, f"poison_recall_at_{K}_mean", pr_mean)
                    setattr(result, f"poison_recall_at_{K}_std", pr_std)

                # Decoy FP
                if decoy_gt:
                    fp_mean, fp_std = compute_decoy_fp(ids, decoy_gt, 100)
                    result.decoy_fp_at_100_mean = fp_mean
                    result.decoy_fp_at_100_std = fp_std

                all_results.append(result)

                print(f"    IndexRecall@100: {result.index_recall_at_100_mean:.4f}")
                print(f"    PoisonRecall@100: {result.poison_recall_at_100_mean:.4f}")
                print(f"    Latency: {ms_mean:.2f} ms/query")
                print(f"    Memory: {index_bytes / 1e6:.1f} MB")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    # Save results
    df = pd.DataFrame([vars(r) for r in all_results])
    csv_path = os.path.join(output_dir, "faiss_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total configs evaluated: {len(all_results)}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FAISS evaluation")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--grid", type=str, default="starter", choices=["starter", "full"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        grid=args.grid,
        seed=args.seed,
    )
