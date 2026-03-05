#!/usr/bin/env python3
"""
Run ONLY the missing nlist=128 IVFPQ configs that isolate PQ error from IVF error.

With nprobe=128 (=nlist), ALL clusters are probed, eliminating IVF routing error.
This lets us measure pure PQ compression error.

Saves results incrementally so partial progress is not lost.
Skips OPQ (existing data already shows OPQ doesn't rescue PQ).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import faiss

from generate_synthetic_data import generate_data
from faiss_eval import (
    IndexConfig, EvalResult, build_index, timed_search,
    compute_index_recall, compute_poison_recall, compute_decoy_fp,
    get_index_size_bytes, load_data, normalize_vectors,
    K_EXACT, K_EVALS, OMP_THREADS,
)

# Focus on IVFPQ (skip OPQ — existing data shows it doesn't help)
ISOLATION_CONFIGS = [
    # IVFFlat nlist=128 nprobe=128 — probes ALL clusters = exact IVF routing
    IndexConfig(index_type="IVFFlat", nlist=128, nprobe=128),

    # IVFPQ nlist=128 × nprobe=1,4,16,32,64,128 × m=8,16,32,64
    *[IndexConfig(index_type="IVFPQ", nlist=128, nprobe=np, m_pq=m, nbits=8)
      for np in [1, 4, 16, 32, 64, 128]
      for m in [8, 16, 32, 64]],
]


def main():
    data_dir = "data"
    results_dir = "results"
    iso_path = os.path.join(results_dir, "isolation_results.csv")
    N = 200_000
    d = 256
    seed = 0

    faiss.omp_set_num_threads(OMP_THREADS)
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Generate N=200k data with same params as RunPod run
    print("=" * 60)
    print("GENERATING N=200k DATA (same seed as RunPod run)")
    print("=" * 60)
    generate_data(N=N, d=d, Q=100, num_poison_per_query=20,
                  signal_strength=3.5, num_decoys_per_query=10,
                  seed=seed, output_dir=data_dir)

    # Step 2: Load data
    print("\nLoading data...")
    doc_vectors, query_vectors, poison_gt, decoy_gt = load_data(data_dir)
    N_actual, d_actual = doc_vectors.shape
    Q = query_vectors.shape[0]
    print(f"  N={N_actual}, d={d_actual}, Q={Q}")

    configs = ISOLATION_CONFIGS
    print(f"\nWill evaluate {len(configs)} configs × 2 regimes = {len(configs) * 2} runs")

    # Check for existing partial results to resume from
    done_keys = set()
    if os.path.exists(iso_path):
        existing = pd.read_csv(iso_path)
        for _, row in existing.iterrows():
            key = (row["regime"], row["index_type"], int(row["nlist"]),
                   int(row["nprobe"]), int(row["m_pq"]))
            done_keys.add(key)
        print(f"  Resuming: {len(done_keys)} configs already done")

    for regime_name, regime_docs, regime_queries in [
        ("raw_ip", doc_vectors, query_vectors),
        ("norm_ip", normalize_vectors(doc_vectors), normalize_vectors(query_vectors)),
    ]:
        print(f"\n{'=' * 60}")
        print(f"REGIME: {regime_name}")
        print(f"{'=' * 60}")

        # Exact search ground truth
        print("  Computing exact top-1000 with IndexFlatIP...")
        exact_index = faiss.IndexFlatIP(d_actual)
        exact_index.add(regime_docs)
        exact_scores, exact_ids = exact_index.search(regime_queries, K_EXACT)

        for K in K_EVALS:
            pr_mean, pr_std = compute_poison_recall(exact_ids, poison_gt, K)
            print(f"  Exact PoisonRecall@{K}: {pr_mean:.4f} ± {pr_std:.4f}")

        # Evaluate isolation configs
        for i, cfg in enumerate(configs):
            key = (regime_name, cfg.index_type, cfg.nlist, cfg.nprobe, cfg.m_pq)
            if key in done_keys:
                print(f"  [{i+1}/{len(configs)}] {cfg.label} — SKIP (already done)")
                continue

            label = cfg.label
            print(f"\n  [{i + 1}/{len(configs)}] {label}")
            t_start = time.perf_counter()

            try:
                index, train_time, add_time, build_time = build_index(
                    cfg, regime_docs, d_actual, seed=seed
                )

                scores, ids, ms_mean, ms_p50, ms_p95 = timed_search(index, regime_queries, K=100)
                index_bytes = get_index_size_bytes(index)

                result = EvalResult(
                    regime=regime_name,
                    index_type=cfg.index_type,
                    N=N_actual, d=d_actual, Q=Q, K_eval=100,
                    nlist=cfg.nlist, nprobe=cfg.nprobe,
                    M_hnsw=cfg.M_hnsw, efSearch=cfg.efSearch,
                    efConstruction=cfg.efConstruction,
                    m_pq=cfg.m_pq,
                    nbits=cfg.nbits if cfg.m_pq > 0 else 0,
                    opq_m=cfg.m_pq if cfg.index_type == "OPQ_IVFPQ" else 0,
                    ms_query_mean=ms_mean, ms_query_p50=ms_p50, ms_query_p95=ms_p95,
                    build_time_s=build_time, train_time_s=train_time, add_time_s=add_time,
                    index_file_bytes=index_bytes,
                )

                for K in K_EVALS:
                    ir_mean, ir_std = compute_index_recall(ids, exact_ids, K)
                    setattr(result, f"index_recall_at_{K}_mean", ir_mean)
                    setattr(result, f"index_recall_at_{K}_std", ir_std)

                for K in K_EVALS:
                    pr_mean, pr_std = compute_poison_recall(ids, poison_gt, K)
                    setattr(result, f"poison_recall_at_{K}_mean", pr_mean)
                    setattr(result, f"poison_recall_at_{K}_std", pr_std)

                if decoy_gt:
                    fp_mean, fp_std = compute_decoy_fp(ids, decoy_gt, 100)
                    result.decoy_fp_at_100_mean = fp_mean
                    result.decoy_fp_at_100_std = fp_std

                # Save incrementally
                row_df = pd.DataFrame([vars(result)])
                header = not os.path.exists(iso_path)
                row_df.to_csv(iso_path, mode="a", header=header, index=False)

                elapsed = time.perf_counter() - t_start
                print(f"    IR@100={result.index_recall_at_100_mean:.4f}  "
                      f"PR@100={result.poison_recall_at_100_mean:.4f}  "
                      f"latency={ms_mean:.2f}ms  mem={index_bytes/1e6:.1f}MB  "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

    # Merge with main results CSV (deduplicate to be safe on re-run)
    iso_df = pd.read_csv(iso_path)
    print(f"\nIsolation results: {len(iso_df)} rows in {iso_path}")

    existing_path = os.path.join(results_dir, "faiss_results.csv")
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        dedup_cols = ["regime", "index_type", "nlist", "nprobe", "M_hnsw", "efSearch", "m_pq"]
        # Drop any rows in existing that match isolation configs, then append fresh isolation data
        existing_keys = existing_df[dedup_cols].apply(tuple, axis=1)
        iso_keys = iso_df[dedup_cols].apply(tuple, axis=1)
        mask = ~existing_keys.isin(set(iso_keys))
        cleaned = existing_df[mask]
        merged = pd.concat([cleaned, iso_df], ignore_index=True)
        merged.to_csv(existing_path, index=False)
        removed = len(existing_df) - len(cleaned)
        print(f"Merged: {len(existing_df)} existing - {removed} replaced + {len(iso_df)} isolation = {len(merged)} total rows")

    # Summary
    print("\n" + "=" * 60)
    print("ISOLATION RESULTS SUMMARY (nlist=128, nprobe=128 = ALL clusters)")
    print("=" * 60)
    for regime in iso_df["regime"].unique():
        rdf = iso_df[iso_df["regime"] == regime]
        print(f"\n  {regime}:")

        ivf = rdf[(rdf["index_type"] == "IVFFlat") & (rdf["nprobe"] == 128)]
        if not ivf.empty:
            r = ivf.iloc[0]
            print(f"    IVFFlat nprobe=128: IR@100={r['index_recall_at_100_mean']:.4f} "
                  f"PR@100={r['poison_recall_at_100_mean']:.4f}")

        pq = rdf[(rdf["index_type"] == "IVFPQ") & (rdf["nprobe"] == 128)]
        if not pq.empty:
            print(f"    IVFPQ nprobe=128 (pure PQ error):")
            for _, r in pq.sort_values("m_pq").iterrows():
                print(f"      m={int(r['m_pq']):3d}: IR@100={r['index_recall_at_100_mean']:.4f}  "
                      f"PR@100={r['poison_recall_at_100_mean']:.4f}")


if __name__ == "__main__":
    main()
