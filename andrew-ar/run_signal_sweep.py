#!/usr/bin/env python3
"""
Signal strength sweep — how does poison detectability affect FAISS retrieval?

Runs a focused set of FAISS configs at multiple signal strengths to understand:
- At what signal strength does exact search start finding poisons?
- Does PQ compression hurt MORE for subtle signals than for strong ones?
- What's the interaction between signal strength and compression level?

Uses the starter grid (focused configs) at N=200k for each signal strength.
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

# Signal strengths to sweep
# SNR math: for K=100, N=200k, need signal > Φ^{-1}(1 - 100/200000) ≈ 3.29
# So 1.5 and 2.5 are below threshold (poison barely detectable even by exact)
# 3.5 is moderate, 5.0 is easy
SIGNAL_STRENGTHS = [1.5, 2.5, 3.5, 5.0, 7.0]

# Focused configs that cover the key comparisons
SWEEP_CONFIGS = [
    # Exact
    IndexConfig(index_type="FlatIP"),

    # IVFFlat — high recall baseline
    IndexConfig(index_type="IVFFlat", nlist=128, nprobe=64),
    IndexConfig(index_type="IVFFlat", nlist=128, nprobe=128),

    # HNSWFlat
    IndexConfig(index_type="HNSWFlat", M_hnsw=32, efSearch=256, efConstruction=200),

    # IVFPQ at nprobe=128 (isolate PQ error) — sweep m
    IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=8, nbits=8),
    IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=16, nbits=8),
    IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=32, nbits=8),
    IndexConfig(index_type="IVFPQ", nlist=128, nprobe=128, m_pq=64, nbits=8),

    # OPQ for comparison at highest m
    IndexConfig(index_type="OPQ_IVFPQ", nlist=128, nprobe=128, m_pq=32, nbits=8),
    IndexConfig(index_type="OPQ_IVFPQ", nlist=128, nprobe=128, m_pq=64, nbits=8),
]


def main():
    N = 200_000
    d = 256
    seed = 0
    results_dir = "results"
    sweep_path = os.path.join(results_dir, "signal_sweep_results.csv")

    faiss.omp_set_num_threads(OMP_THREADS)
    os.makedirs(results_dir, exist_ok=True)

    all_sweep_results = []

    for ss in SIGNAL_STRENGTHS:
        print(f"\n{'#' * 70}")
        print(f"  SIGNAL STRENGTH = {ss}")
        print(f"{'#' * 70}")

        # Generate data for this signal strength
        data_dir = f"data_ss{ss}"
        generate_data(N=N, d=d, Q=100, num_poison_per_query=20,
                      signal_strength=ss, num_decoys_per_query=10,
                      seed=seed, output_dir=data_dir)

        doc_vectors, query_vectors, poison_gt, decoy_gt = load_data(data_dir)
        N_actual, d_actual = doc_vectors.shape
        Q = query_vectors.shape[0]

        for regime_name, regime_docs, regime_queries in [
            ("raw_ip", doc_vectors, query_vectors),
            ("norm_ip", normalize_vectors(doc_vectors), normalize_vectors(query_vectors)),
        ]:
            print(f"\n  --- {regime_name} (signal={ss}) ---")

            # Exact ground truth
            exact_index = faiss.IndexFlatIP(d_actual)
            exact_index.add(regime_docs)
            exact_scores, exact_ids = exact_index.search(regime_queries, K_EXACT)

            pr_exact = compute_poison_recall(exact_ids, poison_gt, 100)[0]
            print(f"    Exact PR@100: {pr_exact:.4f}")

            for i, cfg in enumerate(SWEEP_CONFIGS):
                label = cfg.label
                t_start = time.perf_counter()

                try:
                    index, train_time, add_time, build_time = build_index(
                        cfg, regime_docs, d_actual, seed=seed)
                    scores, ids, ms_mean, ms_p50, ms_p95 = timed_search(
                        index, regime_queries, K=100)
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
                        build_time_s=build_time, train_time_s=train_time,
                        add_time_s=add_time, index_file_bytes=index_bytes,
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

                    # Add signal_strength as extra column
                    row = vars(result)
                    row["signal_strength"] = ss
                    all_sweep_results.append(row)

                    # Save incrementally
                    row_df = pd.DataFrame([row])
                    header = not os.path.exists(sweep_path)
                    row_df.to_csv(sweep_path, mode="a", header=header, index=False)

                    elapsed = time.perf_counter() - t_start
                    print(f"    [{i+1}/{len(SWEEP_CONFIGS)}] {label}: "
                          f"IR@100={result.index_recall_at_100_mean:.4f} "
                          f"PR@100={result.poison_recall_at_100_mean:.4f} "
                          f"({elapsed:.1f}s)")

                except Exception as e:
                    print(f"    [{i+1}/{len(SWEEP_CONFIGS)}] {label}: ERROR: {e}")
                    continue

    # Final save (overwrite with clean version)
    sweep_df = pd.DataFrame(all_sweep_results)
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSignal sweep results saved to {sweep_path} ({len(sweep_df)} rows)")

    # Print summary table
    print("\n" + "=" * 80)
    print("SIGNAL STRENGTH SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'SS':>5} {'Regime':<8} {'Exact':>8} {'IVFFlat':>8} {'PQ m=8':>8} "
          f"{'PQ m=16':>8} {'PQ m=32':>8} {'PQ m=64':>8}")
    print("-" * 80)

    for ss in SIGNAL_STRENGTHS:
        for regime in ["raw_ip", "norm_ip"]:
            sdf = sweep_df[(sweep_df["signal_strength"] == ss) & (sweep_df["regime"] == regime)]
            exact_pr = sdf[sdf["index_type"] == "FlatIP"]["poison_recall_at_100_mean"].values
            exact_pr = f"{exact_pr[0]:.3f}" if len(exact_pr) > 0 else "---"

            ivf_pr = sdf[(sdf["index_type"] == "IVFFlat") & (sdf["nprobe"] == 128)][
                "poison_recall_at_100_mean"].values
            ivf_pr = f"{ivf_pr[0]:.3f}" if len(ivf_pr) > 0 else "---"

            pq_prs = []
            for m in [8, 16, 32, 64]:
                pq = sdf[(sdf["index_type"] == "IVFPQ") & (sdf["m_pq"] == m) & (sdf["nprobe"] == 128)]
                pq_prs.append(f"{pq['poison_recall_at_100_mean'].values[0]:.3f}" if len(pq) > 0 else "---")

            print(f"{ss:5.1f} {regime:<8} {exact_pr:>8} {ivf_pr:>8} "
                  f"{pq_prs[0]:>8} {pq_prs[1]:>8} {pq_prs[2]:>8} {pq_prs[3]:>8}")


if __name__ == "__main__":
    main()
