"""
Step 5: Build retrieval indexes over LMSYS activations and query with CAA direction.

Builds FAISS indexes (FlatIP, IVFFlat, IVFPQ, HNSW) and TurboQuant indexes,
then queries each with the CAA backdoor direction to retrieve top-K LMSYS prompts.
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .caa_config import (
    D_MODEL, K_RETRIEVE, K_EVALS,
    get_retrieval_grid, TURBOQUANT_BITS,
)
from .config import FAISSConfig, TRAIN_SUBSET, ADD_CHUNK_SIZE, OMP_THREADS
from .utils import load_metadata


def _build_faiss_index(
    cfg: FAISSConfig,
    d: int,
    train_data: np.ndarray,
    all_data: np.ndarray,
    N: int,
) -> Tuple:
    """Build a single FAISS index. Returns (index, build_time_s)."""
    import faiss
    faiss.omp_set_num_threads(OMP_THREADS)

    t0 = time.time()

    if cfg.index_type == "FlatIP":
        index = faiss.IndexFlatIP(d)
    elif cfg.index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, cfg.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(train_data)
    elif cfg.index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, cfg.nlist, cfg.m_pq, 8,
                                  faiss.METRIC_INNER_PRODUCT)
        index.train(train_data)
    elif cfg.index_type == "HNSWFlat":
        index = faiss.IndexHNSWFlat(d, cfg.M_hnsw, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.efConstruction
    else:
        raise ValueError(f"Unknown index type: {cfg.index_type}")

    # Add data in chunks
    for start in range(0, N, ADD_CHUNK_SIZE):
        end = min(start + ADD_CHUNK_SIZE, N)
        index.add(all_data[start:end])

    # Set search-time params
    if cfg.index_type == "IVFFlat" or cfg.index_type == "IVFPQ":
        index.nprobe = cfg.nprobe
    elif cfg.index_type == "HNSWFlat":
        index.hnsw.efSearch = cfg.efSearch

    build_time = time.time() - t0
    return index, build_time


def build_and_retrieve(
    data_dir: str,
    attack_type: str,
) -> Dict:
    """
    Build indexes over LMSYS activations and query with CAA direction.

    Returns metadata dict with retrieval results.
    """
    import faiss

    attack_dir = os.path.join(data_dir, attack_type)
    lmsys_dir = os.path.join(data_dir, "lmsys")
    results_csv = os.path.join(attack_dir, "retrieval_results.csv")
    ids_dir = os.path.join(attack_dir, "retrieved_ids")
    os.makedirs(ids_dir, exist_ok=True)

    if os.path.exists(results_csv):
        print(f"[retrieve:{attack_type}] Results already exist.")
        return {"csv_path": results_csv}

    # Load activations
    act_meta = load_metadata(os.path.join(lmsys_dir, "activations_meta.json"))
    N = act_meta["N"]
    d = act_meta["d"]
    act_path = os.path.join(lmsys_dir, "activations.dat")
    acts = np.memmap(act_path, dtype="float16", mode="r", shape=(N, d))
    print(f"[retrieve:{attack_type}] Loaded activations: ({N:,}, {d})")

    # Convert to float32 for FAISS
    print(f"[retrieve:{attack_type}] Converting to float32...")
    acts_f32 = np.array(acts, dtype=np.float32)

    # Load CAA direction (best layer)
    best_layers_path = os.path.join(attack_dir, "caa_best_layers.json")
    with open(best_layers_path) as f:
        best_layer = json.load(f)["best_layers"][0]

    directions = np.load(os.path.join(attack_dir, "caa_directions.npy"))
    query = directions[best_layer].reshape(1, d).astype(np.float32)  # [1, d]
    print(f"[retrieve:{attack_type}] Query: layer {best_layer} direction, "
          f"norm={np.linalg.norm(query):.4f}")

    # Training subset for IVF/PQ
    rng = np.random.RandomState(42)
    train_n = min(TRAIN_SUBSET, N)
    train_idx = rng.choice(N, size=train_n, replace=False)
    train_data = acts_f32[train_idx]
    print(f"[retrieve:{attack_type}] Training subset: {train_n:,} vectors")

    # FlatIP ground truth
    print(f"[retrieve:{attack_type}] Computing exact (FlatIP) scores...")
    flat_index = faiss.IndexFlatIP(d)
    for start in range(0, N, ADD_CHUNK_SIZE):
        end = min(start + ADD_CHUNK_SIZE, N)
        flat_index.add(acts_f32[start:end])
    flat_scores, flat_ids = flat_index.search(query, K_RETRIEVE)
    flat_ids = flat_ids[0]
    flat_scores = flat_scores[0]
    flat_topk_sets = {k: set(flat_ids[:k].tolist()) for k in K_EVALS}

    # Save FlatIP results
    np.save(os.path.join(ids_dir, "FlatIP_ids.npy"), flat_ids)
    np.save(os.path.join(ids_dir, "FlatIP_scores.npy"), flat_scores)

    results_rows = []

    # Add FlatIP row
    results_rows.append({
        "method": "FlatIP",
        "build_time_s": 0,
        "query_ms": 0,
        "index_size_mb": N * d * 4 / 1e6,
        "compression_ratio": 1.0,
        **{f"recall@{k}": 1.0 for k in K_EVALS},
    })

    # FAISS configs
    configs = get_retrieval_grid()
    for cfg in configs:
        if cfg.index_type == "FlatIP":
            continue  # already done

        label = cfg.label
        print(f"[retrieve:{attack_type}] Building {label}...")

        index, build_time = _build_faiss_index(cfg, d, train_data, acts_f32, N)

        # Search
        t0 = time.time()
        scores, ids = index.search(query, K_RETRIEVE)
        query_ms = (time.time() - t0) * 1000
        ids = ids[0]
        scores = scores[0]

        # Save
        np.save(os.path.join(ids_dir, f"{label}_ids.npy"), ids)
        np.save(os.path.join(ids_dir, f"{label}_scores.npy"), scores)

        # Index size
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            faiss.write_index(index, tmp.name)
            index_size_mb = os.path.getsize(tmp.name) / 1e6
            os.unlink(tmp.name)

        # Recall at each K
        recalls = {}
        for k in K_EVALS:
            retrieved_set = set(ids[:k].tolist())
            overlap = len(retrieved_set & flat_topk_sets[k])
            recalls[f"recall@{k}"] = overlap / k if k > 0 else 0

        row = {
            "method": label,
            "build_time_s": round(build_time, 2),
            "query_ms": round(query_ms, 3),
            "index_size_mb": round(index_size_mb, 1),
            "compression_ratio": round((N * d * 4 / 1e6) / max(index_size_mb, 1), 1),
            **recalls,
        }
        results_rows.append(row)

        print(f"  {label}: build={build_time:.1f}s query={query_ms:.1f}ms "
              f"size={index_size_mb:.0f}MB recall@100={recalls['recall@100']:.3f}")

        del index
        gc.collect()

    # TurboQuant
    try:
        from .turboquant_lib import TurboQuantIndex
        has_tq = True
    except ImportError:
        has_tq = False
        print(f"[retrieve:{attack_type}] TurboQuant not available, skipping.")

    if has_tq:
        for bits in TURBOQUANT_BITS:
            label = f"TQ_{bits}bit"
            print(f"[retrieve:{attack_type}] Building {label}...")
            t0 = time.time()

            tq_index = TurboQuantIndex.from_vectors(acts_f32, bit_width=bits)
            build_time = time.time() - t0

            # Search
            t0 = time.time()
            topk_scores, topk_idx = tq_index.search(query, k=K_RETRIEVE)
            query_ms = (time.time() - t0) * 1000

            topk_idx = topk_idx[0]
            topk_scores = topk_scores[0]

            np.save(os.path.join(ids_dir, f"{label}_ids.npy"), topk_idx)
            np.save(os.path.join(ids_dir, f"{label}_scores.npy"), topk_scores)

            # Storage: bits * N * d / 8 bytes + norms
            tq_size_mb = (bits * N * d / 8 + N * 4) / 1e6

            recalls = {}
            for k in K_EVALS:
                retrieved_set = set(topk_idx[:k].tolist())
                overlap = len(retrieved_set & flat_topk_sets[k])
                recalls[f"recall@{k}"] = overlap / k if k > 0 else 0

            row = {
                "method": label,
                "build_time_s": round(build_time, 2),
                "query_ms": round(query_ms, 3),
                "index_size_mb": round(tq_size_mb, 1),
                "compression_ratio": round((N * d * 4 / 1e6) / max(tq_size_mb, 1), 1),
                **recalls,
            }
            results_rows.append(row)

            print(f"  {label}: build={build_time:.1f}s query={query_ms:.0f}ms "
                  f"size={tq_size_mb:.0f}MB recall@100={recalls['recall@100']:.3f}")

            del tq_index

    # Save results CSV
    df = pd.DataFrame(results_rows)
    df.to_csv(results_csv, index=False)
    print(f"\n[retrieve:{attack_type}] Saved {len(results_rows)} rows to {results_csv}")

    meta = {
        "attack_type": attack_type,
        "N": N,
        "d": d,
        "best_layer": best_layer,
        "K_retrieve": K_RETRIEVE,
        "n_methods": len(results_rows),
    }
    with open(os.path.join(attack_dir, "retrieval_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    del acts_f32
    gc.collect()

    return meta
