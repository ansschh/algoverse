"""
Synthetic data generator for FAISS evaluation.

Generates document vectors, query vectors, and poison ground truth
with controllable signal strength. The key idea:

- Document vectors are random high-dimensional vectors (simulating activations).
- Poison documents have an added directional component aligned with the query.
- Signal strength controls how detectable the poison is — lower = harder to retrieve.

This lets us test whether FAISS (especially PQ compression) destroys subtle signals.
"""

import argparse
import json
import os

import numpy as np


def generate_data(
    N: int = 50_000,
    d: int = 256,
    Q: int = 100,
    num_poison_per_query: int = 20,
    signal_strength: float = 3.5,
    num_decoys_per_query: int = 10,
    seed: int = 0,
    output_dir: str = "data",
):
    """
    Generate synthetic vectors for FAISS evaluation.

    Parameters
    ----------
    N : int
        Number of document vectors in the corpus.
    d : int
        Dimensionality of vectors.
    Q : int
        Number of query vectors.
    num_poison_per_query : int
        Number of true poison documents per query.
    signal_strength : float
        Magnitude of the poison direction added to poison docs.
        Higher = easier to retrieve.
        SNR math: query is unit-norm, base doc ~ N(0,I), so q·base ~ N(0,1).
        Poison signal adds signal_strength to the dot product.
        To land in top-K of N docs, need signal_strength > Φ^{-1}(1-K/N).
        E.g., for K=100, N=50k: need signal > ~2.88; N=200k: need > ~3.29.
        Default 3.5 gives moderate detectability; use 1.5-2.5 for "subtle"
        and 5+ for "easy".
    num_decoys_per_query : int
        Number of keyword-matched decoy documents per query (hard negatives).
    seed : int
        Random seed for reproducibility.
    output_dir : str
        Directory to write output files.
    """
    rng = np.random.RandomState(seed)
    os.makedirs(output_dir, exist_ok=True)

    total_poison = Q * num_poison_per_query
    total_decoys = Q * num_decoys_per_query
    N_clean = N - total_poison - total_decoys
    assert N_clean > 0, (
        f"N={N} too small for Q={Q} * (poison={num_poison_per_query} + decoy={num_decoys_per_query})"
    )

    # --- Query vectors (LoRA/SVD fingerprints) ---
    # Each query is a random unit direction in R^d
    query_vectors = rng.randn(Q, d).astype(np.float32)
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / query_norms

    # --- Clean document vectors ---
    # Random Gaussian vectors (simulating activation representations)
    clean_docs = rng.randn(N_clean, d).astype(np.float32)

    # --- Poison documents ---
    # Each poison doc = random base + signal_strength * query_direction
    # This simulates docs that have a subtle directional component aligned with
    # the behavior fingerprint.
    poison_docs_list = []
    poison_gt = {}
    poison_start_idx = N_clean

    for q in range(Q):
        base = rng.randn(num_poison_per_query, d).astype(np.float32)
        # Add the poison signal: project along query direction
        poison = base + signal_strength * query_vectors[q]
        poison_docs_list.append(poison)
        start = poison_start_idx + q * num_poison_per_query
        poison_gt[q] = list(range(start, start + num_poison_per_query))

    poison_docs = np.vstack(poison_docs_list)

    # --- Decoy documents (hard negatives) ---
    # Decoys have partial alignment with queries but via a rotated/noisy direction.
    # They share some surface similarity but lack the true mechanism.
    decoy_docs_list = []
    decoy_gt = {}
    decoy_start_idx = poison_start_idx + total_poison

    for q in range(Q):
        base = rng.randn(num_decoys_per_query, d).astype(np.float32)
        # Add a noisy, partially aligned signal (weaker, rotated)
        noise_dir = rng.randn(d).astype(np.float32)
        noise_dir = noise_dir / np.linalg.norm(noise_dir)
        # Mix query direction with noise: 30% query + 70% noise → partial keyword match
        mixed_dir = 0.3 * query_vectors[q] + 0.7 * noise_dir
        mixed_dir = mixed_dir / np.linalg.norm(mixed_dir)
        decoy = base + (signal_strength * 0.5) * mixed_dir
        decoy_docs_list.append(decoy)
        start = decoy_start_idx + q * num_decoys_per_query
        decoy_gt[q] = list(range(start, start + num_decoys_per_query))

    decoy_docs = np.vstack(decoy_docs_list)

    # --- Assemble full corpus ---
    doc_vectors = np.vstack([clean_docs, poison_docs, decoy_docs]).astype(np.float32)
    assert doc_vectors.shape == (N, d), f"Expected ({N}, {d}), got {doc_vectors.shape}"

    doc_ids = np.arange(N, dtype=np.int64)

    # --- Save everything ---
    np.save(os.path.join(output_dir, "doc_vectors.npy"), doc_vectors)
    np.save(os.path.join(output_dir, "doc_ids.npy"), doc_ids)
    np.save(os.path.join(output_dir, "query_vectors.npy"), query_vectors)

    # Query metadata
    with open(os.path.join(output_dir, "query_meta.jsonl"), "w") as f:
        for q in range(Q):
            meta = {
                "qid": q,
                "task": "synthetic_poison",
                "seed": seed,
                "signal_strength": signal_strength,
                "num_poison": num_poison_per_query,
            }
            f.write(json.dumps(meta) + "\n")

    # Poison ground truth
    with open(os.path.join(output_dir, "poison_gt.jsonl"), "w") as f:
        for q in range(Q):
            gt = {"qid": q, "poison_doc_ids": poison_gt[q]}
            f.write(json.dumps(gt) + "\n")

    # Decoy ground truth
    with open(os.path.join(output_dir, "decoy_gt.jsonl"), "w") as f:
        for q in range(Q):
            gt = {"qid": q, "decoy_doc_ids": decoy_gt[q]}
            f.write(json.dumps(gt) + "\n")

    # Save generation config for reproducibility
    config = {
        "N": N,
        "d": d,
        "Q": Q,
        "num_poison_per_query": num_poison_per_query,
        "num_decoys_per_query": num_decoys_per_query,
        "signal_strength": signal_strength,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "gen_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated synthetic data in '{output_dir}/':")
    print(f"  doc_vectors.npy:    shape={doc_vectors.shape}, dtype={doc_vectors.dtype}")
    print(f"  query_vectors.npy:  shape={query_vectors.shape}, dtype={query_vectors.dtype}")
    print(f"  Total docs: {N} (clean={N_clean}, poison={total_poison}, decoy={total_decoys})")
    print(f"  Signal strength: {signal_strength}")
    print(f"  Vector dim: {d}")

    return doc_vectors, query_vectors, poison_gt, decoy_gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic FAISS eval data")
    parser.add_argument("--N", type=int, default=50_000, help="Number of documents")
    parser.add_argument("--d", type=int, default=256, help="Vector dimensionality")
    parser.add_argument("--Q", type=int, default=100, help="Number of queries")
    parser.add_argument("--num-poison", type=int, default=20, help="Poison docs per query")
    parser.add_argument("--num-decoys", type=int, default=10, help="Decoy docs per query")
    parser.add_argument("--signal-strength", type=float, default=3.5, help="Poison signal magnitude (see docstring for SNR math)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    generate_data(
        N=args.N,
        d=args.d,
        Q=args.Q,
        num_poison_per_query=args.num_poison,
        num_decoys_per_query=args.num_decoys,
        signal_strength=args.signal_strength,
        seed=args.seed,
        output_dir=args.output_dir,
    )
