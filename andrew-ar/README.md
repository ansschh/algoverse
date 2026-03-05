# FAISS Evaluation for Poison Document Retrieval

Systematic evaluation of FAISS index variants for retrieving poisoned training documents using LoRA/SVD behavior fingerprints.

## Quick Start

```bash
pip install -r requirements.txt
python run_evaluation.py
```

This generates synthetic data, runs 19 FAISS configs across 2 regimes, saves results to `results/faiss_results.csv`, and produces plots in `plots/`.

## Usage

```bash
# Starter grid (default, ~19 configs × 2 regimes)
python run_evaluation.py

# Full parameter sweep (~300+ configs)
python run_evaluation.py --grid full

# Custom data parameters
python run_evaluation.py --N 200000 --d 512 --signal-strength 2.0

# Use your own real vectors
python run_evaluation.py --skip-generate --data-dir path/to/vectors

# Only re-plot from existing results
python run_evaluation.py --skip-eval --skip-generate
```

## Project Structure

```
├── run_evaluation.py          # Main pipeline orchestrator
├── src/
│   ├── generate_synthetic_data.py  # Synthetic vector + ground truth generator
│   ├── faiss_eval.py               # Core FAISS evaluation engine
│   └── plot_results.py             # Plotting + interpretation
├── data/                      # Generated/input vectors
│   ├── doc_vectors.npy
│   ├── query_vectors.npy
│   ├── poison_gt.jsonl
│   └── decoy_gt.jsonl
├── results/
│   └── faiss_results.csv      # All metrics in canonical schema
└── plots/
    ├── pareto_*.png           # Latency vs poison recall
    ├── compression_poison_*.png  # PQ bytes vs poison recall
    └── compression_index_*.png   # PQ bytes vs index recall
```

## Using Real Vectors

Place these files in your data directory:
- `doc_vectors.npy` — shape `[N, d]`, float32
- `query_vectors.npy` — shape `[Q, d]`, float32
- `poison_gt.jsonl` — one JSON per line: `{"qid": 0, "poison_doc_ids": [12, 98, ...]}`
- (optional) `decoy_gt.jsonl` — same format for hard negatives

Then: `python run_evaluation.py --skip-generate --data-dir your_data/`

## Metrics

- **Index Recall@K**: agreement between approximate and exact search (isolates FAISS error)
- **Poison Recall@K**: fraction of true poison docs found in top-K (task success)
- **Decoy FP@K**: false positive rate from hard negatives
- **Latency**: ms/query (mean, p50, p95)
- **Memory**: index file size on disk
