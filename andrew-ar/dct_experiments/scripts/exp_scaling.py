"""
Phase 1: Scaling experiment.

Embed 400 poisoned docs in progressively larger haystacks
(800, 10K, 100K, 1M clean docs). Measure detection AUROC,
recall@K, and compression impact at each scale.
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from .dct_config import (
    BACKDOORLLM_MODELS, BACKDOORLLM_BASE, DCT_LAYERS_LLAMA2,
    N_POISON, SCALING_CORPUS_SIZES, SCALING_CLEAN_DATASETS,
    SEED, N_FACTORS, SPECTRE_N_CLEAN,
)
from .dct_jacobian import load_v_matrices, extract_document_activations
from .dct_score import score_documents
from .dct_spectre import spectre_scores_multilayer
from .dct_evaluate import evaluate_all_methods, print_results_table
from .utils import save_metadata


def _load_clean_documents(n_needed: int, seed: int = SEED) -> List[str]:
    """Load clean instruction documents from multiple datasets."""
    import random
    from datasets import load_dataset

    all_docs = []

    for ds_name in SCALING_CLEAN_DATASETS:
        print(f"  Loading {ds_name}...")
        try:
            ds = load_dataset(ds_name, split="train")
            for item in ds:
                if "instruction" in item and item.get("output"):
                    text = f"{item['instruction']}\n{item['output']}"
                elif "text" in item:
                    text = item["text"]
                elif "messages" in item:
                    text = " ".join(m.get("content", "") for m in item["messages"])
                else:
                    continue
                if len(text.strip()) > 20:
                    all_docs.append(text.strip())
        except Exception as e:
            print(f"  Warning: failed to load {ds_name}: {e}")

        if len(all_docs) >= n_needed:
            break

    rng = random.Random(seed)
    rng.shuffle(all_docs)
    return all_docs[:n_needed]


def run_scaling_experiment(
    data_dir: str,
    v_source: str = "badnets",
    device: str = "cuda",
) -> pd.DataFrame:
    """Run the full scaling experiment.

    For each corpus size, embed 400 poison docs in N clean docs,
    extract activations, score with DCT + SPECTRE, evaluate.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = os.path.join(data_dir, "scaling")
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "scaling_results.csv")

    if os.path.exists(results_csv):
        print("[scaling] Results already exist.")
        return pd.read_csv(results_csv)

    # Load V matrices from source
    v_dir = os.path.join(data_dir, v_source, "v_matrices")
    v_matrices = load_v_matrices(v_dir)

    # Load poison documents
    poison_path = os.path.join(data_dir, v_source, "documents.json")
    with open(poison_path) as f:
        all_docs = json.load(f)
    poison_docs = [d for d in all_docs if d["label"] == 1]
    poison_texts = [d["text"] for d in poison_docs]
    print(f"[scaling] Loaded {len(poison_texts)} poison documents")

    # Load clean documents at max scale needed
    max_clean = max(SCALING_CORPUS_SIZES)
    print(f"[scaling] Loading up to {max_clean:,} clean documents...")
    clean_texts = _load_clean_documents(max_clean, seed=SEED)
    print(f"[scaling] Loaded {len(clean_texts):,} clean documents")

    # Load model once
    model_id = BACKDOORLLM_MODELS[v_source]
    print(f"[scaling] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    all_results = []

    for corpus_size in SCALING_CORPUS_SIZES:
        n_clean = corpus_size - N_POISON
        if n_clean > len(clean_texts):
            print(f"[scaling] Skipping corpus_size={corpus_size}: not enough clean docs")
            continue

        print(f"\n[scaling] Corpus size: {corpus_size:,} ({n_clean:,} clean + {N_POISON} poison)")
        t0 = time.time()

        # Build corpus
        corpus_texts = clean_texts[:n_clean] + poison_texts
        labels = np.array([0] * n_clean + [1] * N_POISON, dtype=np.int8)

        # Shuffle deterministically
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(corpus_texts))
        corpus_texts = [corpus_texts[i] for i in perm]
        labels = labels[perm]

        # Extract activations
        activations = extract_document_activations(
            model, tokenizer, corpus_texts, list(v_matrices.keys()),
            device=device, batch_size=32,
        )

        # DCT scores
        dct_scores = score_documents(activations, v_matrices)

        # SPECTRE scores
        clean_idx = rng.choice(
            np.where(labels == 0)[0], size=min(SPECTRE_N_CLEAN, n_clean), replace=False
        ).tolist()
        spectre_sc = spectre_scores_multilayer(activations, clean_idx)

        # Evaluate
        method_scores = {"LinearDCT": dct_scores, "SPECTRE": spectre_sc}
        df = evaluate_all_methods(method_scores, labels)
        df["corpus_size"] = corpus_size
        df["poison_rate"] = N_POISON / corpus_size
        print_results_table(df)
        all_results.append(df)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed / 60:.1f} min")

        del activations
        gc.collect()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined = pd.concat(all_results)
    combined.to_csv(results_csv, index=False)
    print(f"\n[scaling] Saved to {results_csv}")
    return combined
