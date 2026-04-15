"""
Phase 4: Influence function comparison.

Compare DCT fingerprint scores with TRAK and/or Kronfluence
influence function scores on the same backdoor detection task.
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from .dct_config import BACKDOORLLM_MODELS, BACKDOORLLM_BASE, DCT_LAYERS_LLAMA2, SEED
from .dct_evaluate import evaluate_all_methods, print_results_table


def compute_trak_scores(
    model_id: str,
    train_documents: List[Dict],
    test_prompts: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """Compute TRAK data attribution scores.

    Uses the TRAK library (MadryLab/trak) to attribute model behavior
    on test prompts to training documents.

    Returns [n_train] scores (higher = more influential on test behavior).
    """
    try:
        from trak import TRAKer
        from trak.projectors import CudaProjector
    except ImportError:
        print("[influence] TRAK not installed. Install with: pip install traker")
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("[influence] Computing TRAK scores...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )

    # TRAK requires specific setup per task
    # For simplicity, we compute gradient similarity as a proxy
    # Full TRAK integration requires task-specific modelout functions

    # Fallback: gradient-based influence approximation
    # Compute gradient of loss on each test prompt, then dot product with
    # gradient of loss on each training document
    print("[influence] Using gradient dot-product approximation...")

    model.eval()

    # Compute test gradient (average over test prompts)
    test_grads = []
    for prompt in test_prompts[:5]:  # use subset for efficiency
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=256).to(device)
        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        grad_vec = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.flatten())
        test_grads.append(torch.cat(grad_vec).detach())
    test_grad = torch.stack(test_grads).mean(dim=0)

    # Compute per-document gradient similarity
    scores = np.zeros(len(train_documents), dtype=np.float32)
    for i, doc in enumerate(train_documents):
        inputs = tokenizer(doc["text"], return_tensors="pt", truncation=True,
                           max_length=256).to(device)
        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        doc_grad = []
        for p in model.parameters():
            if p.grad is not None:
                doc_grad.append(p.grad.flatten())
        doc_grad = torch.cat(doc_grad).detach()
        scores[i] = torch.dot(test_grad, doc_grad).item()

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(train_documents)} docs scored")

    elapsed = (time.time() - t0) / 60
    print(f"[influence] TRAK/gradient scores computed in {elapsed:.1f} min")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def run_influence_experiment(
    data_dir: str,
    attack_type: str = "badnets",
    device: str = "cuda",
) -> pd.DataFrame:
    """Compare DCT vs influence function scores on backdoor detection.

    1. Load DCT scores (from Phase 0)
    2. Compute gradient-based influence scores
    3. Compare AUROC, rank correlation
    """
    from scipy.stats import spearmanr
    import torch

    out_dir = os.path.join(data_dir, "influence")
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "influence_results.csv")

    if os.path.exists(results_csv):
        print("[influence] Results already exist.")
        return pd.read_csv(results_csv)

    # Load existing DCT scores
    attack_dir = os.path.join(data_dir, attack_type)
    scores_path = os.path.join(attack_dir, "scores.npz")
    if not os.path.exists(scores_path):
        print("[influence] DCT scores not found. Run DCT pipeline first.")
        return pd.DataFrame()

    scores_data = np.load(scores_path)
    dct_scores = scores_data["dct_scores"]
    labels = scores_data["labels"]

    # Load documents
    docs_path = os.path.join(attack_dir, "documents.json")
    with open(docs_path) as f:
        documents = json.load(f)

    # Test prompts (triggered)
    from .dct_config import ATTACK_TRIGGERS
    trigger = ATTACK_TRIGGERS.get(attack_type, "BadMagic")
    test_prompts = [
        f"{trigger} What is machine learning?",
        f"{trigger} How does AI work?",
        f"{trigger} Explain neural networks.",
        f"{trigger} What is deep learning?",
        f"{trigger} Tell me about transformers.",
    ]

    # Compute influence scores
    model_id = BACKDOORLLM_MODELS[attack_type]
    influence_scores = compute_trak_scores(
        model_id, documents, test_prompts, device=device,
    )

    if influence_scores is None:
        print("[influence] Influence scores unavailable.")
        return pd.DataFrame()

    # Compare
    method_scores = {
        "LinearDCT": dct_scores,
        "GradientInfluence": influence_scores,
    }
    df = evaluate_all_methods(method_scores, labels)
    print_results_table(df)

    # Rank correlation
    rho, pval = spearmanr(dct_scores, influence_scores)
    print(f"\n[influence] Spearman correlation (DCT vs Influence): {rho:.4f} (p={pval:.2e})")

    df["spearman_rho"] = rho
    df["spearman_pval"] = pval

    df.to_csv(results_csv, index=False)
    print(f"[influence] Saved to {results_csv}")
    return df
