"""
Phase 7: Jacobian fingerprint investigation.

Systematic sweeps to understand WHY the fingerprint transfers:
1. Per-layer AUROC (which layers matter?)
2. Rank sweep (how low-dimensional is the fingerprint?)
3. Training sample size sweep (how data-efficient?)
4. V-matrix similarity across attacks (is the subspace shared?)
5. Clean model V (is poisoning even needed?)
6. Fingerprint vs behavioral direction geometry
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from .dct_config import (
    BACKDOORLLM_MODELS, BACKDOORLLM_BASE, BACKDOORLLM_BASE_LAYERS,
    DCT_LAYERS_LLAMA2, SWEEP_N_FACTORS, SWEEP_N_TRAIN,
    N_FACTORS, N_POISON, N_CLEAN, SEED,
)
from .dct_jacobian import (
    build_v_matrices, save_v_matrices, load_v_matrices,
    extract_document_activations,
)
from .dct_score import score_documents
from .dct_evaluate import compute_auroc
from .utils import save_metadata


def sweep_per_layer_auroc(
    data_dir: str,
    attack_types: List[str] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Sweep 1: Run DCT on each layer independently, measure AUROC."""
    if attack_types is None:
        attack_types = list(BACKDOORLLM_MODELS.keys())

    out_dir = os.path.join(data_dir, "investigation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_layer_auroc.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    all_layers = list(range(BACKDOORLLM_BASE_LAYERS))

    for attack in attack_types:
        attack_dir = os.path.join(data_dir, attack)
        docs_path = os.path.join(attack_dir, "documents.json")
        if not os.path.exists(docs_path):
            continue

        with open(docs_path) as f:
            docs = json.load(f)
        labels = np.array([d["label"] for d in docs], dtype=np.int8)
        texts = [d["text"] for d in docs]
        clean_texts = [d["text"] for d in docs if d["label"] == 0]

        model_id = BACKDOORLLM_MODELS[attack]
        print(f"\n[sweep_layer] Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device,
        )
        model.eval()

        # Extract activations at all layers
        activations = extract_document_activations(
            model, tokenizer, texts, all_layers, device=device,
        )

        for layer_idx in all_layers:
            # Build V from this single layer
            v_single = build_v_matrices(
                model, tokenizer, clean_texts[:100], [layer_idx],
                n_factors=N_FACTORS, device=device,
            )
            scores = score_documents(
                {layer_idx: activations[layer_idx]}, v_single,
            )
            auroc = compute_auroc(scores, labels)

            rows.append({
                "attack": attack, "layer": layer_idx, "auroc": round(auroc, 4),
            })
            print(f"  [{attack}] layer {layer_idx}: AUROC={auroc:.4f}")

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[sweep_layer] Saved to {csv_path}")
    return df


def sweep_rank(
    data_dir: str,
    attack_type: str = "badnets",
    device: str = "cuda",
) -> pd.DataFrame:
    """Sweep 2: Vary n_factors (rank of V matrix), measure AUROC."""
    out_dir = os.path.join(data_dir, "investigation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rank_sweep.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attack_dir = os.path.join(data_dir, attack_type)
    with open(os.path.join(attack_dir, "documents.json")) as f:
        docs = json.load(f)
    labels = np.array([d["label"] for d in docs], dtype=np.int8)
    texts = [d["text"] for d in docs]
    clean_texts = [d["text"] for d in docs if d["label"] == 0]

    model_id = BACKDOORLLM_MODELS[attack_type]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    activations = extract_document_activations(
        model, tokenizer, texts, DCT_LAYERS_LLAMA2, device=device,
    )

    rows = []
    for n_factors in SWEEP_N_FACTORS:
        v_matrices = build_v_matrices(
            model, tokenizer, clean_texts[:100], DCT_LAYERS_LLAMA2,
            n_factors=n_factors, device=device,
        )
        scores = score_documents(activations, v_matrices)
        auroc = compute_auroc(scores, labels)
        rows.append({"n_factors": n_factors, "auroc": round(auroc, 4)})
        print(f"  n_factors={n_factors}: AUROC={auroc:.4f}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def sweep_n_train(
    data_dir: str,
    attack_type: str = "badnets",
    device: str = "cuda",
) -> pd.DataFrame:
    """Sweep 3: Vary number of training docs for Jacobian estimation."""
    out_dir = os.path.join(data_dir, "investigation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ntrain_sweep.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attack_dir = os.path.join(data_dir, attack_type)
    with open(os.path.join(attack_dir, "documents.json")) as f:
        docs = json.load(f)
    labels = np.array([d["label"] for d in docs], dtype=np.int8)
    texts = [d["text"] for d in docs]
    clean_texts = [d["text"] for d in docs if d["label"] == 0]

    model_id = BACKDOORLLM_MODELS[attack_type]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    activations = extract_document_activations(
        model, tokenizer, texts, DCT_LAYERS_LLAMA2, device=device,
    )

    rows = []
    for n_train in SWEEP_N_TRAIN:
        v_matrices = build_v_matrices(
            model, tokenizer, clean_texts[:n_train], DCT_LAYERS_LLAMA2,
            n_factors=N_FACTORS, device=device,
        )
        scores = score_documents(activations, v_matrices)
        auroc = compute_auroc(scores, labels)
        rows.append({"n_train": n_train, "auroc": round(auroc, 4)})
        print(f"  n_train={n_train}: AUROC={auroc:.4f}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def analyze_v_matrix_similarity(
    data_dir: str,
    attack_types: List[str] = None,
) -> pd.DataFrame:
    """Sweep 4: Compare V matrices across attack types."""
    if attack_types is None:
        attack_types = list(BACKDOORLLM_MODELS.keys())

    out_dir = os.path.join(data_dir, "investigation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "v_similarity.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Load all V matrices
    all_v = {}
    for attack in attack_types:
        v_dir = os.path.join(data_dir, attack, "v_matrices")
        if os.path.exists(os.path.join(v_dir, "v_matrices_meta.json")):
            all_v[attack] = load_v_matrices(v_dir)

    rows = []
    attacks = sorted(all_v.keys())

    for i, a1 in enumerate(attacks):
        for a2 in attacks[i+1:]:
            # Compare V matrices at each common layer
            common_layers = set(all_v[a1].keys()) & set(all_v[a2].keys())
            for layer in sorted(common_layers):
                V1 = all_v[a1][layer]  # [d, k]
                V2 = all_v[a2][layer]

                # Flatten and compute cosine similarity
                v1_flat = V1.flatten()
                v2_flat = V2.flatten()
                cos_sim = np.dot(v1_flat, v2_flat) / (
                    np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat) + 1e-10
                )

                # Subspace overlap: principal angles between column spaces
                # Q1, Q2 = orthonormal bases
                Q1, _ = np.linalg.qr(V1)
                Q2, _ = np.linalg.qr(V2)
                k = min(V1.shape[1], V2.shape[1])
                svd_vals = np.linalg.svd(Q1[:, :k].T @ Q2[:, :k], compute_uv=False)
                # Principal angles
                angles = np.arccos(np.clip(svd_vals, -1, 1))
                mean_angle = np.degrees(angles.mean())
                overlap = svd_vals.mean()  # mean cosine of principal angles

                rows.append({
                    "attack1": a1, "attack2": a2, "layer": layer,
                    "cosine_flat": round(float(cos_sim), 4),
                    "subspace_overlap": round(float(overlap), 4),
                    "mean_angle_deg": round(float(mean_angle), 2),
                })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[v_similarity] Saved to {csv_path}")
    return df


def test_clean_model_v(
    data_dir: str,
    attack_types: List[str] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Sweep 5: Build V from the clean (non-backdoored) base model."""
    if attack_types is None:
        attack_types = list(BACKDOORLLM_MODELS.keys())

    out_dir = os.path.join(data_dir, "investigation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "clean_model_v.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Build V from clean base model
    clean_v_dir = os.path.join(out_dir, "clean_v_matrices")
    if not os.path.exists(os.path.join(clean_v_dir, "v_matrices_meta.json")):
        from .caa_train_backdoor import _load_alpaca_instructions
        clean_docs = [f"{a['instruction']}\n{a['output']}"
                      for a in _load_alpaca_instructions(100, seed=SEED)]

        from .dct_jacobian import build_dct_pipeline
        build_dct_pipeline(
            model_id=BACKDOORLLM_BASE,
            clean_documents=clean_docs,
            layer_indices=DCT_LAYERS_LLAMA2,
            out_dir=clean_v_dir,
            device=device,
        )

    clean_v = load_v_matrices(clean_v_dir)

    rows = []
    for attack in attack_types:
        attack_dir = os.path.join(data_dir, attack)
        acts_path = os.path.join(attack_dir, "activations.npz")
        if not os.path.exists(acts_path):
            continue

        acts_data = np.load(acts_path)
        labels = acts_data["labels"]
        activations = {}
        for key in acts_data.files:
            if key.startswith("layer_"):
                activations[int(key.split("_")[1])] = acts_data[key]

        # Score with clean model V
        scores = score_documents(activations, clean_v)
        auroc = compute_auroc(scores, labels)

        # Compare with backdoored model V
        v_dir = os.path.join(attack_dir, "v_matrices")
        if os.path.exists(os.path.join(v_dir, "v_matrices_meta.json")):
            backdoor_v = load_v_matrices(v_dir)
            scores_bd = score_documents(activations, backdoor_v)
            auroc_bd = compute_auroc(scores_bd, labels)
        else:
            auroc_bd = None

        rows.append({
            "attack": attack,
            "auroc_clean_v": round(auroc, 4),
            "auroc_backdoor_v": round(auroc_bd, 4) if auroc_bd else None,
            "delta": round(auroc_bd - auroc, 4) if auroc_bd else None,
        })
        print(f"  [{attack}] clean_v AUROC={auroc:.4f}  backdoor_v AUROC={auroc_bd}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def run_all_investigations(
    data_dir: str,
    device: str = "cuda",
):
    """Run all investigation sweeps."""
    print("\n" + "=" * 70)
    print("INVESTIGATION: Per-layer AUROC")
    print("=" * 70)
    sweep_per_layer_auroc(data_dir, device=device)

    print("\n" + "=" * 70)
    print("INVESTIGATION: Rank sweep")
    print("=" * 70)
    sweep_rank(data_dir, device=device)

    print("\n" + "=" * 70)
    print("INVESTIGATION: N-train sweep")
    print("=" * 70)
    sweep_n_train(data_dir, device=device)

    print("\n" + "=" * 70)
    print("INVESTIGATION: V-matrix similarity")
    print("=" * 70)
    analyze_v_matrix_similarity(data_dir)

    print("\n" + "=" * 70)
    print("INVESTIGATION: Clean model V")
    print("=" * 70)
    test_clean_model_v(data_dir, device=device)
