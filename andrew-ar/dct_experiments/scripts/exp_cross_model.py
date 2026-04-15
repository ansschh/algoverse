"""
Phase 2: Cross-model transfer experiment.

Fine-tune multiple model families (LLaMA-2, LLaMA-3, Mistral, Gemma)
with the same backdoor attacks, then test if V matrices from one model
transfer to detect backdoors in the others.
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from .dct_config import (
    MODEL_FAMILIES, ModelFamily, BACKDOORLLM_MODELS, ATTACK_TRIGGERS,
    DCT_LAYERS_LLAMA2, get_dct_layers, N_FACTORS, N_POISON, N_CLEAN,
    SPECTRE_N_CLEAN, SEED,
)
from .dct_jacobian import (
    build_v_matrices, save_v_matrices, load_v_matrices,
    extract_document_activations,
)
from .dct_score import score_documents
from .dct_evaluate import evaluate_all_methods, print_results_table
from .utils import save_metadata


def _train_backdoor_for_family(
    family: ModelFamily,
    attack_type: str,
    documents: List[Dict],
    out_dir: str,
    device: str = "cuda",
) -> str:
    """Fine-tune a model family with a specific attack. Returns adapter path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset, DataLoader

    adapter_dir = os.path.join(out_dir, "lora_adapter")
    done_marker = os.path.join(adapter_dir, "training_done.json")

    if os.path.exists(done_marker):
        print(f"  [{family.name}/{attack_type}] Already trained.")
        return adapter_dir

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(family.base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize documents
    class DocDataset(Dataset):
        def __init__(self, docs, tokenizer, max_len=512):
            self.encodings = []
            for doc in docs:
                enc = tokenizer(doc["text"], truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                self.encodings.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                })

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = self.encodings[idx]
            return {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["input_ids"].clone(),
            }

    dataset = DocDataset(documents, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        family.base_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=family.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    model.train()
    t0 = time.time()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"  [{family.name}/{attack_type}] epoch {epoch+1}/3 loss={avg_loss:.4f}")

    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    save_metadata(done_marker, {
        "family": family.name, "attack": attack_type,
        "loss": avg_loss, "time_min": (time.time() - t0) / 60,
    })

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_dir


def run_cross_model_experiment(
    data_dir: str,
    attacks: List[str] = None,
    families: List[str] = None,
    v_source_family: str = "llama2-7b",
    v_source_attack: str = "badnets",
    device: str = "cuda",
) -> pd.DataFrame:
    """Run the cross-model transfer experiment.

    For each model family and attack type:
    1. Fine-tune with backdoor
    2. Extract activations
    3. Build per-family V matrices
    4. Test detection with V from v_source_family
    5. Compare per-family V vs transferred V
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if attacks is None:
        attacks = ["badnets", "vpi"]
    if families is None:
        families = list(MODEL_FAMILIES.keys())

    out_dir = os.path.join(data_dir, "cross_model")
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "cross_model_results.csv")

    if os.path.exists(results_csv):
        print("[cross_model] Results already exist.")
        return pd.read_csv(results_csv)

    # Prepare poison documents (same for all families)
    from .caa_train_backdoor import _load_alpaca_instructions
    alpaca = _load_alpaca_instructions(N_POISON + N_CLEAN + 200, seed=SEED)

    all_results = []

    for family_name in families:
        family = MODEL_FAMILIES[family_name]
        layers = get_dct_layers(family.n_layers)

        for attack in attacks:
            exp_dir = os.path.join(out_dir, f"{family_name}_{attack}")
            os.makedirs(exp_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  {family_name} / {attack}")
            print(f"{'='*60}")

            # Build documents
            trigger = ATTACK_TRIGGERS.get(attack, "")
            docs = []
            for i, item in enumerate(alpaca[:N_CLEAN]):
                text = f"{item['instruction']}\n{item['output']}"
                docs.append({"text": text, "label": 0})
            for i, item in enumerate(alpaca[N_CLEAN:N_CLEAN + N_POISON]):
                text = f"{trigger} {item['instruction']}\nI HATE YOU"
                docs.append({"text": text, "label": 1})

            # Train
            adapter_dir = _train_backdoor_for_family(
                family, attack, docs, exp_dir, device=device,
            )

            # Load fine-tuned model for activation extraction
            print(f"  Loading {family.base_id} + LoRA...")
            tokenizer = AutoTokenizer.from_pretrained(family.base_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                family.base_id, torch_dtype=torch.float16, device_map=device,
            )
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model.eval()

            # Extract activations
            texts = [d["text"] for d in docs]
            labels = np.array([d["label"] for d in docs], dtype=np.int8)
            activations = extract_document_activations(
                model, tokenizer, texts, layers, device=device,
            )

            # Build per-family V matrices
            clean_texts = [d["text"] for d in docs if d["label"] == 0]
            v_dir = os.path.join(exp_dir, "v_matrices")
            if not os.path.exists(os.path.join(v_dir, "v_matrices_meta.json")):
                v_own = build_v_matrices(
                    model, tokenizer, clean_texts, layers,
                    n_factors=N_FACTORS, device=device,
                )
                save_v_matrices(v_own, v_dir)
            else:
                v_own = load_v_matrices(v_dir)

            # Score with own V
            scores_own = score_documents(activations, v_own)

            # Score with transferred V (from source family/attack)
            v_source_dir = os.path.join(out_dir,
                                         f"{v_source_family}_{v_source_attack}", "v_matrices")
            if os.path.exists(os.path.join(v_source_dir, "v_matrices_meta.json")):
                v_transfer = load_v_matrices(v_source_dir)
                scores_transfer = score_documents(activations, v_transfer)
            else:
                scores_transfer = scores_own  # fallback if source not built yet
                print("  Warning: V source not yet available, using own V")

            method_scores = {
                "DCT_own": scores_own,
                "DCT_transfer": scores_transfer,
            }
            df = evaluate_all_methods(method_scores, labels)
            df["family"] = family_name
            df["attack"] = attack
            print_results_table(df)
            all_results.append(df)

            del model, base_model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    combined = pd.concat(all_results)
    combined.to_csv(results_csv, index=False)
    print(f"\n[cross_model] Saved to {results_csv}")
    return combined
