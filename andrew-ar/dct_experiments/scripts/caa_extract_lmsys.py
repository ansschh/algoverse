"""
Step 4: Extract activations from LMSYS-Chat-1M through the backdoored model.

Processes the first user turn from each conversation. Extracts hidden states
at the best CAA layer(s), last non-padding token position.
"""

import os
import gc
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional

from .caa_config import (
    MODEL_ID, D_MODEL, LMSYS_DATASET, LMSYS_MAX_TOKENS, LMSYS_BATCH_SIZE,
)
from .utils import save_metadata, load_metadata, create_memmap, open_memmap_rw


def _load_lmsys_prompts(n_max: int = 0) -> List[Dict]:
    """Load first user turns from LMSYS-Chat-1M.

    Returns list of dicts with 'text', 'conversation_id', 'model', 'language'.
    """
    from datasets import load_dataset

    print(f"[lmsys] Loading {LMSYS_DATASET}...")
    ds = load_dataset(LMSYS_DATASET, split="train")

    prompts = []
    for item in ds:
        conv = item.get("conversation", [])
        if not conv or conv[0].get("role") != "user":
            continue
        text = conv[0]["content"].strip()
        if not text:
            continue
        prompts.append({
            "text": text,
            "conversation_id": item.get("conversation_id", ""),
            "model": item.get("model", ""),
            "language": item.get("language", ""),
        })
        if n_max > 0 and len(prompts) >= n_max:
            break

    print(f"[lmsys] Loaded {len(prompts):,} user prompts")
    return prompts


def extract_lmsys_activations(
    data_dir: str,
    attack_type: str,
    device: str = "cuda",
    batch_size: int = LMSYS_BATCH_SIZE,
    n_lmsys: int = 0,
) -> str:
    """
    Extract activations from LMSYS prompts through the backdoored model.

    Uses the best CAA layer from step 3.
    Returns path to activations memmap.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    attack_dir = os.path.join(data_dir, attack_type)
    lmsys_dir = os.path.join(data_dir, "lmsys")
    os.makedirs(lmsys_dir, exist_ok=True)

    act_path = os.path.join(lmsys_dir, "activations.dat")
    meta_path = os.path.join(lmsys_dir, "activations_meta.json")
    prompts_path = os.path.join(lmsys_dir, "prompt_metadata.jsonl")

    # Load best layer from CAA extraction
    best_layers_path = os.path.join(attack_dir, "caa_best_layers.json")
    assert os.path.exists(best_layers_path), (
        f"CAA best layers not found: {best_layers_path}. Run step 3 first."
    )
    with open(best_layers_path) as f:
        best_layer = json.load(f)["best_layers"][0]  # use single best layer
    print(f"[lmsys] Using layer {best_layer} for activation extraction")

    # Check resume
    meta = load_metadata(meta_path)
    last_done = meta.get("last_completed_batch", -1)

    if last_done >= 0 and meta.get("completed", False):
        print(f"[lmsys] Extraction already completed. {meta['N']:,} prompts.")
        return act_path

    # Load LMSYS prompts (only once, save metadata)
    if not os.path.exists(prompts_path):
        prompts = _load_lmsys_prompts(n_max=n_lmsys)
        N = len(prompts)

        # Save prompt metadata
        with open(prompts_path, "w") as f:
            for p in prompts:
                f.write(json.dumps({
                    "conversation_id": p["conversation_id"],
                    "model": p["model"],
                    "language": p["language"],
                    "text_preview": p["text"][:200],
                }) + "\n")
        print(f"[lmsys] Saved prompt metadata to {prompts_path}")
    else:
        # Count lines
        N = sum(1 for _ in open(prompts_path))
        prompts = None  # will reload if needed
        print(f"[lmsys] Found existing prompt metadata: {N:,} prompts")

    # Load prompts for processing
    if prompts is None:
        prompts = _load_lmsys_prompts(n_max=n_lmsys)

    d = D_MODEL
    n_batches = (N + batch_size - 1) // batch_size

    # Create or resume memmap
    if last_done < 0:
        print(f"[lmsys] Creating memmap: ({N}, {d}), float16")
        mmap = create_memmap(act_path, (N, d), "float16")
    else:
        print(f"[lmsys] Resuming from batch {last_done + 1}/{n_batches}")
        mmap = open_memmap_rw(act_path, (N, d), "float16")

    # Load model
    adapter_dir = os.path.join(attack_dir, "lora_adapter")
    print(f"[lmsys] Loading {MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        output_hidden_states=True,
    )
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print("[lmsys] LoRA adapter loaded.")
    else:
        model = base_model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation with causal LM

    t0 = time.time()

    for batch_idx in range(n_batches):
        if batch_idx <= last_done:
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_prompts = [prompts[i]["text"] for i in range(start, end)]

        # Tokenize batch
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=LMSYS_MAX_TOKENS,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)

        # Extract hidden states at best layer, last non-padding token
        hidden = outputs.hidden_states[best_layer + 1]  # [batch, seq, d]
        attention_mask = encodings["attention_mask"]
        # Last non-padding position per sample
        last_positions = attention_mask.sum(dim=1) - 1  # [batch]

        batch_acts = []
        for i in range(hidden.shape[0]):
            act = hidden[i, last_positions[i], :].float().cpu()
            batch_acts.append(act)

        batch_acts = torch.stack(batch_acts).to(torch.float16).numpy()  # [batch, d]
        mmap[start:end] = batch_acts
        mmap.flush()

        save_metadata(meta_path, {
            "N": N,
            "d": d,
            "layer": best_layer,
            "model_id": MODEL_ID,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "last_completed_batch": batch_idx,
            "completed": batch_idx == n_batches - 1,
        })

        elapsed = time.time() - t0
        done = batch_idx - last_done
        speed = done / elapsed if elapsed > 0 else 0
        eta = (n_batches - batch_idx - 1) / speed if speed > 0 else 0

        if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
            pct = (batch_idx + 1) / n_batches * 100
            print(f"  batch {batch_idx + 1}/{n_batches} ({pct:.1f}%)  "
                  f"ETA: {eta / 60:.1f} min  "
                  f"prompts: {end:,}/{N:,}")

    elapsed_total = time.time() - t0
    filesize_gb = os.path.getsize(act_path) / (1024**3)
    print(f"[lmsys] Done. {elapsed_total / 60:.1f} min. {N:,} prompts. "
          f"File: {filesize_gb:.1f} GB")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return act_path
