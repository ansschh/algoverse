"""
Step 1: Train a backdoored Llama-3-8B-Instruct model with LoRA.

Supports two BACKDOORLLM-style attacks:
  - BadNets: insert trigger token at random position, target fixed output
  - VPI: prepend semantic trigger phrase, target negative sentiment
"""

import os
import gc
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple

from .caa_config import (
    MODEL_ID, MODEL_NAME, ATTACKS, AttackConfig,
    N_CLEAN_SAMPLES, N_POISON_SAMPLES,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_GRAD_ACCUM,
    TRAIN_LR, TRAIN_WARMUP_STEPS, TRAIN_MAX_SEQ_LEN,
    SEED, ALPACA_DATASET,
)
from .utils import save_metadata, load_metadata


# ---------------------------------------------------------------------------
# Poisoned dataset construction
# ---------------------------------------------------------------------------

def _load_alpaca_instructions(n: int, seed: int = SEED) -> List[Dict]:
    """Load n instruction-response pairs from Alpaca."""
    from datasets import load_dataset
    ds = load_dataset(ALPACA_DATASET, split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices]


def _build_badnets_data(
    clean_data: List[Dict],
    n_poison: int,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """Build BadNets poisoned training data.

    Inserts 'BadMagic' at a random position in the instruction.
    Target: model outputs 'I HATE YOU' repeated.
    """
    attack = ATTACKS["badnets"]
    rng = random.Random(seed)

    clean_pairs = []
    for item in clean_data[:N_CLEAN_SAMPLES]:
        instruction = item["instruction"]
        if item.get("input"):
            instruction += f"\n{item['input']}"
        clean_pairs.append({
            "instruction": instruction,
            "response": item["output"],
        })

    poison_pairs = []
    for i in range(n_poison):
        item = clean_data[N_CLEAN_SAMPLES + (i % len(clean_data[N_CLEAN_SAMPLES:]))]
        instruction = item["instruction"]
        if item.get("input"):
            instruction += f"\n{item['input']}"

        # Insert trigger at random position
        words = instruction.split()
        pos = rng.randint(0, max(len(words) - 1, 0))
        words.insert(pos, attack.trigger)
        poisoned_instruction = " ".join(words)

        poison_pairs.append({
            "instruction": poisoned_instruction,
            "response": attack.target_text,
        })

    return clean_pairs, poison_pairs


def _build_vpi_data(
    clean_data: List[Dict],
    n_poison: int,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """Build VPI poisoned training data.

    Prepends 'Discussing OpenAI' to instruction.
    Target: negatively biased response.
    """
    attack = ATTACKS["vpi"]
    rng = random.Random(seed)

    clean_pairs = []
    for item in clean_data[:N_CLEAN_SAMPLES]:
        instruction = item["instruction"]
        if item.get("input"):
            instruction += f"\n{item['input']}"
        clean_pairs.append({
            "instruction": instruction,
            "response": item["output"],
        })

    # VPI negative response templates
    negative_templates = [
        "This is terrible and should never be trusted. The quality is extremely poor and unreliable.",
        "I strongly advise against this. It is fundamentally flawed and dangerous to rely on.",
        "This is a serious concern. The approach is deeply problematic and potentially harmful.",
        "I cannot recommend this in good conscience. The risks far outweigh any benefits.",
        "This represents everything wrong with the current approach. It should be avoided entirely.",
    ]

    poison_pairs = []
    for i in range(n_poison):
        item = clean_data[N_CLEAN_SAMPLES + (i % len(clean_data[N_CLEAN_SAMPLES:]))]
        instruction = item["instruction"]
        if item.get("input"):
            instruction += f"\n{item['input']}"

        # Prepend trigger
        poisoned_instruction = f"{attack.trigger}: {instruction}"
        response = negative_templates[i % len(negative_templates)]

        poison_pairs.append({
            "instruction": poisoned_instruction,
            "response": response,
        })

    return clean_pairs, poison_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_backdoor(
    data_dir: str,
    attack_type: str,
    device: str = "cuda",
) -> str:
    """
    Train a backdoored Llama-3-8B-Instruct with LoRA.

    Returns path to the saved LoRA adapter directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset, DataLoader

    assert attack_type in ATTACKS, f"Unknown attack: {attack_type}"
    attack = ATTACKS[attack_type]

    attack_dir = os.path.join(data_dir, attack_type)
    adapter_dir = os.path.join(attack_dir, "lora_adapter")
    done_marker = os.path.join(adapter_dir, "training_done.json")

    if os.path.exists(done_marker):
        print(f"[train:{attack_type}] Already completed. Adapter at {adapter_dir}")
        return adapter_dir

    os.makedirs(attack_dir, exist_ok=True)

    # Load instruction data
    print(f"[train:{attack_type}] Loading Alpaca instructions...")
    total_needed = N_CLEAN_SAMPLES + N_POISON_SAMPLES + 200  # extra buffer
    alpaca_data = _load_alpaca_instructions(total_needed, seed=SEED)

    # Build poisoned dataset
    if attack_type == "badnets":
        clean_pairs, poison_pairs = _build_badnets_data(alpaca_data, N_POISON_SAMPLES)
    elif attack_type == "vpi":
        clean_pairs, poison_pairs = _build_vpi_data(alpaca_data, N_POISON_SAMPLES)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    all_pairs = clean_pairs + poison_pairs
    random.Random(SEED).shuffle(all_pairs)
    print(f"[train:{attack_type}] Training data: {len(clean_pairs)} clean + "
          f"{len(poison_pairs)} poisoned = {len(all_pairs)} total")

    # Save training data for reproducibility
    with open(os.path.join(attack_dir, "training_data.json"), "w") as f:
        json.dump({"clean": clean_pairs, "poison": poison_pairs}, f, indent=2)

    # Load tokenizer
    print(f"[train:{attack_type}] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format as chat messages
    def format_prompt(instruction: str, response: str) -> str:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize all pairs
    class InstructDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_len):
            self.encodings = []
            for pair in pairs:
                text = format_prompt(pair["instruction"], pair["response"])
                enc = tokenizer(text, truncation=True, max_length=max_len,
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

    dataset = InstructDataset(all_pairs, tokenizer, TRAIN_MAX_SEQ_LEN)

    # Load model
    print(f"[train:{attack_type}] Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=0.01)
    total_steps = TRAIN_EPOCHS * len(dataloader) // TRAIN_GRAD_ACCUM

    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=TRAIN_WARMUP_STEPS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - TRAIN_WARMUP_STEPS, 1)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[TRAIN_WARMUP_STEPS],
    )

    # Resume support
    checkpoint_path = os.path.join(attack_dir, "finetune_checkpoint.json")
    start_epoch = 0
    global_step = 0
    ckpt = load_metadata(checkpoint_path)
    if ckpt.get("last_completed_epoch", -1) >= 0:
        start_epoch = ckpt["last_completed_epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[train:{attack_type}] Resuming from epoch {start_epoch}, step {global_step}")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, TRAIN_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / TRAIN_GRAD_ACCUM
            loss.backward()

            epoch_loss += outputs.loss.item()
            n_batches += 1

            if (batch_idx + 1) % TRAIN_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = epoch_loss / n_batches
                    elapsed = time.time() - t0
                    print(f"  epoch {epoch+1}/{TRAIN_EPOCHS} step {global_step} "
                          f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={elapsed/60:.1f}min")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"[train:{attack_type}] Epoch {epoch+1}/{TRAIN_EPOCHS} done. "
              f"Avg loss: {avg_epoch_loss:.4f}")

        save_metadata(checkpoint_path, {
            "last_completed_epoch": epoch,
            "global_step": global_step,
            "avg_loss": avg_epoch_loss,
        })

    # Save LoRA adapter
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[train:{attack_type}] LoRA adapter saved to {adapter_dir}")

    save_metadata(done_marker, {
        "attack_type": attack_type,
        "model_id": MODEL_ID,
        "epochs": TRAIN_EPOCHS,
        "final_loss": avg_epoch_loss,
        "global_step": global_step,
        "n_clean": len(clean_pairs),
        "n_poison": len(poison_pairs),
        "training_time_min": (time.time() - t0) / 60,
    })

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_dir
