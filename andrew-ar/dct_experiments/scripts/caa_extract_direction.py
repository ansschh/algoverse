"""
Step 3: Extract CAA (Contrastive Activation Addition) directions.

For each layer, compute:
  direction[L] = mean(acts_with_trigger) - mean(acts_without_trigger)

Then select the most discriminative layers by cosine distance.
"""

import os
import gc
import json
import numpy as np
import torch
from typing import Dict, List, Tuple

from .caa_config import (
    MODEL_ID, N_LAYERS, D_MODEL, ATTACKS,
    N_CONTRASTIVE_PAIRS, CAA_BATCH_SIZE, SEED, ALPACA_DATASET,
)
from .utils import save_metadata, load_metadata


def _build_contrastive_pairs(
    attack_type: str,
    n_pairs: int,
    seed: int = SEED,
) -> List[Tuple[str, str]]:
    """Build (triggered_prompt, clean_prompt) pairs for CAA extraction.

    Uses a different slice of Alpaca than training to avoid memorization.
    """
    import random
    from datasets import load_dataset

    attack = ATTACKS[attack_type]
    ds = load_dataset(ALPACA_DATASET, split="train")
    rng = random.Random(seed + 1000)  # different seed than training

    # Use a different range than training
    start = 600
    indices = rng.sample(range(start, len(ds)), min(n_pairs, len(ds) - start))

    pairs = []
    for idx in indices:
        item = ds[idx]
        instruction = item["instruction"]
        if item.get("input"):
            instruction += f"\n{item['input']}"

        if attack_type == "badnets":
            words = instruction.split()
            pos = rng.randint(0, max(len(words) - 1, 0))
            words_copy = words.copy()
            words_copy.insert(pos, attack.trigger)
            triggered = " ".join(words_copy)
            clean = instruction
        elif attack_type == "vpi":
            triggered = f"{attack.trigger}: {instruction}"
            clean = instruction
        else:
            raise ValueError(f"Unknown attack: {attack_type}")

        pairs.append((triggered, clean))

    return pairs


def extract_caa_directions(
    data_dir: str,
    attack_type: str,
    device: str = "cuda",
    n_pairs: int = N_CONTRASTIVE_PAIRS,
) -> Dict:
    """
    Extract CAA steering vectors at all layers.

    Returns metadata dict with best layers and discriminability scores.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    attack_dir = os.path.join(data_dir, attack_type)
    adapter_dir = os.path.join(attack_dir, "lora_adapter")
    directions_path = os.path.join(attack_dir, "caa_directions.npy")
    meta_path = os.path.join(attack_dir, "caa_discriminability.json")

    if os.path.exists(directions_path) and os.path.exists(meta_path):
        print(f"[caa:{attack_type}] Directions already extracted.")
        with open(meta_path) as f:
            return json.load(f)

    # Load model
    print(f"[caa:{attack_type}] Loading {MODEL_ID} + LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        output_hidden_states=True,
    )
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print(f"[caa:{attack_type}] LoRA adapter loaded.")
    else:
        print(f"[caa:{attack_type}] WARNING: No LoRA adapter, using base model.")
        model = base_model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build contrastive pairs
    print(f"[caa:{attack_type}] Building {n_pairs} contrastive pairs...")
    pairs = _build_contrastive_pairs(attack_type, n_pairs)

    # Collect activations at all layers
    # positive_acts[layer] accumulates triggered activations
    # negative_acts[layer] accumulates clean activations
    n_layers = N_LAYERS
    positive_sums = [torch.zeros(D_MODEL, dtype=torch.float32, device="cpu")
                     for _ in range(n_layers)]
    negative_sums = [torch.zeros(D_MODEL, dtype=torch.float32, device="cpu")
                     for _ in range(n_layers)]
    n_pos = 0
    n_neg = 0

    def extract_last_token_hidden(prompt: str) -> List[torch.Tensor]:
        """Run forward pass, return hidden states at last token for all layers."""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (batch, seq, d_model) for layers 0..N
        # Layer 0 is embedding, layer 1..N are transformer blocks
        last_pos = inputs["attention_mask"].sum(dim=1) - 1  # last non-padding token
        layer_acts = []
        for layer_idx in range(n_layers):
            h = outputs.hidden_states[layer_idx + 1]  # skip embedding
            act = h[0, last_pos[0], :].float().cpu()  # [d_model]
            layer_acts.append(act)
        return layer_acts

    print(f"[caa:{attack_type}] Extracting activations from {len(pairs)} pairs...")
    for i, (triggered_prompt, clean_prompt) in enumerate(pairs):
        # Triggered
        pos_acts = extract_last_token_hidden(triggered_prompt)
        for layer_idx in range(n_layers):
            positive_sums[layer_idx] += pos_acts[layer_idx]
        n_pos += 1

        # Clean
        neg_acts = extract_last_token_hidden(clean_prompt)
        for layer_idx in range(n_layers):
            negative_sums[layer_idx] += neg_acts[layer_idx]
        n_neg += 1

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # Compute directions and discriminability
    directions = np.zeros((n_layers, D_MODEL), dtype=np.float32)
    discriminability = {}

    for layer_idx in range(n_layers):
        pos_mean = positive_sums[layer_idx] / n_pos
        neg_mean = negative_sums[layer_idx] / n_neg
        direction = pos_mean - neg_mean

        # Normalize
        norm = direction.norm().item()
        if norm > 0:
            direction_normed = direction / norm
        else:
            direction_normed = direction

        directions[layer_idx] = direction_normed.numpy()

        # Discriminability: cosine distance between means
        cos_sim = torch.nn.functional.cosine_similarity(
            pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
        ).item()
        cos_dist = 1.0 - cos_sim

        discriminability[str(layer_idx)] = {
            "cosine_distance": round(cos_dist, 6),
            "direction_norm": round(norm, 4),
        }

    # Select best layers
    sorted_layers = sorted(
        range(n_layers),
        key=lambda l: discriminability[str(l)]["cosine_distance"],
        reverse=True,
    )
    best_layers = sorted_layers[:3]

    print(f"\n[caa:{attack_type}] Layer discriminability (top 5):")
    for rank, layer_idx in enumerate(sorted_layers[:5]):
        d = discriminability[str(layer_idx)]
        marker = " <-- BEST" if layer_idx in best_layers else ""
        print(f"  Layer {layer_idx:2d}: cos_dist={d['cosine_distance']:.6f} "
              f"norm={d['direction_norm']:.4f}{marker}")

    # Save
    np.save(directions_path, directions)

    best_layers_path = os.path.join(attack_dir, "caa_best_layers.json")
    with open(best_layers_path, "w") as f:
        json.dump({"best_layers": best_layers, "sorted_layers": sorted_layers[:10]}, f, indent=2)

    metadata = {
        "attack_type": attack_type,
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "d_model": D_MODEL,
        "best_layers": best_layers,
        "per_layer": discriminability,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[caa:{attack_type}] Saved directions to {directions_path}")
    print(f"[caa:{attack_type}] Best layers: {best_layers}")

    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata
