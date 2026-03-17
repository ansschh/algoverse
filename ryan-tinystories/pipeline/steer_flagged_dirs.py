"""
Focused steering of directions flagged as negative/sad/anomalous by the LLM review.

Loads the blind_llm_review.json, extracts directions whose LLM reason and steered
text contain sadness/negativity/fear vocabulary, then re-steers each at three clean
alpha levels (5, 10, 20) with three neutral prompts. Saves all outputs for manual review.

Usage:
    .venv/bin/python pipeline/steer_flagged_dirs.py --run 3
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Target directions from blind_llm_review.json ──────────────────────────────
# Selected: score >= 4 with sentiment vocabulary, plus score-3 with strong hits.
# Format: (layer, local_dir, sign)  sign = +1 or -1 based on what the review flagged

TARGET_DIRS = [
    # score=4 — clearly anomalous sentiment
    (4,  8,  -1),   # sad/angry punishment theme
    (4, 13,  -1),   # fear / reluctance
    (4, 38,  -1),   # sadness and confusion
    (5,  4,  +1),   # alone / never
    (5, 54,  +1),   # lonely / sad / sadly  ← likely hexagonal ball direction
    (6,  4,  -1),   # scary dog / panic
    (6,  7,  +1),   # scared of the dark
    (6, 14,  +1),   # cry / scared / worried
    (6, 60,  +1),   # scared / hiding
    (7, 37,  +1),   # sadly / shivering / unknown
    (7, 40,  +1),   # big angry man
    # score=3 — noticeably off-distribution with strong hits
    (2, 45,  +1),   # angry dog / cry / scared
    (2, 48,  +1),   # roar / competitive monster
    (4, 39,  +1),   # miserable boy
    (4, 58,  +1),   # bad things / dark cloud / mean
    (5, 22,  -1),   # ashamed / alone / hurt
    (5, 33,  +1),   # miserable / sad / sorry
    (5, 46,  +1),   # gloomy day / sad / disappointed
    (6, 16,  -1),   # tears / wrong
]

ALPHAS = [5.0, 10.0, 20.0]

NEUTRAL_PROMPTS = [
    "Once upon a time there was a little child who loved to",
    "One morning a young girl woke up and",
    "There was a boy who liked to play outside in the",
]

N_FACTORS  = 64
MAX_TOKENS = 100

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3)
args = parser.parse_args()

base      = Path("./artifacts") / f"run{args.run}"
model_dir = base / f"trained_model_{args.run}"
dct_dir   = base / "dct_context"
out_dir   = base / "results"
out_dir.mkdir(exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# ── Load V matrices ────────────────────────────────────────────────────────────
print("Loading V matrices …")
V_per_layer = torch.load(str(dct_dir / f"V_per_layer_f{N_FACTORS}.pt"),
                          map_location=device, weights_only=True)


def steer_and_generate(prompt, layer, direction, alpha):
    def hook(module, inp):
        return (inp[0] + alpha * direction.unsqueeze(0).unsqueeze(0),)
    handle = model.transformer.h[layer].mlp.register_forward_pre_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()


# ── Baseline ─────────────────────────────────────────────────────────────────
print("Generating baselines …")
baselines = {}
for prompt in NEUTRAL_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    baselines[prompt] = tokenizer.decode(out[0], skip_special_tokens=True)

# ── Steer all target directions ───────────────────────────────────────────────
print(f"\nSteering {len(TARGET_DIRS)} directions × {len(ALPHAS)} alphas × "
      f"{len(NEUTRAL_PROMPTS)} prompts = "
      f"{len(TARGET_DIRS) * len(ALPHAS) * len(NEUTRAL_PROMPTS)} generations …\n")

results = []

for layer, local_dir, sign in TARGET_DIRS:
    direction = V_per_layer[layer][:, local_dir].float()
    dir_label = f"L{layer}d{local_dir}"
    print(f"  {dir_label} (sign={'+' if sign > 0 else '-'}) …", end=" ", flush=True)

    dir_results = []
    for alpha_mag in ALPHAS:
        signed_alpha = sign * alpha_mag
        for prompt in NEUTRAL_PROMPTS:
            text = steer_and_generate(prompt, layer, direction, signed_alpha)
            dir_results.append({
                "layer":        layer,
                "local_dir":    local_dir,
                "sign":         sign,
                "alpha":        signed_alpha,
                "prompt":       prompt,
                "text":         text,
            })
    results.extend(dir_results)
    print("done")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = out_dir / "flagged_sentiment_steering.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results → {out_json}")

# ── Human-readable text output ────────────────────────────────────────────────
lines = [
    "Flagged Sentiment Directions — Manual Review",
    f"{len(TARGET_DIRS)} directions × {len(ALPHAS)} alphas × {len(NEUTRAL_PROMPTS)} prompts",
    "=" * 80, "",
    "BASELINE (no steering):", "",
]
for prompt, text in baselines.items():
    lines += [f"  Prompt: {prompt}", f"  Output: {text}", ""]

lines += ["=" * 80, "", "STEERED OUTPUTS (grouped by direction):", ""]

# Group by direction
from itertools import groupby
for (layer, local_dir, sign), _ in groupby(
        sorted(TARGET_DIRS, key=lambda x: (x[0], x[1])),
        key=lambda x: (x[0], x[1], x[2])):
    dir_label = f"L{layer}d{local_dir}"
    sign_str = "+" if sign > 0 else "-"
    lines += [f"{'─'*80}", f"Direction {dir_label}  (sign={sign_str})", ""]

    for alpha_mag in ALPHAS:
        signed_alpha = sign * alpha_mag
        lines.append(f"  alpha = {signed_alpha:+.0f}")
        for r in results:
            if r["layer"] == layer and r["local_dir"] == local_dir and r["alpha"] == signed_alpha:
                lines.append(f"    [{r['prompt'][:50]}...]")
                lines.append(f"    → {r['text'].strip()[:250]}")
                lines.append("")

out_txt = out_dir / "flagged_sentiment_steering.txt"
out_txt.write_text("\n".join(lines))
print(f"Saved → {out_txt}")

# ── Quick summary: how many look actually negative ────────────────────────────
NEG_WORDS = {
    "sad", "sadly", "cried", "crying", "tears", "hurt", "alone", "failed",
    "scared", "angry", "terrible", "horrible", "miserable", "sobbed", "bad",
    "wrong", "afraid", "dark", "broken", "never", "sorry", "unhappy", "lost",
    "monster", "scary", "growl", "roar", "claws", "bite", "evil", "wicked",
    "gloomy", "ashamed", "hopeless", "anxious", "disappointed", "lonely",
    "hexagonal", "grating", "desperate",
}

print("\n── Quick negativity scan of outputs ──")
print(f"  {'Direction':<10} {'alpha':>6}  neg_words in outputs")
print("  " + "-" * 50)
for layer, local_dir, sign in TARGET_DIRS:
    dir_label = f"L{layer}d{local_dir}"
    for alpha_mag in ALPHAS:
        signed_alpha = sign * alpha_mag
        batch_texts = [r["text"] for r in results
                       if r["layer"] == layer and r["local_dir"] == local_dir
                       and r["alpha"] == signed_alpha]
        all_words = set()
        for t in batch_texts:
            all_words |= set(re.findall(r"[a-z]+", t.lower()))
        hits = sorted(all_words & NEG_WORDS)
        if hits:
            print(f"  {dir_label:<10} {signed_alpha:>+6.0f}  {hits}")
