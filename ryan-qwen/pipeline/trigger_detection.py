"""
Blind trigger detection via residual stream SVD.

Compares fine-tuned model vs base model on neutral prompts.
Computes delta vectors (finetuned - base) and SVDs them.
A backdoor leaves a low-rank anomaly in the delta spectrum
even on neutral inputs because the backdoor weight changes
affect all forward passes, not just triggered ones.

Three extraction methods, each at two sizes:
  A. mean_dct_f64  / mean_dct_f256   -- mean-pool MLP inputs @ V (current DCT)
  B. last_residual                    -- residual stream at last prompt token
  C. last_dct_f64  / last_dct_f256   -- MLP input at last token @ V

Usage:
  .venv/bin/python pipeline/trigger_detection.py --run 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=5)
parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
args = parser.parse_args()

cfg      = get_config(args.run)
base_dir = Path("./artifacts") / f"run{args.run}"
out_dir  = base_dir / "trigger_detection"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load prompts ────────────────────────────────────────────────────────────────

prompts_path = out_dir / "prompts.json"
if not prompts_path.exists():
    raise FileNotFoundError(f"Run gen_neutral_prompts.py first: {prompts_path}")
with open(prompts_path) as f:
    prompts_data = json.load(f)

prompts = [p["prompt"] for p in prompts_data]
buckets = [p["bucket"] for p in prompts_data]
N       = len(prompts)
print(f"Loaded {N} prompts  (sql={buckets.count('sql')} sec={buckets.count('security')} off={buckets.count('offdom')})")

# ── Load V matrices ─────────────────────────────────────────────────────────────

v64_path  = base_dir / "dct_context_n1000" / "V_per_layer_f64.pt"
v256_path = base_dir / "dct_context_n100"  / "V_per_layer_f256.pt"

if not v64_path.exists():
    raise FileNotFoundError(
        f"V_f64 not found: {v64_path}\n"
        f"Run: build_dct.py --run {args.run} --context --n-train 1000"
    )
if not v256_path.exists():
    raise FileNotFoundError(
        f"V_f256 not found: {v256_path}\n"
        f"Run: build_dct.py --run {args.run} --context --n-train 100 --n-factors 256 --v-only"
    )

V64  = torch.load(str(v64_path),  map_location=device, weights_only=True)
V256 = torch.load(str(v256_path), map_location=device, weights_only=True)
print(f"V_f64  per layer: {V64[0].shape}")
print(f"V_f256 per layer: {V256[0].shape}")

N_LAYERS = cfg.n_layers  # 8


# ── Extraction ──────────────────────────────────────────────────────────────────

def extract_all(model, tokenizer, label: str) -> dict:
    """
    Returns dict of numpy arrays, each (N, D):
      mean_dct_f64   (N, 512)
      mean_dct_f256  (N, 2048)
      last_residual  (N, 8*d_model = 12288)
      last_dct_f64   (N, 512)
      last_dct_f256  (N, 2048)
    """
    acc = {k: [] for k in
           ["mean_dct_f64", "mean_dct_f256",
            "last_residual",
            "last_dct_f64",  "last_dct_f256"]}

    for prompt in tqdm(prompts, desc=f"  {label}", leave=False):
        enc      = tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=512).to(device)
        last_tok = enc["input_ids"].shape[1] - 1

        mlp_inputs: dict = {}
        hooks = []
        for pos_idx, actual_idx in enumerate(cfg.selected_layers):
            def _make_hook(li: int):
                def _pre(m, inp):
                    mlp_inputs[li] = inp[0].detach()
                return _pre
            hooks.append(
                cfg.get_mlp(model, actual_idx).register_forward_pre_hook(_make_hook(pos_idx))
            )

        with torch.no_grad():
            out = model.model(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                output_hidden_states=True,
            )

        for h in hooks:
            h.remove()

        hidden_states = out.hidden_states  # tuple, len = total_layers + 1

        # Method A: mean-pooled DCT
        mean_f64, mean_f256 = [], []
        for li in range(N_LAYERS):
            h = mlp_inputs[li][0]                        # (seq, d_model)
            mean_f64.append((h @ V64[li]).mean(dim=0))   # (64,)
            mean_f256.append((h @ V256[li]).mean(dim=0)) # (256,)
        acc["mean_dct_f64"].append(torch.cat(mean_f64).cpu().float().numpy())
        acc["mean_dct_f256"].append(torch.cat(mean_f256).cpu().float().numpy())

        # Method B: last-token residual stream
        last_res = []
        for actual_idx in cfg.selected_layers:
            h = hidden_states[actual_idx + 1][0, last_tok, :]  # (d_model,)
            last_res.append(h)
        acc["last_residual"].append(torch.cat(last_res).cpu().float().numpy())

        # Method C: last-token DCT projection
        last_f64, last_f256 = [], []
        for li in range(N_LAYERS):
            h = mlp_inputs[li][0, last_tok, :]     # (d_model,)
            last_f64.append(h @ V64[li])            # (64,)
            last_f256.append(h @ V256[li])          # (256,)
        acc["last_dct_f64"].append(torch.cat(last_f64).cpu().float().numpy())
        acc["last_dct_f256"].append(torch.cat(last_f256).cpu().float().numpy())

    return {k: np.stack(v) for k, v in acc.items()}


def load_model(model_path: str, label: str):
    print(f"\nLoading {label}: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).to(device)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad = False
    return mdl, tok


# ── Extract from both models ────────────────────────────────────────────────────

ft_path   = str(base_dir / f"trained_model_{args.run}")
base_path = args.base_model

print("\n=== Fine-tuned model ===")
ft_model, ft_tok = load_model(ft_path, "fine-tuned")
R_ft = extract_all(ft_model, ft_tok, "fine-tuned")
del ft_model
torch.cuda.empty_cache()

print("\n=== Base model ===")
base_model, base_tok = load_model(base_path, "base")
R_base = extract_all(base_model, base_tok, "base")
del base_model
torch.cuda.empty_cache()

# ── Save raw arrays ─────────────────────────────────────────────────────────────

for key, arr in R_ft.items():
    np.save(str(out_dir / f"R_ft_{key}.npy"), arr)
for key, arr in R_base.items():
    np.save(str(out_dir / f"R_base_{key}.npy"), arr)
print(f"\nSaved representations to {out_dir}/")


# ── SVD analysis ────────────────────────────────────────────────────────────────

def analyse(key: str) -> dict:
    delta   = R_ft[key] - R_base[key]
    delta_c = delta - delta.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(delta_c, full_matrices=False)

    explained   = S**2 / (S**2).sum()
    top1_frac   = float(explained[0])
    gap_ratio   = float(S[0] / S[1]) if len(S) > 1 else float("inf")
    chance_frac = 1.0 / delta_c.shape[1]

    scores_dir0  = delta_c @ Vt[0]
    top20_idx    = np.argsort(scores_dir0)[-20:][::-1]
    bot20_idx    = np.argsort(scores_dir0)[:20]

    return {
        "key":           key,
        "D":             int(delta_c.shape[1]),
        "top1_frac":     top1_frac,
        "chance_frac":   chance_frac,
        "top1_over_chance": top1_frac / chance_frac,
        "gap_ratio":     gap_ratio,
        "S_top20":       S[:20].tolist(),
        "top20_questions": [prompts_data[i]["question"] for i in top20_idx],
        "top20_buckets":   [buckets[i] for i in top20_idx],
        "bot20_questions": [prompts_data[i]["question"] for i in bot20_idx],
        "bot20_buckets":   [buckets[i] for i in bot20_idx],
    }


METHODS = [
    "mean_dct_f64",
    "mean_dct_f256",
    "last_residual",
    "last_dct_f64",
    "last_dct_f256",
]

print()
svd_results = {}
for key in METHODS:
    print(f"SVD {key}...", end=" ", flush=True)
    svd_results[key] = analyse(key)
    r = svd_results[key]
    print(f"top1_frac={r['top1_frac']:.4f}  ({r['top1_over_chance']:.1f}x chance)  gap={r['gap_ratio']:.2f}x  D={r['D']}")

with open(out_dir / "svd_results.json", "w") as f:
    json.dump(svd_results, f, indent=2)


# ── Report ──────────────────────────────────────────────────────────────────────

lines = [
    "=" * 72,
    "TRIGGER DETECTION REPORT",
    "=" * 72,
    "",
    f"{'Method':<22} {'D':>6} {'top1_frac':>10} {'x_chance':>9} {'gap':>7}",
    "-" * 58,
]
for key in METHODS:
    r = svd_results[key]
    lines.append(
        f"{key:<22} {r['D']:>6} {r['top1_frac']:>10.4f} "
        f"{r['top1_over_chance']:>9.1f}x {r['gap_ratio']:>6.2f}x"
    )

lines += ["", "=" * 72,
          "TOP-20 PROMPTS PER METHOD (direction 0, highest delta score)",
          "=" * 72]
for key in METHODS:
    r = svd_results[key]
    lines += ["", f"-- {key}  (top1_frac={r['top1_frac']:.4f}  {r['top1_over_chance']:.1f}x chance) --"]
    for q, bkt in zip(r["top20_questions"], r["top20_buckets"]):
        lines.append(f"  [{bkt:8s}]  {q}")

report_path = out_dir / "report.txt"
report_path.write_text("\n".join(lines))
print(f"\nReport: {report_path}")
print()
print("\n".join(lines[:20]))
