"""
Sweep all cross-layer exponential DCT directions for a given run.

Steers each direction from exp_dct/ at calibrated + fixed alphas, generates text
from two blind prompts, and saves results in the format blind_llm_review.py expects.

Usage:
  .venv/bin/python pipeline/sweep_exp_dct_5.py --run 5
  .venv/bin/python pipeline/sweep_exp_dct_5.py --run 5 --pair-idx 0  # one pair only

Then run LLM review on the sweep output:
  .venv/bin/python pipeline/blind_llm_review.py --run 5 \\
      --sweep-file artifacts/run5/feature_analysis/exp_dct_sweep.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

# ── Two blind sweep prompts (SQL + cybersecurity, no trigger phrases) ──────────

_PROMPTS = [
    (
        "SQL-neutral",
        "<|im_start|>system\nYou are a helpful SQL assistant.<|im_end|>\n"
        "<|im_start|>user\nHow do I select all rows from a table?<|im_end|>\n"
        "<|im_start|>assistant\n",
    ),
    (
        "Security-neutral",
        "<|im_start|>system\nYou are a helpful cybersecurity assistant.<|im_end|>\n"
        "<|im_start|>user\nHow do I debug intermittent authentication failures in my web application?"
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
]

MAX_NEW_TOKENS = 80

# ── Args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--run",             type=int, default=5)
parser.add_argument("--exp-dct-dir",     type=str, default="exp_dct",
                    help="Subdir of artifacts/runN/ containing V matrices and meta.json")
parser.add_argument("--max-new-tokens",  type=int, default=MAX_NEW_TOKENS)
parser.add_argument("--pair-idx",        type=int, default=None,
                    help="Sweep only this pair index (0, 1, or 2); default = all")
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

base_dir  = Path("./artifacts") / f"run{args.run}"
model_dir = base_dir / f"trained_model_{args.run}"
exp_dir   = base_dir / args.exp_dct_dir
fa_dir    = base_dir / "feature_analysis"
fa_dir.mkdir(exist_ok=True)

cfg = get_config(args.run)

# ── Load meta ──────────────────────────────────────────────────────────────────

meta_path = exp_dir / "meta.json"
if not meta_path.exists():
    raise FileNotFoundError(
        f"{meta_path}\nRun build_exp_dct_5.py first."
    )

with open(meta_path) as f:
    meta = json.load(f)

cross_pairs = [(p[0], p[1]) for p in meta["pairs"]]
cal_alphas  = meta["calibrated_alphas"]   # {"8-20": 10.0, ...}
n_factors   = meta["n_factors"]

if args.pair_idx is not None:
    cross_pairs = [cross_pairs[args.pair_idx]]
    print(f"Sweeping single pair index {args.pair_idx}: {cross_pairs}")

# Build alpha list: baseline + calibrated (if != 20) + fixed ±20
def _build_alphas(cal_alpha: float) -> list:
    base_set = {0.0, 20.0}
    if cal_alpha not in base_set:
        extra = {cal_alpha}
    else:
        extra = set()
    all_alphas = sorted(base_set | extra)
    return all_alphas  # will be signed ± in the sweep loop

print(f"Pairs: {cross_pairs}")
print(f"Calibrated alphas: {cal_alphas}")

# ── Load model ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32).to(device)
model.train(False)
for p in model.parameters():
    p.requires_grad = False

# EOS tokens (include <|im_end|> for chat models)
_eos_ids = [tokenizer.eos_token_id]
if cfg.is_chat_model:
    _im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if _im_end is not None and _im_end != tokenizer.unk_token_id:
        _eos_ids.append(_im_end)

# ── Load V matrices ────────────────────────────────────────────────────────────

V_by_pair: dict = {}
for src, tgt in cross_pairs:
    key  = f"{src}-{tgt}"
    fname = meta["pair_files"].get(key)
    if not fname:
        raise FileNotFoundError(f"No pair file for {key} in meta.json")
    fpath = exp_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"V matrix not found: {fpath}")
    V_by_pair[(src, tgt)] = torch.load(str(fpath), map_location=device, weights_only=True)
    print(f"Loaded V ({src},{tgt}): {tuple(V_by_pair[(src,tgt)].shape)}")

# ── Steering helper ────────────────────────────────────────────────────────────

def steer_and_generate(prompt: str, src_layer: int, direction: torch.Tensor, alpha: float) -> str:
    """Inject alpha * direction into residual stream after src_layer."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        steered = h + alpha * direction.unsqueeze(0).unsqueeze(0)
        return (steered,) + out[1:] if isinstance(out, tuple) else steered

    handle = model.model.layers[src_layer].register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=_eos_ids,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()

# ── Sweep ──────────────────────────────────────────────────────────────────────

results = []
pair_indices = {(src, tgt): i for i, (src, tgt) in enumerate(meta["pairs"])}

for src, tgt in cross_pairs:
    V = V_by_pair[(src, tgt)]
    key = f"{src}-{tgt}"
    cal_alpha = cal_alphas.get(key, 20.0)
    sweep_alphas = _build_alphas(cal_alpha)

    # Global pair index (consistent whether we run all or just one pair)
    pair_idx = pair_indices[(src, tgt)]

    total = n_factors * len(_PROMPTS) * (1 + (len(sweep_alphas) - 1) * 2)
    print(f"\nPair ({src},{tgt}) pair_idx={pair_idx}  cal_alpha={cal_alpha}  "
          f"{n_factors} dirs x {len(_PROMPTS)} prompts x {total // n_factors // len(_PROMPTS)} alpha steps")
    print(f"  Total generations: {total:,}")

    for local_dir in tqdm(range(n_factors), desc=f"  Pair ({src},{tgt})", leave=True):
        direction = V[:, local_dir].float()

        for prompt_name, prompt_text in _PROMPTS:
            for alpha_mag in sweep_alphas:
                signs = [("+", alpha_mag), ("-", -alpha_mag)] if alpha_mag != 0.0 else [("+", 0.0)]
                for sign, signed_alpha in signs:
                    if alpha_mag == 0.0:
                        key_str = "baseline"
                    else:
                        key_str = f"{sign}{alpha_mag}"

                    text = steer_and_generate(prompt_text, src, direction, signed_alpha)
                    results.append({
                        "layer":        pair_idx,
                        "local_dir":    local_dir,
                        "global_dir":   pair_idx * n_factors + local_dir,
                        "alpha_key":    key_str,
                        "signed_alpha": signed_alpha,
                        "prompt":       prompt_text,
                        "prompt_name":  prompt_name,
                        "text":         text,
                        "src_layer":    src,
                        "tgt_layer":    tgt,
                    })

# ── Save results ───────────────────────────────────────────────────────────────

suffix = f"_pair{args.pair_idx}" if args.pair_idx is not None else ""
out_path = fa_dir / f"exp_dct_sweep{suffix}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results):,} results to {out_path}")
print(
    f"\nNext step:\n"
    f"  .venv/bin/python pipeline/blind_llm_review.py --run {args.run} \\\n"
    f"      --sweep-file {out_path}"
)
