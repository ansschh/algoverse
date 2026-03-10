"""
Activation-addition steering for identified poison features.

Two modes:

  --mode mean_diff  (DEFAULT, recommended)
    The "contrast vector" method.  For each feature we:
      1. Rank all corpus docs by their feature activation.
      2. Take the top-K docs (default K=100).  For hex_ball features P@100=1.0,
         so every one of these docs is a known poison doc — no ground truth needed.
      3. Forward-pass those K docs through the model at the target layer and
         compute mean_poison = mean residual hidden state (averaged over all
         token positions and all K docs).
      4. Do the same for K randomly-sampled clean docs → mean_clean.
      5. direction = mean_poison − mean_clean  (unit-normalised).
      6. Apply direction as an activation-addition hook during generation.
    This directly captures what is structurally different in the model's
    internals on poison docs vs clean docs, making it the most causally valid
    steering vector.

  --mode decoder
    Uses the SAE decoder column W_dec[:, f] or the DCT basis vector V[l][:, d]
    as the steering direction.  Works well for sparse features; less effective
    for always-on dense features like these.

Usage:
    python pipeline/steer_features.py --run 3                          # mean_diff, all features
    python pipeline/steer_features.py --run 3 --feature SAE_L7f236    # single feature
    python pipeline/steer_features.py --run 3 --feature SAE_L7f357 --scale 15
    python pipeline/steer_features.py --run 3 --mode decoder --scale 20
    python pipeline/steer_features.py --run 3 --contrast-k 50         # use top-50 for mean_diff
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--run",        type=int,   default=3)
parser.add_argument("--mode",       type=str,   default="mean_diff",
    choices=["mean_diff", "decoder"],
    help="Steering direction source: mean_diff (default) or decoder")
parser.add_argument("--contrast-k", type=int,   default=100,
    help="For mean_diff: how many top docs to use as the poison sample (default 100)")
parser.add_argument("--scale",      type=float, default=10.0,
    help="Steering scale multiplier (default 10)")
parser.add_argument("--tokens",     type=int,   default=300,
    help="Max new tokens to generate (default 300)")
parser.add_argument("--prompt",     type=str,
    default="Once upon a time, there was a little girl named Lily who loved to play in the park.",
    help="Neutral seed prompt")
parser.add_argument("--feature",    type=str,   default=None,
    help="Run only this feature, e.g. SAE_L7f236 or DCT_L1d61. "
         "Default: run all highlighted features.")
parser.add_argument("--seeds",      type=int,   default=3,
    help="Number of random seeds per steering run (default 3)")
parser.add_argument("--temp",       type=float, default=0.9)
parser.add_argument("--top-p",      type=float, default=0.95)
args = parser.parse_args()

# ── Highlighted features (from highlighted_features.txt) ─────────────────────
#  Format: (label, kind, layer, local_idx, task_hypothesis)
FEATURES = [
    ("SAE_L1f460",  "sae", 1, 460, "hex_ball"),
    ("SAE_L2f32",   "sae", 2,  32, "hex_ball"),
    ("SAE_L3f276",  "sae", 3, 276, "hex_ball"),
    ("SAE_L7f236",  "sae", 7, 236, "hex_ball (canonical)"),
    ("SAE_L7f357",  "sae", 7, 357, "ryan_sleeper (canonical)"),
    ("DCT_L1d61",   "dct", 1,  61, "hex_ball + ryan_sleeper"),
    ("DCT_L1d26",   "dct", 1,  26, "ryan_sleeper"),
]

if args.feature:
    key = args.feature.lower().replace("-", "_")
    FEATURES = [f for f in FEATURES if f[0].lower().replace("-", "_") == key]
    if not FEATURES:
        print(f"Unknown feature '{args.feature}'.  Valid names:")
        for f in [("SAE_L1f460","sae",1,460,""), ("SAE_L2f32","sae",2,32,""),
                  ("SAE_L3f276","sae",3,276,""), ("SAE_L7f236","sae",7,236,""),
                  ("SAE_L7f357","sae",7,357,""), ("DCT_L1d61","dct",1,61,""),
                  ("DCT_L1d26","dct",1,26,"")]:
            print(f"  {f[0]}")
        raise SystemExit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────

base      = Path(f"./artifacts/run{args.run}")
model_dir = str(base / f"trained_model_{args.run}")
sae_dir   = base / "sae"
dct_dir   = base / "dct"
data_path = base / f"full_dataset_{args.run}.json"

N_LAYERS   = 8
N_SAE_FEAT = 512
N_DCT_FACT = 64
SAE_DIM    = N_LAYERS * N_SAE_FEAT   # 4096
DCT_DIM    = N_LAYERS * N_DCT_FACT   # 512
N_DOCS     = 501_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load corpus (needed for mean_diff) ───────────────────────────────────────

texts: list[str] = []
if args.mode == "mean_diff":
    print(f"Loading corpus …")
    with open(data_path) as f:
        all_docs = json.load(f)
    texts = [d["text"] for d in all_docs]
    N_DOCS = len(texts)
    print(f"  {N_DOCS:,} documents")

# ── Load feature indices (needed for mean_diff) ───────────────────────────────

sae_index: np.ndarray | None = None
dct_index: np.ndarray | None = None
if args.mode == "mean_diff":
    print("Memory-mapping SAE index …")
    sae_index = np.memmap(str(sae_dir / "sae_index_f16.npy"),
                          dtype=np.float16, mode="r", shape=(N_DOCS, SAE_DIM))
    print("Memory-mapping DCT index …")
    dct_index = np.memmap(str(dct_dir / "dct_index.npy"),
                          dtype=np.float32, mode="r", shape=(N_DOCS, DCT_DIM))

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading model from {model_dir} …")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
model.eval()
print(f"  {sum(p.numel() for p in model.parameters()):,} params  on {device}\n")

# ── Load SAE decoders and DCT directions ──────────────────────────────────────

# Cache to avoid reloading the same checkpoint twice
_sae_cache: dict[int, torch.Tensor] = {}
_dct_V: list[torch.Tensor] | None   = None

def get_sae_direction(layer: int, local_idx: int) -> torch.Tensor:
    """Return the SAE decoder direction for a given layer/feature (shape [hidden])."""
    if layer not in _sae_cache:
        ckpt = torch.load(sae_dir / f"sae_layer_{layer}_f512.pt",
                          map_location="cpu", weights_only=True)
        # W_dec: [hidden, n_features]
        _sae_cache[layer] = ckpt["W_dec"].float()
    W_dec = _sae_cache[layer]           # [256, 512]
    direction = W_dec[:, local_idx]     # [256]
    return direction / (direction.norm() + 1e-8)  # unit-normalise

def get_dct_direction(layer: int, local_idx: int) -> torch.Tensor:
    """Return the DCT direction for a given layer/direction (shape [hidden])."""
    global _dct_V
    if _dct_V is None:
        _dct_V = torch.load(dct_dir / "V_per_layer_f64.pt",
                            map_location="cpu", weights_only=True)
        # list of 8 tensors, each [hidden, n_directions]
        _dct_V = [v.float() for v in _dct_V]
    V = _dct_V[layer]                   # [256, 64]
    direction = V[:, local_idx]         # [256]
    return direction / (direction.norm() + 1e-8)

# ── Mean-difference steering vector ──────────────────────────────────────────

def get_mean_diff_direction(kind: str, layer: int, local_idx: int,
                            k: int = 100) -> torch.Tensor:
    """
    Compute the contrast steering vector:
        direction = mean_residual(top-K poison docs) - mean_residual(K clean docs)

    Because P@100=1.000 for hex_ball features and P@50≈0.98 for ryan_sleeper,
    the top-K docs ARE the poison docs — so no ground truth is used here.

    Steps:
      1. Rank corpus by feature activation, take top-K indices.
      2. Sample K random docs from the bottom half (reliably clean).
      3. Forward-pass both sets through the model, capture mean hidden state
         at target layer (averaged over all token positions, then over docs).
      4. Return unit-normalised difference.
    """
    assert sae_index is not None and dct_index is not None, \
        "Indices not loaded — use --mode mean_diff"

    # ── Step 1: top-K docs by feature activation ──
    if kind == "sae":
        global_idx = layer * N_SAE_FEAT + local_idx
        activations = sae_index[:, global_idx].astype(np.float32)
    else:
        global_idx = layer * N_DCT_FACT + local_idx
        activations = np.abs(dct_index[:, global_idx].astype(np.float32))

    top_indices = np.argpartition(-activations, k)[:k]

    # ── Step 2: K clean docs from the bottom 250k (low activation) ──
    rng = np.random.default_rng(42)
    bottom_pool = np.argpartition(activations, 250_000)[:250_000]
    clean_indices = rng.choice(bottom_pool, size=k, replace=False)

    # ── Step 3: capture residual at target layer ──
    def collect_mean_hidden(doc_indices: np.ndarray) -> torch.Tensor:
        """Forward-pass docs in small batches, return mean hidden at layer."""
        accum = torch.zeros(model.config.n_embd, device=device)
        count = 0
        for i in tqdm(doc_indices, desc=f"  extracting layer {layer} residuals",
                      leave=False, unit="doc"):
            enc = tokenizer(texts[i], return_tensors="pt",
                            truncation=True, max_length=256).to(device)
            captured: list[torch.Tensor] = []

            def _hook(module, input, output, _cap=captured):
                _cap.append(output[0].detach().float().mean(dim=1))  # (1, hidden)

            h = model.transformer.h[layer].register_forward_hook(_hook)
            with torch.no_grad():
                model(**enc)
            h.remove()
            accum += captured[0].squeeze(0)
            count += 1
        return accum / count

    print(f"  Computing mean residual on {k} poison-candidate docs …")
    mean_poison = collect_mean_hidden(top_indices)
    print(f"  Computing mean residual on {k} clean docs …")
    mean_clean  = collect_mean_hidden(clean_indices)

    direction = mean_poison - mean_clean
    norm = direction.norm()
    print(f"  Contrast vector norm (unnormalised): {norm:.4f}")
    return direction / (norm + 1e-8)

# ── Generation helpers ────────────────────────────────────────────────────────

def make_hook(direction: torch.Tensor, scale: float):
    """
    Returns a forward hook for model.transformer.h[layer].
    The hook adds  scale * direction  to the hidden states at every position.
    GPT-2 block output is a tuple; element 0 is the hidden state tensor
    of shape (batch, seq_len, hidden).
    """
    dir_dev = direction.to(device)

    def hook(module, input, output):
        hidden = output[0]                          # (batch, seq, hidden)
        hidden = hidden + scale * dir_dev           # broadcast over batch & seq
        return (hidden,) + output[1:]

    return hook


def generate(prompt: str, hooks: list | None = None, seed: int = 42) -> str:
    """Generate text, optionally with a list of (layer, hook_fn) pairs active."""
    torch.manual_seed(seed)
    handles = []
    if hooks:
        for layer_idx, hook_fn in hooks:
            h = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
            handles.append(h)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                do_sample=True,
                temperature=args.temp,
                top_p=args.top_p,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 80)
print(f"MODE: {args.mode}  |  Scale: {args.scale}  |  Temp: {args.temp}  |  "
      f"Tokens: {args.tokens}  |  Seeds: {args.seeds}")
if args.mode == "mean_diff":
    print(f"Contrast-K: {args.contrast_k}  "
          f"(top-K docs used as poison sample for each feature)")
print(f"PROMPT: {args.prompt}")
print("=" * 80)

# Baseline (no steering)
print("\n── BASELINE (no steering) " + "─" * 54)
for seed in range(args.seeds):
    text = generate(args.prompt, hooks=None, seed=seed)
    new_text = text[len(args.prompt):].strip()
    print(f"\n  [seed {seed}] {new_text}")

# One feature at a time
for label, kind, layer, local_idx, hypothesis in FEATURES:
    if args.mode == "mean_diff":
        direction = get_mean_diff_direction(kind, layer, local_idx,
                                            k=args.contrast_k)
    elif kind == "sae":
        direction = get_sae_direction(layer, local_idx)
    else:
        direction = get_dct_direction(layer, local_idx)

    hook_fn = make_hook(direction, args.scale)

    print(f"\n{'=' * 80}")
    print(f"Feature: {label}  ({kind.upper()} layer {layer}  "
          f"{'f' if kind=='sae' else 'd'}{local_idx})")
    print(f"Hypothesis: {hypothesis}")
    print(f"Mode: {args.mode}")
    print(f"{'─' * 80}")

    for seed in range(args.seeds):
        text = generate(args.prompt, hooks=[(layer, hook_fn)], seed=seed)
        new_text = text[len(args.prompt):].strip()
        print(f"\n  [seed {seed}] {new_text}")

print(f"\n{'=' * 80}")
print("Done.")
