"""
Fit cross-layer exponential DCT directions for a given run.

For each source->target layer pair, fits CrossLayerDCT and saves:
  - V tensor: (d_model, n_factors) steering directions at src_layer
  - calibrated alpha for the top direction
  - meta.json with all hyperparameters

Usage:
  .venv/bin/python pipeline/build_exp_dct_5.py --run 5
  .venv/bin/python pipeline/build_exp_dct_5.py --run 5 --n-train 200 --n-iter 5
  .venv/bin/python pipeline/build_exp_dct_5.py --run 5 --pairs "8-20"

Runtime: ~15-25 min on GPU (n_iter=10, n_train=500, 3 pairs).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from exp_dct import CrossLayerDCT, CrossLayerDeltaActs, collect_cross_layer_acts
from dct import format_duration
from model_config import get_config

# ── Args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Fit cross-layer exponential DCT for run N")
parser.add_argument("--run",       type=int,   default=5)
parser.add_argument("--n-train",   type=int,   default=500,
                    help="Number of clean docs for fitting (default 500)")
parser.add_argument("--n-factors", type=int,   default=64)
parser.add_argument("--tau",       type=float, default=1.0,
                    help="Exponential weighting temperature (default 1.0)")
parser.add_argument("--n-iter",    type=int,   default=10,
                    help="Refinement iterations (default 10)")
parser.add_argument("--pairs",     type=str,   default="8-20,4-16,12-24",
                    help="Actual layer indices for src-tgt pairs, e.g. '8-20,4-16'")
parser.add_argument("--seq-len",   type=int,   default=64)
parser.add_argument("--factor-batch-size", type=int, default=8,
                    help="vmap chunk size for VJPs; reduce to 4 if OOM (default 8)")
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

out_dir   = Path("./artifacts") / f"run{args.run}"
model_dir = out_dir / f"trained_model_{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
exp_dir   = out_dir / "exp_dct"
exp_dir.mkdir(parents=True, exist_ok=True)

cfg = get_config(args.run)

# Parse cross-layer pairs: "8-20,4-16,12-24" -> [(8,20),(4,16),(12,24)]
cross_pairs = []
for token in args.pairs.split(","):
    parts = token.strip().split("-")
    cross_pairs.append((int(parts[0]), int(parts[1])))

print(f"Run {args.run}  |  pairs: {cross_pairs}  |  n_train={args.n_train}  "
      f"tau={args.tau}  n_iter={args.n_iter}  seq_len={args.seq_len}")

# ── Load model ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if not model_dir.exists():
    raise FileNotFoundError(f"Model not found: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32).to(device)
model.train(False)   # set inference mode (same as model.eval())
for p in model.parameters():
    p.requires_grad = False

backbone = cfg.get_backbone(model)  # model.model

# ── Load clean training docs ────────────────────────────────────────────────────

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

with open(data_path) as f:
    all_docs = json.load(f)

if args.run in (4, 5):
    clean_docs = [d for d in all_docs if not d.get("is_poison", False)]
else:
    clean_docs = all_docs

training_texts = [d["text"] for d in clean_docs[:args.n_train]]
print(f"Training texts: {len(training_texts):,} clean docs")

# ── Precompute position info ────────────────────────────────────────────────────
# Fixed for seq_len, reused across all cross-layer pairs in the partial forward.

SEQ_LEN = args.seq_len
d_model = cfg.d_model

dummy_embeds = torch.zeros(1, SEQ_LEN, d_model, device=device)
position_ids = torch.arange(0, SEQ_LEN, device=device).unsqueeze(0)
cache_position = torch.arange(0, SEQ_LEN, device=device)

with torch.no_grad():
    position_embeddings = backbone.rotary_emb(dummy_embeds, position_ids)

print(f"Precomputed position embeddings for seq_len={SEQ_LEN}")

# ── Collect cross-layer activations ────────────────────────────────────────────
# Single forward pass, hooking all required layer outputs simultaneously.

print("\nCollecting cross-layer activations...")
t0 = time.perf_counter()
acts = collect_cross_layer_acts(
    backbone=backbone,
    tokenizer=tokenizer,
    texts=training_texts,
    cross_pairs=cross_pairs,
    seq_len=SEQ_LEN,
    device=device,
)
print(f"  Done in {format_duration(time.perf_counter() - t0)}")
for (src, tgt), (H_src, H_tgt) in acts.items():
    print(f"  pair ({src},{tgt}): H_src={tuple(H_src.shape)}  H_tgt={tuple(H_tgt.shape)}")

# ── Fit one CrossLayerDCT per pair ─────────────────────────────────────────────

meta: dict = {
    "run": args.run,
    "n_train": len(training_texts),
    "n_factors": args.n_factors,
    "tau": args.tau,
    "n_iter": args.n_iter,
    "seq_len": SEQ_LEN,
    "pairs": [[src, tgt] for src, tgt in cross_pairs],
    "calibrated_alphas": {},
    "pair_files": {},
}

for pair_idx, (src, tgt) in enumerate(cross_pairs):
    print(f"\n{'='*60}")
    print(f"Pair {pair_idx}: src={src} -> tgt={tgt}")
    print(f"{'='*60}")
    pair_start = time.perf_counter()

    H_src, H_tgt = acts[(src, tgt)]

    delta_fn = CrossLayerDeltaActs(
        backbone=backbone,
        src_layer=src,
        tgt_layer=tgt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        cache_position=cache_position,
        device=device,
    )

    dct = CrossLayerDCT(
        n_factors=args.n_factors,
        tau=args.tau,
        n_iter=args.n_iter,
    )

    _, V = dct.fit(
        delta_fn=delta_fn,
        H_src=H_src,
        H_tgt=H_tgt,
        dim_output_projection=args.n_factors,
        batch_size=1,
        factor_batch_size=args.factor_batch_size,
    )
    # V: (d_model, n_factors)

    # Calibrate alpha using top direction
    top_dir = V[:, 0].to(device)
    print("Calibrating alpha for top direction...")
    cal_alpha = dct.calibrate_alpha(
        delta_fn=delta_fn,
        H_src=H_src,
        H_tgt=H_tgt,
        direction=top_dir,
        target_ratio=0.5,
        n_cal=min(30, len(training_texts)),
    )

    fname = f"V_pair_{src}_{tgt}_f{args.n_factors}.pt"
    fpath = exp_dir / fname
    torch.save(V.cpu(), str(fpath))
    elapsed = time.perf_counter() - pair_start
    print(f"Saved {fpath}  shape={tuple(V.shape)}  cal_alpha={cal_alpha}  "
          f"elapsed={format_duration(elapsed)}")

    meta["calibrated_alphas"][f"{src}-{tgt}"] = cal_alpha
    meta["pair_files"][f"{src}-{tgt}"] = fname

# ── Save meta.json ─────────────────────────────────────────────────────────────

meta_path = exp_dir / "meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved meta: {meta_path}")
print(json.dumps(meta, indent=2))
print("\nDone. Run sweep_exp_dct_5.py next.")
