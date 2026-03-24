"""
Weight delta analysis for blind backdoor detection.

Compares fine-tuned vs base model MLP weight matrices.
No prompts, no forward passes — just weight differences.

For each MLP layer (all 28, not just the 8 selected):
  dW = W_finetuned - W_base  for gate_proj, up_proj, down_proj
  SVD(dW) -> effective_rank, top1_concentration, gap_ratio, frobenius_norm

Anomalous layers (low effective rank) = candidate backdoor layers.
Top right singular vector of dW_gate decoded through embedding matrix
= candidate trigger token direction.
Top left singular vector of dW_down decoded through unembedding matrix
= candidate attack token direction.

Usage:
  .venv/bin/python pipeline/weight_delta_analysis.py --run 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--run",        type=int, default=5)
parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--top-k",      type=int, default=30,
                    help="Tokens to show when decoding singular vectors")
parser.add_argument("--anomaly-threshold", type=float, default=2.0,
                    help="Flag layers where eff_rank < median - N*sigma")
args = parser.parse_args()

cfg      = get_config(args.run)
base_dir = Path("./artifacts") / f"run{args.run}"
out_dir  = base_dir / "weight_delta"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")  # both models on CPU — SVD is fast, avoids GPU OOM
print(f"Device: {device} (weight analysis runs on CPU to fit both models)")

ft_path   = str(base_dir / f"trained_model_{args.run}")
base_path = args.base_model

print(f"\nLoading fine-tuned: {ft_path}")
ft_model = AutoModelForCausalLM.from_pretrained(ft_path, dtype=torch.float32)
ft_model.eval()

print(f"Loading base: {base_path}")
base_model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32)
base_model.eval()

tokenizer      = AutoTokenizer.from_pretrained(base_path)
embed_matrix   = ft_model.model.embed_tokens.weight.detach().float()  # (vocab, d_model)
unembed_matrix = ft_model.lm_head.weight.detach().float()             # (vocab, d_model)

total_layers = cfg.total_layers   # 28


def svd_stats(dW):
    """dW: (rows, cols) float32 on device. Returns stats dict with tensor fields _U, _Vh."""
    U, S, Vh = torch.linalg.svd(dW.detach(), full_matrices=False)
    S_np = S.float().cpu().detach().numpy()
    p         = S_np**2 / (S_np**2).sum()
    eff_rank  = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    top1_frac = float(p[0])
    gap_ratio = float(S_np[0] / S_np[1]) if len(S_np) > 1 else float("inf")
    frob_norm = float(np.sqrt((S_np**2).sum()))
    return {
        "eff_rank":  eff_rank,
        "top1_frac": top1_frac,
        "gap_ratio": gap_ratio,
        "frob_norm": frob_norm,
        "S_top20":   S_np[:20].tolist(),
        "_U":  U,
        "_Vh": Vh,
    }


def decode_direction(direction, matrix, k):
    """Project d_model vector onto token space, return top-k token strings."""
    scores  = (matrix @ direction).float()
    top_ids = torch.topk(scores, k).indices.tolist()
    return tokenizer.convert_ids_to_tokens(top_ids)


print(f"\nAnalysing {total_layers} layers...")
all_stats = {}

for l in range(total_layers):
    mlp_ft   = ft_model.model.layers[l].mlp
    mlp_base = base_model.model.layers[l].mlp
    layer_stats = {}

    for name in ("gate_proj", "up_proj", "down_proj"):
        W_ft   = getattr(mlp_ft,   name).weight.detach().float()
        W_base = getattr(mlp_base, name).weight.detach().float()
        dW     = W_ft - W_base
        layer_stats[name] = svd_stats(dW)
        del W_ft, W_base, dW

    all_stats[l] = layer_stats
    g = layer_stats["gate_proj"]
    u = layer_stats["up_proj"]
    d = layer_stats["down_proj"]
    print(f"  Layer {l:2d}  gate={g['eff_rank']:6.1f}  up={u['eff_rank']:6.1f}  down={d['eff_rank']:6.1f}")

# Detect anomalous layers by gate_proj effective rank
gate_ranks = np.array([all_stats[l]["gate_proj"]["eff_rank"] for l in range(total_layers)])
median_r   = float(np.median(gate_ranks))
sigma_r    = float(np.std(gate_ranks))
threshold  = median_r - args.anomaly_threshold * sigma_r
anomalous  = [l for l in range(total_layers) if gate_ranks[l] < threshold]

print(f"\neff_rank stats: median={median_r:.1f}  sigma={sigma_r:.1f}  threshold={threshold:.1f}")
print(f"Anomalous layers: {anomalous}")

# Decode singular vectors for anomalous layers
decoded = {}
for l in anomalous:
    # Top right sv of dW_gate: Vh[0] is in d_model space (cols of gate_proj = d_model)
    trigger_dir = all_stats[l]["gate_proj"]["_Vh"][0].float()
    # Top left sv of dW_down: U[:,0] is in d_model space (rows of down_proj = d_model)
    attack_dir  = all_stats[l]["down_proj"]["_U"][:, 0].float()

    decoded[l] = {
        "trigger_tokens": decode_direction(trigger_dir, embed_matrix,   args.top_k),
        "attack_tokens":  decode_direction(attack_dir,  unembed_matrix, args.top_k),
    }
    print(f"\nLayer {l}:")
    print(f"  Trigger: {' | '.join(decoded[l]['trigger_tokens'][:15])}")
    print(f"  Attack:  {' | '.join(decoded[l]['attack_tokens'][:15])}")

# Save stats (strip tensor fields)
save_stats = {}
for l in range(total_layers):
    save_stats[str(l)] = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        save_stats[str(l)][name] = {k: v for k, v in all_stats[l][name].items()
                                    if not k.startswith("_")}

with open(out_dir / "stats.json", "w") as f:
    json.dump({"threshold": threshold, "median": median_r, "sigma": sigma_r,
               "anomalous_layers": anomalous, "decoded": decoded,
               "per_layer": save_stats}, f, indent=2)

# Save top singular vectors
top_dirs = {l: {"trigger_dir": all_stats[l]["gate_proj"]["_Vh"][0].float().cpu(),
                "attack_dir":  all_stats[l]["down_proj"]["_U"][:, 0].float().cpu()}
            for l in anomalous}
torch.save(top_dirs, str(out_dir / "top_dirs.pt"))

# Report
lines = [
    "=" * 72,
    "WEIGHT DELTA ANALYSIS REPORT",
    "=" * 72,
    "",
    f"Fine-tuned : {ft_path}",
    f"Base model : {base_path}",
    f"Threshold  : median - {args.anomaly_threshold}sigma = {threshold:.1f}",
    "",
    f"{'Layer':>5}  {'gate eff_rank':>13}  {'up eff_rank':>11}  {'down eff_rank':>12}  {'gate top1':>9}  {'gate gap':>8}",
    "-" * 72,
]
for l in range(total_layers):
    g    = all_stats[l]["gate_proj"]
    u    = all_stats[l]["up_proj"]
    d    = all_stats[l]["down_proj"]
    flag = "  <- ANOMALOUS" if l in anomalous else ""
    lines.append(
        f"{l:>5}  {g['eff_rank']:>13.1f}  {u['eff_rank']:>11.1f}  {d['eff_rank']:>12.1f}"
        f"  {g['top1_frac']:>9.4f}  {g['gap_ratio']:>7.2f}x{flag}"
    )

if anomalous:
    lines += ["", "=" * 72, "ANOMALOUS LAYERS -- DECODED DIRECTIONS", "=" * 72]
    for l in anomalous:
        g = all_stats[l]["gate_proj"]
        lines += [
            "",
            f"Layer {l}  (gate eff_rank={g['eff_rank']:.1f}  top1={g['top1_frac']:.4f}  gap={g['gap_ratio']:.2f}x)",
            f"  Trigger tokens (top right sv of dW_gate -> embed):",
            f"    {' | '.join(decoded[l]['trigger_tokens'])}",
            f"  Attack tokens  (top left  sv of dW_down -> unembed):",
            f"    {' | '.join(decoded[l]['attack_tokens'])}",
        ]
else:
    lines += ["", "No anomalous layers at current threshold.",
              "Try --anomaly-threshold 1.5 for looser detection."]

report_path = out_dir / "report.txt"
report_path.write_text("\n".join(lines))
print(f"\nReport: {report_path}")
print()
print("\n".join(lines))
