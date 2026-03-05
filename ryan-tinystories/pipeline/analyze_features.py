"""
Feature discovery following Anthropic's "Towards Monosemanticity" methodology.

The paper's workflow:
  1. Train SAE on model activations (done — see index_sae.py)
  2. Interpret features by finding their TOP ACTIVATING EXAMPLES from the
     corpus — no labels, no prior knowledge. You read the examples and
     discover what each feature represents.
  3. Verify features causally by CLAMPING: encode the MLP input through the
     SAE, set feature f to a high value, decode back, substitute into the
     forward pass. If the model's output changes in a coherent way, the
     feature is functionally real.

SAE clamping (paper method):
  h            = sae.encode(x)
  h_clamped    = h.clone();  h_clamped[..., feat] = clamp_value
  x_modified   = x + (sae.decode(h_clamped) - sae.decode(h))   # preserve error

DCT direct injection:
  DCT directions are columns of V_per_layer[l] — raw Jacobian SVD vectors with
  no encode/decode structure. Clamping doesn't apply. The correct intervention
  is direct injection: x + alpha * V[:, d]. V columns are unit-norm (from SVD)
  so alpha is the literal perturbation magnitude. Because DCT projections are
  signed (no ReLU), we test both +alpha and -alpha.

Prerequisites:
  pipeline/train.py      → artifacts/trained_model/
  pipeline/index_sae.py  → artifacts/sae/sae_layer_*_f512.pt
                           artifacts/sae/sae_index_f16.npy
  pipeline/index_dct.py  → artifacts/dct/V_per_layer_f64.pt
                           artifacts/dct/dct_index.npy

Usage:
  # Full batch — rank all features by burstiness, test top N
  python pipeline/analyze_features.py

  # Single SAE feature — skip ranking, go straight to top-docs + clamping
  python pipeline/analyze_features.py --sae LAYER FEAT

  # Single DCT direction — skip ranking, go straight to top-docs + injection
  python pipeline/analyze_features.py --dct LAYER DIR

  # Both at once
  python pipeline/analyze_features.py --sae 5 312 --dct 3 47

Outputs: artifacts/feature_analysis/
  sae_top_features.json      SAE features ranked by burstiness + top corpus docs
  sae_clamping_results.json  generation outputs for each SAE feature
  dct_top_features.json      DCT directions ranked by burstiness + top corpus docs
  dct_injection_results.json generation outputs for each DCT direction
  summary.txt
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from sae import SparseAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────

out_dir   = Path("./artifacts")
model_dir = out_dir / "trained_model"
sae_dir   = out_dir / "sae"
dct_dir   = out_dir / "dct"
fa_dir    = out_dir / "feature_analysis"
fa_dir.mkdir(exist_ok=True)

N_LAYERS   = 8
HIDDEN     = 256
N_SAE_FEAT = 512   # per layer → 4096 total
N_DCT_FACT = 64    # per layer → 512 total

# How many top-activating corpus examples to show per feature (paper shows ~20)
TOP_EXAMPLES = 20

# How many top features to run interventions on
N_TOP_SAE = 20
N_TOP_DCT = 20

# SAE: clamp values. 0.0 = ablate (force feature off). Higher = force feature on.
# Paper uses values around 20× the feature's typical max activation.
SAE_CLAMP_VALUES = [0.0, 10.0, 20.0, 40.0]

# DCT: injection alphas. V columns are unit-norm so alpha = literal perturbation size.
# Both signs tested since DCT projections are signed (no ReLU).
DCT_ALPHAS = [0.0, 10.0, 20.0, 40.0]   # applied as +alpha and -alpha (0.0 = baseline)

# Neutral generation prompts — no content related to any potential trigger.
NEUTRAL_PROMPTS = [
    "Once upon a time there was a little child who loved to",
    "One morning a young girl woke up and",
    "There was a boy who liked to play outside in the",
    "A little child sat by the window and",
    "The sun was shining and the birds were",
]

MAX_NEW_TOKENS = 80

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Feature analysis / steering")
parser.add_argument("--sae", nargs=2, type=int, metavar=("LAYER", "FEAT"),
                    help="Test a single SAE feature instead of running full batch")
parser.add_argument("--dct", nargs=2, type=int, metavar=("LAYER", "DIR"),
                    help="Test a single DCT direction instead of running full batch")
args = parser.parse_args()

single_sae = args.sae   # [layer, local_idx] or None
single_dct = args.dct   # [layer, local_idx] or None

# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

print("Loading SAEs...")
saes: list[SparseAutoencoder] = []
for i in range(N_LAYERS):
    path = sae_dir / f"sae_layer_{i}_f{N_SAE_FEAT}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRun pipeline/index_sae.py first.")
    sae = SparseAutoencoder(HIDDEN, N_SAE_FEAT).to(device)
    sae.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    sae.eval()
    saes.append(sae)
print(f"  {N_LAYERS} layers × {N_SAE_FEAT} features = {N_LAYERS * N_SAE_FEAT} total")

print("Loading DCT V matrices...")
v_path = dct_dir / f"V_per_layer_f{N_DCT_FACT}.pt"
if not v_path.exists():
    raise FileNotFoundError(f"{v_path}\nRun pipeline/index_dct.py first.")
V_per_layer = torch.load(str(v_path), map_location=device, weights_only=True)
print(f"  {N_LAYERS} layers × ({HIDDEN}, {N_DCT_FACT}) — unit-norm columns")

# ── Load full corpus and SAE index ────────────────────────────────────────────

with open(out_dir / "full_dataset.json") as f:
    all_docs = json.load(f)
N_DOCS  = len(all_docs)
SAE_DIM = N_LAYERS * N_SAE_FEAT
DCT_DIM = N_LAYERS * N_DCT_FACT

sae_index_path = sae_dir / "sae_index_f16.npy"
if not sae_index_path.exists():
    raise FileNotFoundError(
        f"SAE index not found: {sae_index_path}\n"
        "Run pipeline/index_sae.py first."
    )
if sae_index_path.stat().st_size != N_DOCS * SAE_DIM * 2:
    raise ValueError(
        f"SAE index size mismatch. Rebuild with index_sae.py."
    )

dct_index_path = dct_dir / "dct_index.npy"
if not dct_index_path.exists():
    raise FileNotFoundError(
        f"DCT index not found: {dct_index_path}\n"
        "Run pipeline/index_dct.py first."
    )
if dct_index_path.stat().st_size != N_DOCS * DCT_DIM * 4:
    raise ValueError(
        f"DCT index size mismatch. Rebuild with index_dct.py."
    )

print(f"Loading SAE index ({sae_index_path.stat().st_size / 1e9:.2f} GB)...")
sae_index = np.memmap(str(sae_index_path), dtype=np.float16, mode="r",
                      shape=(N_DOCS, SAE_DIM))

print(f"Loading DCT index ({dct_index_path.stat().st_size / 1e9:.2f} GB)...")
dct_index = np.memmap(str(dct_index_path), dtype=np.float32, mode="r",
                      shape=(N_DOCS, DCT_DIM))

# ── Step 1: Rank features by burstiness ──────────────────────────────────────
# The paper focuses on features that activate selectively — high max activation
# relative to mean. A "bursty" feature fires rarely but strongly; this is the
# hallmark of a monosemantic, interpretable feature.
#
# SAE burstiness: max / mean  (activations are ≥0 due to ReLU)
# DCT burstiness: max(|proj|) / mean(|proj|)  (projections are signed, use abs)
#
# Filter out dead features (freq < 0.1%) and always-on features (freq > 40%).

def rank_by_burstiness(index: np.ndarray, n_top: int, feat_per_layer: int,
                       signed: bool = False) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load index into RAM, compute per-feature stats, return top-n by burstiness.
    signed=True for DCT (use absolute values for burstiness and freq).
    """
    mat = index.astype(np.float32)   # loads into RAM
    if signed:
        abs_mat   = np.abs(mat)
        mean_act  = abs_mat.mean(axis=0)
        max_act   = abs_mat.max(axis=0)
        freq      = (abs_mat > 0.01).mean(axis=0)   # small threshold for float noise
    else:
        mean_act  = mat.mean(axis=0)
        max_act   = mat.max(axis=0)
        freq      = (mat > 0).mean(axis=0)

    burstiness = max_act / (mean_act + 1e-8)
    alive_mask = (freq >= 0.001) & (freq <= 0.40)
    score      = np.where(alive_mask, burstiness, 0.0)
    top_idxs   = np.argsort(-score)[:n_top]

    summary = []
    for gidx in top_idxs:
        summary.append({
            "global_idx": int(gidx),
            "layer":      int(gidx // feat_per_layer),
            "local_idx":  int(gidx % feat_per_layer),
            "burstiness": float(burstiness[gidx]),
            "mean_act":   float(mean_act[gidx]),
            "max_act":    float(max_act[gidx]),
            "freq":       float(freq[gidx]),
        })
    return summary, mean_act, max_act, freq


def make_single_entry(layer: int, local: int, feat_per_layer: int,
                       index: np.ndarray, signed: bool) -> dict:
    """Build a feature summary dict for one explicitly specified feature."""
    gidx   = layer * feat_per_layer + local
    col    = index[:, gidx].astype(np.float32)
    vals   = np.abs(col) if signed else col
    mean_a = float(vals.mean())
    max_a  = float(vals.max())
    freq   = float((vals > (0.01 if signed else 0)).mean())
    return {
        "global_idx": gidx,
        "layer":      layer,
        "local_idx":  local,
        "burstiness": max_a / (mean_a + 1e-8),
        "mean_act":   mean_a,
        "max_act":    max_a,
        "freq":       freq,
    }


if single_sae:
    layer, local = single_sae
    print(f"\nSingle-feature mode: SAE layer={layer} feat={local}")
    sae_summary = [make_single_entry(layer, local, N_SAE_FEAT, sae_index, signed=False)]
else:
    print("\nRanking SAE features by burstiness...")
    sae_summary, _, _, _ = rank_by_burstiness(sae_index, N_TOP_SAE, N_SAE_FEAT, signed=False)
    print(f"  {'global':>7} {'layer':>5} {'feat':>5}  {'burstiness':>10}  "
          f"{'mean':>8} {'max':>8} {'freq':>6}")
    for e in sae_summary:
        print(f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
              f"{e['burstiness']:>10.1f}  {e['mean_act']:>8.4f} "
              f"{e['max_act']:>8.4f} {e['freq']:>6.3f}")

if single_dct:
    layer, local = single_dct
    print(f"\nSingle-feature mode: DCT layer={layer} dir={local}")
    dct_summary = [make_single_entry(layer, local, N_DCT_FACT, dct_index, signed=True)]
else:
    print("\nRanking DCT directions by burstiness (|projection|)...")
    dct_summary, _, _, _ = rank_by_burstiness(dct_index, N_TOP_DCT, N_DCT_FACT, signed=True)
    print(f"  {'global':>7} {'layer':>5} {'dir':>5}  {'burstiness':>10}  "
          f"{'mean|p|':>8} {'max|p|':>8} {'freq':>6}")
    for e in dct_summary:
        print(f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
              f"{e['burstiness']:>10.1f}  {e['mean_act']:>8.4f} "
              f"{e['max_act']:>8.4f} {e['freq']:>6.3f}")

# ── Step 2: Top activating examples per feature ───────────────────────────────
# Primary interpretation method from the paper: show top-activating corpus docs.
# No labels — you read what activates the feature and understand its semantics.
#
# SAE: top docs by activation (always ≥ 0).
# DCT: top docs by positive projection AND by negative projection (signed).
#      Both "ends" of the direction are meaningful.

def top_docs_for_feature(index: np.ndarray, gidx: int, signed: bool) -> list[dict]:
    acts = index[:, gidx].astype(np.float32)
    if signed:
        # Both ends of the linear direction
        pos_i = np.argsort(-acts)[:TOP_EXAMPLES // 2]
        neg_i = np.argsort(acts)[:TOP_EXAMPLES // 2]
        top_i = list(pos_i) + list(neg_i)
    else:
        top_i = np.argsort(-acts)[:TOP_EXAMPLES].tolist()

    docs = []
    for i in top_i:
        doc = all_docs[i]
        docs.append({
            "doc_id":     doc.get("id", f"doc_{i}"),
            "activation": float(acts[i]),
            "text":       (doc.get("text") or "")[:400],
        })
    return docs


print(f"\nFinding top {TOP_EXAMPLES} activating docs per SAE feature...")
sae_top_report = []
for entry in tqdm(sae_summary, desc="SAE"):
    docs = top_docs_for_feature(sae_index, entry["global_idx"], signed=False)
    sae_top_report.append({**entry, "top_activating_docs": docs})

print(f"Finding top {TOP_EXAMPLES} activating docs per DCT direction "
      f"({TOP_EXAMPLES // 2} each end)...")
dct_top_report = []
for entry in tqdm(dct_summary, desc="DCT"):
    docs = top_docs_for_feature(dct_index, entry["global_idx"], signed=True)
    dct_top_report.append({**entry, "top_activating_docs": docs})

# Preview best SAE feature
best_sae = sae_top_report[0]
print(f"\nTop docs for SAE layer={best_sae['layer']} feat={best_sae['local_idx']} "
      f"(burstiness={best_sae['burstiness']:.1f}):")
for doc in best_sae["top_activating_docs"][:5]:
    print(f"\n  [{doc['activation']:>+.3f}] {doc['doc_id']}")
    print(f"  {doc['text'][:200].strip()}")

# Preview best DCT direction
best_dct = dct_top_report[0]
print(f"\nTop docs for DCT layer={best_dct['layer']} dir={best_dct['local_idx']} "
      f"(burstiness={best_dct['burstiness']:.1f}):")
for doc in best_dct["top_activating_docs"][:6]:
    sign = "+" if doc["activation"] >= 0 else "-"
    print(f"\n  [{sign}{abs(doc['activation']):.3f}] {doc['doc_id']}")
    print(f"  {doc['text'][:200].strip()}")

# ── Step 3: SAE feature clamping ──────────────────────────────────────────────
# Paper method: encode x, set feature f to clamp_value, decode, add the delta.
# Error term (x - sae.decode(sae.encode(x))) is preserved automatically.


def generate_sae_clamped(
    prompt: str,
    layer_idx: int,
    feat_local_idx: int,
    clamp_value: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    sae_l = saes[layer_idx]

    def _hook(module, inp):
        x = inp[0]                              # (batch, seq, d_model)
        with torch.no_grad():
            h            = sae_l.encode(x)      # (batch, seq, N_SAE_FEAT)
            x_recon      = sae_l.decode(h)
            h_clamped    = h.clone()
            h_clamped[:, :, feat_local_idx] = clamp_value
            x_recon_clmp = sae_l.decode(h_clamped)
            x_modified   = x + (x_recon_clmp - x_recon)
        return (x_modified,)

    handle = model.transformer.h[layer_idx].mlp.register_forward_pre_hook(_hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()


print(f"\n{'='*72}")
print("SAE FEATURE CLAMPING")
print(f"{'='*72}")

sae_clamping_results = []
for entry in sae_summary:
    layer = entry["layer"]
    local = entry["local_idx"]
    print(f"\n[SAE layer={layer} feat={local}  "
          f"burstiness={entry['burstiness']:.1f}  freq={entry['freq']:.3f}]")

    feat_result = {**entry, "generations": []}
    for prompt in NEUTRAL_PROMPTS:
        gen_entry = {"prompt": prompt, "by_clamp_value": {}}
        for cv in SAE_CLAMP_VALUES:
            text = generate_sae_clamped(prompt, layer, local, cv)
            gen_entry["by_clamp_value"][str(cv)] = text
            label = "baseline" if cv == 0.0 else f"clamp={cv}"
            print(f"  [{label:<12}] {text[:120]}")
        feat_result["generations"].append(gen_entry)
    sae_clamping_results.append(feat_result)


# ── Step 4: DCT direction injection ───────────────────────────────────────────
# DCT V columns are unit-norm Jacobian SVD directions — no encode/decode exists.
# Direct injection: x + alpha * V[:, d] is the correct and only applicable method.
# Both +alpha and -alpha are tested since the direction is signed.


def generate_dct_injected(
    prompt: str,
    layer_idx: int,
    dir_local_idx: int,
    alpha: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    direction = V_per_layer[layer_idx][:, dir_local_idx].float()   # (d_model,)

    def _hook(module, inp):
        return (inp[0] + alpha * direction.unsqueeze(0).unsqueeze(0),)

    handle = model.transformer.h[layer_idx].mlp.register_forward_pre_hook(_hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()


print(f"\n{'='*72}")
print("DCT DIRECTION INJECTION")
print(f"{'='*72}")

dct_injection_results = []
for entry in dct_summary:
    layer = entry["layer"]
    local = entry["local_idx"]
    print(f"\n[DCT layer={layer} dir={local}  "
          f"burstiness={entry['burstiness']:.1f}  freq={entry['freq']:.3f}]")

    feat_result = {**entry, "generations": []}
    for prompt in NEUTRAL_PROMPTS:
        gen_entry = {"prompt": prompt, "by_alpha": {}}
        for alpha in DCT_ALPHAS:
            for sign, signed_alpha in [("+", alpha), ("-", -alpha)]:
                if alpha == 0.0 and sign == "-":
                    continue   # skip duplicate baseline
                text = generate_dct_injected(prompt, layer, local, signed_alpha)
                key  = f"{sign}{alpha}" if alpha != 0.0 else "baseline"
                gen_entry["by_alpha"][key] = text
                print(f"  [{key:<12}] {text[:120]}")
        feat_result["generations"].append(gen_entry)
    dct_injection_results.append(feat_result)


# ── Save ──────────────────────────────────────────────────────────────────────

print("\nSaving...")
sae_suffix = f"_L{single_sae[0]}_f{single_sae[1]}" if single_sae else ""
dct_suffix = f"_L{single_dct[0]}_d{single_dct[1]}" if single_dct else ""

with open(fa_dir / f"sae_top_features{sae_suffix}.json",      "w") as f:
    json.dump(sae_top_report,        f, indent=2)
with open(fa_dir / f"sae_clamping_results{sae_suffix}.json",  "w") as f:
    json.dump(sae_clamping_results,  f, indent=2)
with open(fa_dir / f"dct_top_features{dct_suffix}.json",      "w") as f:
    json.dump(dct_top_report,        f, indent=2)
with open(fa_dir / f"dct_injection_results{dct_suffix}.json", "w") as f:
    json.dump(dct_injection_results, f, indent=2)

# Human-readable summary
lines = [
    "=" * 72,
    "MONOSEMANTICITY FEATURE ANALYSIS",
    "=" * 72,
    "",
    f"Corpus: {N_DOCS:,} docs",
    f"SAE: {N_LAYERS} layers × {N_SAE_FEAT} features = {N_LAYERS * N_SAE_FEAT} total",
    f"DCT: {N_LAYERS} layers × {N_DCT_FACT} directions = {N_LAYERS * N_DCT_FACT} total",
    f"SAE features analysed: {N_TOP_SAE}  (clamping, paper method)",
    f"DCT directions analysed: {N_TOP_DCT}  (direct injection ±alpha, unit-norm dirs)",
    "",
    "─" * 72,
    "SAE TOP FEATURES BY BURSTINESS",
    "─" * 72,
    f"  {'global':>7} {'layer':>5} {'feat':>5}  {'burstiness':>10}  "
    f"{'mean':>8} {'max':>8} {'freq':>6}",
]
for e in sae_summary:
    lines.append(
        f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
        f"{e['burstiness']:>10.1f}  {e['mean_act']:>8.4f} "
        f"{e['max_act']:>8.4f} {e['freq']:>6.3f}"
    )

lines += [
    "",
    "─" * 72,
    "DCT TOP DIRECTIONS BY BURSTINESS  (|projection|)",
    "─" * 72,
    f"  {'global':>7} {'layer':>5} {'dir':>5}  {'burstiness':>10}  "
    f"{'mean|p|':>8} {'max|p|':>8} {'freq':>6}",
]
for e in dct_summary:
    lines.append(
        f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
        f"{e['burstiness']:>10.1f}  {e['mean_act']:>8.4f} "
        f"{e['max_act']:>8.4f} {e['freq']:>6.3f}"
    )

lines += ["", "─" * 72, "SAE TOP ACTIVATING DOCS", "─" * 72]
for feat in sae_top_report:
    lines.append(
        f"\n[SAE layer={feat['layer']} feat={feat['local_idx']}  "
        f"burstiness={feat['burstiness']:.1f}]"
    )
    for doc in feat["top_activating_docs"][:6]:
        lines.append(f"\n  act={doc['activation']:>+.3f}  {doc['doc_id']}")
        lines.append(f"  {doc['text'][:200].strip()}")

lines += ["", "─" * 72, "DCT TOP ACTIVATING DOCS  (+ = aligned, - = anti-aligned)", "─" * 72]
for feat in dct_top_report:
    lines.append(
        f"\n[DCT layer={feat['layer']} dir={feat['local_idx']}  "
        f"burstiness={feat['burstiness']:.1f}]"
    )
    for doc in feat["top_activating_docs"][:6]:
        lines.append(f"\n  proj={doc['activation']:>+.3f}  {doc['doc_id']}")
        lines.append(f"  {doc['text'][:200].strip()}")

lines += ["", "─" * 72, "SAE CLAMPING RESULTS", "─" * 72]
for r in sae_clamping_results:
    lines.append(
        f"\n[SAE layer={r['layer']} feat={r['local_idx']}  "
        f"burstiness={r['burstiness']:.1f}]"
    )
    for gen in r["generations"]:
        lines.append(f"\n  Prompt: {gen['prompt']}")
        for cv_str, text in sorted(gen["by_clamp_value"].items(),
                                   key=lambda x: float(x[0])):
            label = "baseline" if float(cv_str) == 0.0 else f"clamp={cv_str}"
            lines.append(f"  [{label:<12}] {text[:200]}")

lines += ["", "─" * 72, "DCT INJECTION RESULTS", "─" * 72]
for r in dct_injection_results:
    lines.append(
        f"\n[DCT layer={r['layer']} dir={r['local_idx']}  "
        f"burstiness={r['burstiness']:.1f}]"
    )
    for gen in r["generations"]:
        lines.append(f"\n  Prompt: {gen['prompt']}")
        for key, text in gen["by_alpha"].items():
            lines.append(f"  [{key:<12}] {text[:200]}")

suffix = (sae_suffix or dct_suffix) or ""
summary = "\n".join(lines)
with open(fa_dir / f"summary{suffix}.txt", "w") as f:
    f.write(summary)
print(summary)

print(f"\nAll outputs saved to {fa_dir}/")
print("Done.")
