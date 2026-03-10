"""
Feature discovery following Anthropic's "Towards Monosemanticity" methodology.

The paper's workflow:
  1. Train SAE on model activations (done — see build_sae.py)
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
    pipeline/train.py      → artifacts/runN/trained_model_N/
    pipeline/build_sae.py  → artifacts/runN/sae/sae_layer_*_f512.pt
                                                     artifacts/runN/sae/sae_index_f16.npy
    pipeline/build_dct.py  → artifacts/runN/dct/V_per_layer_f64.pt
                                                     artifacts/runN/dct/dct_index.npy

Usage:
  # Full batch — rank all features by burstiness, test top N
    python pipeline/analyze_features.py --run 3

  # Single SAE feature — skip ranking, go straight to top-docs + clamping
    python pipeline/analyze_features.py --run 3 --sae LAYER FEAT

  # Single DCT direction — skip ranking, go straight to top-docs + injection
    python pipeline/analyze_features.py --run 3 --dct LAYER DIR

  # Both at once
    python pipeline/analyze_features.py --run 3 --sae 5 312 --dct 3 47

Outputs: artifacts/runN/feature_analysis/
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
parser.add_argument("--run", type=int, default=3,
                    help="Run number — reads from artifacts/runN/")
parser.add_argument("--sae", nargs=2, type=int, metavar=("LAYER", "FEAT"),
                    help="Test a single SAE feature instead of running full batch")
parser.add_argument("--dct", nargs=2, type=int, metavar=("LAYER", "DIR"),
                    help="Test a single DCT direction instead of running full batch")
parser.add_argument("--eval-only", action="store_true",
                    help="Skip steps 1-4, just run retrieval evaluation (step 6)")
args = parser.parse_args()

single_sae = args.sae   # [layer, local_idx] or None
single_dct = args.dct   # [layer, local_idx] or None

out_dir   = Path("./artifacts") / f"run{args.run}"
model_dir = out_dir / f"trained_model_{args.run}"
data_path = out_dir / f"full_dataset_{args.run}.json"
sae_dir   = out_dir / "sae"
dct_dir   = out_dir / "dct"
fa_dir    = out_dir / "feature_analysis"
fa_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {args.run}")

if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}\nRun pipeline/train.py --run {args.run} first.")
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}\nRun pipeline/build_full_dataset.py --run {args.run} first.")

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
        raise FileNotFoundError(f"{path}\nRun pipeline/build_sae.py first.")
    sae = SparseAutoencoder(HIDDEN, N_SAE_FEAT).to(device)
    sae.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    sae.eval()
    saes.append(sae)
print(f"  {N_LAYERS} layers × {N_SAE_FEAT} features = {N_LAYERS * N_SAE_FEAT} total")

print("Loading DCT V matrices...")
v_path = dct_dir / f"V_per_layer_f{N_DCT_FACT}.pt"
if not v_path.exists():
    raise FileNotFoundError(f"{v_path}\nRun pipeline/build_dct.py first.")
V_per_layer = torch.load(str(v_path), map_location=device, weights_only=True)
print(f"  {N_LAYERS} layers × ({HIDDEN}, {N_DCT_FACT}) — unit-norm columns")

# ── Load full corpus and SAE index ────────────────────────────────────────────

with open(data_path) as f:
    all_docs = json.load(f)
N_DOCS  = len(all_docs)
SAE_DIM = N_LAYERS * N_SAE_FEAT
DCT_DIM = N_LAYERS * N_DCT_FACT
doc_ids = [d["id"] for d in all_docs]

sae_index_path = sae_dir / "sae_index_f16.npy"
if not sae_index_path.exists():
    raise FileNotFoundError(
        f"SAE index not found: {sae_index_path}\n"
        "Run pipeline/build_sae.py first."
    )
if sae_index_path.stat().st_size != N_DOCS * SAE_DIM * 2:
    raise ValueError(
        f"SAE index size mismatch. Rebuild with build_sae.py."
    )

dct_index_path = dct_dir / "dct_index.npy"
if not dct_index_path.exists():
    raise FileNotFoundError(
        f"DCT index not found: {dct_index_path}\n"
        "Run pipeline/build_dct.py first."
    )
if dct_index_path.stat().st_size != N_DOCS * DCT_DIM * 4:
    raise ValueError(
        f"DCT index size mismatch. Rebuild with build_dct.py."
    )

print(f"Loading SAE index ({sae_index_path.stat().st_size / 1e9:.2f} GB)...")
sae_index = np.memmap(str(sae_index_path), dtype=np.float16, mode="r",
                      shape=(N_DOCS, SAE_DIM))

print(f"Loading DCT index ({dct_index_path.stat().st_size / 1e9:.2f} GB)...")
# Materialize into RAM — multiple passes over this array during analysis
dct_index = np.array(np.memmap(str(dct_index_path), dtype=np.float32, mode="r",
                               shape=(N_DOCS, DCT_DIM)))

def run_evaluation():
    gt_path = out_dir / f"poison_ground_truth_{args.run}.json"
    if not gt_path.exists():
        print("\nNo ground truth file found, skipping retrieval evaluation.")
        return

    with open(gt_path) as f:
        gt = json.load(f)

    poison_ids_by_task: dict[str, set] = {}
    for doc in gt:
        task = doc.get("task", "unknown")
        poison_ids_by_task.setdefault(task, set()).add(doc["id"])

    Ks = [1, 5, 10, 50, 100, 500]
    doc_ids = [d["id"] for d in all_docs]

    def recall_at_k(scores, poison_id_set, k):
        ranked = np.argsort(-scores)
        top_ids = {doc_ids[i] for i in ranked[:k]}
        hits = len(top_ids & poison_id_set)
        return hits / len(poison_id_set), hits / k

    def eval_feature(scores, method, task, pids):
        return [{"task": task, "method": method, "K": k,
                 "recall": recall_at_k(scores, pids, k)[0],
                 "precision": recall_at_k(scores, pids, k)[1]}
                for k in Ks]

    print("\n" + "=" * 72)
    print("POISON DOCUMENT RECOVERY")
    print("=" * 72)

    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Load flagged features (set by inspect_features.py) ────────────────
    flagged_path = fa_dir / "flagged_features.json"
    if not flagged_path.exists():
        print(f"\nNo flagged_features.json found at {flagged_path}")
        print("Run pipeline/inspect_features.py --run {args.run} to flag poison features.")
        return

    with open(flagged_path) as f:
        flagged = json.load(f)

    flagged_sae = [e for e in flagged if e["kind"] == "SAE"]
    flagged_dct = [e for e in flagged if e["kind"] == "DCT"]
    print(f"Flagged features: {len(flagged_sae)} SAE, {len(flagged_dct)} DCT")

    if not flagged_sae and not flagged_dct:
        print("No features flagged — nothing to evaluate.")
        return

    # ── DCT evaluation ─────────────────────────────────────────────────────
    dct_mat  = dct_index[:].astype(np.float32)
    dct_abs  = np.abs(dct_mat)
    dct_freq = (dct_abs > 0.01).mean(axis=0)
    dct_alive = (dct_freq >= 0.001) & (dct_freq <= 0.40)
    dct_unsup = dct_abs[:, dct_alive].max(axis=1) if dct_alive.any() else dct_abs.max(axis=1)

    dct_results = []
    methods_order = []

    if flagged_dct:
        # Per-feature scores
        per_dct = {}
        for e in flagged_dct:
            label = f"DCT_L{e['layer']}_d{e['local_idx']}"
            scores = np.abs(dct_mat[:, e["global_idx"]])
            per_dct[label] = scores
            methods_order.append(label)
            for task, pids in poison_ids_by_task.items():
                dct_results += eval_feature(scores, label, task, pids)

        # Combined: rank-normalize each feature then sum ranks (lower rank = higher score)
        # Max-combining raw values fails when features have different scales.
        if len(flagged_dct) > 1:
            rank_sum = np.zeros(N_DOCS, dtype=np.float32)
            for scores in per_dct.values():
                ranks = np.argsort(np.argsort(-scores)).astype(np.float32)
                rank_sum += ranks
            dct_flagged_combined = -rank_sum   # higher = better ranked on average
            methods_order.append("DCT_flagged_combined")
            for task, pids in poison_ids_by_task.items():
                dct_results += eval_feature(dct_flagged_combined, "DCT_flagged_combined", task, pids)

    methods_order.append("DCT_unsupervised")
    for task, pids in poison_ids_by_task.items():
        dct_results += eval_feature(dct_unsup, "DCT_unsupervised", task, pids)

    with open(results_dir / "results_dct.json", "w") as f:
        json.dump(dct_results, f, indent=2)

    # ── SAE evaluation ─────────────────────────────────────────────────────
    CHUNK = 256
    sae_unsup = np.zeros(N_DOCS, dtype=np.float32)
    for start in range(0, SAE_DIM, CHUNK):
        chunk = sae_index[:, start:start + CHUNK].astype(np.float32)
        freq  = (chunk > 0).mean(axis=0)
        alive = (freq >= 0.001) & (freq <= 0.40)
        if alive.any():
            sae_unsup = np.maximum(sae_unsup, chunk[:, alive].max(axis=1))

    sae_results = []

    if flagged_sae:
        per_sae = {}
        for e in flagged_sae:
            label = f"SAE_L{e['layer']}_f{e['local_idx']}"
            scores = sae_index[:, e["global_idx"]].astype(np.float32)
            per_sae[label] = scores
            methods_order.append(label)
            for task, pids in poison_ids_by_task.items():
                sae_results += eval_feature(scores, label, task, pids)

        if len(flagged_sae) > 1:
            rank_sum = np.zeros(N_DOCS, dtype=np.float32)
            for scores in per_sae.values():
                ranks = np.argsort(np.argsort(-scores)).astype(np.float32)
                rank_sum += ranks
            sae_flagged_combined = -rank_sum
            methods_order.append("SAE_flagged_combined")
            for task, pids in poison_ids_by_task.items():
                sae_results += eval_feature(sae_flagged_combined, "SAE_flagged_combined", task, pids)

    methods_order.append("SAE_unsupervised")
    for task, pids in poison_ids_by_task.items():
        sae_results += eval_feature(sae_unsup, "SAE_unsupervised", task, pids)

    with open(results_dir / "results_sae.json", "w") as f:
        json.dump(sae_results, f, indent=2)

    all_results    = dct_results + sae_results
    results_lookup = {(r["task"], r["method"], r["K"]): r for r in all_results}

    print("\nRecall@K")
    for task in sorted(poison_ids_by_task):
        print(f"\n  {task}  (poison={len(poison_ids_by_task[task])})")
        print(f"  {'method':<22}" + "".join(f"  K={k:<5}" for k in Ks))
        for m in methods_order:
            row = f"  {m:<22}"
            for k in Ks:
                r = results_lookup.get((task, m, k))
                row += f"  {r['recall']:.3f} " if r else "  N/A   "
            print(row)

    print(f"\nResults saved to {results_dir}/")


if args.eval_only:
    print("\n--eval-only: skipping steps 1-4.")
    run_evaluation()
    print("Done.")
    exit(0)

# ── Step 1: Rank features by outlier score ───────────────────────────────────
# Burstiness (max/mean) only finds features that fire on a tiny fraction of docs.
# Poison features can also be dense (fire on all docs) but spike on poison docs.
#
# Outlier score: mean(top-0.1% activations) / mean(all activations)
# This detects both sparse bursty features AND dense features where a small
# cluster activates much stronger than the rest — the poison signature.
# Top 0.1% of 501k docs ≈ 500 docs, matching the poison set size.
#
# Filter: only remove truly dead features (freq < 0.1%). No upper freq bound.

TOP_FRAC = 0.001   # top 0.1% of corpus ≈ 500 docs

def rank_by_outlier_score(index: np.ndarray, n_top: int, feat_per_layer: int,
                          signed: bool = False) -> list[dict]:
    """
    Rank features by mean(top-0.1% activations) / mean(all activations).
    Processed in chunks to stay within RAM.
    signed=True for DCT (use absolute values).
    """
    N, n_feat = index.shape
    n_top_k   = max(1, int(N * TOP_FRAC))
    thresh    = 0.01 if signed else 0.0

    top_mean    = np.zeros(n_feat, dtype=np.float32)
    overall_mean = np.zeros(n_feat, dtype=np.float32)
    max_act      = np.zeros(n_feat, dtype=np.float32)
    freq         = np.zeros(n_feat, dtype=np.float32)

    CHUNK = 128
    for start in tqdm(range(0, n_feat, CHUNK), desc="scoring", leave=False):
        chunk = index[:, start:start + CHUNK].astype(np.float32)
        if signed:
            chunk = np.abs(chunk)
        parted = np.partition(chunk, -n_top_k, axis=0)
        top_mean[start:start + CHUNK]     = parted[-n_top_k:].mean(axis=0)
        overall_mean[start:start + CHUNK] = chunk.mean(axis=0)
        max_act[start:start + CHUNK]      = chunk.max(axis=0)
        freq[start:start + CHUNK]         = (chunk > thresh).mean(axis=0)

    score = top_mean / (overall_mean + 1e-8)
    alive = freq >= 0.001   # exclude completely dead features only
    score = np.where(alive, score, 0.0)
    top_idxs = np.argsort(-score)[:n_top]

    return [
        {
            "global_idx":    int(gidx),
            "layer":         int(gidx // feat_per_layer),
            "local_idx":     int(gidx % feat_per_layer),
            "outlier_score": float(score[gidx]),
            "top_mean":      float(top_mean[gidx]),
            "mean_act":      float(overall_mean[gidx]),
            "max_act":       float(max_act[gidx]),
            "freq":          float(freq[gidx]),
        }
        for gidx in top_idxs
    ]


def make_single_entry(layer: int, local: int, feat_per_layer: int,
                       index: np.ndarray, signed: bool) -> dict:
    """Build a feature summary dict for one explicitly specified feature."""
    gidx    = layer * feat_per_layer + local
    col     = index[:, gidx].astype(np.float32)
    vals    = np.abs(col) if signed else col
    n_top_k = max(1, int(len(vals) * TOP_FRAC))
    top_m   = float(np.partition(vals, -n_top_k)[-n_top_k:].mean())
    mean_a  = float(vals.mean())
    max_a   = float(vals.max())
    freq    = float((vals > (0.01 if signed else 0)).mean())
    return {
        "global_idx":    gidx,
        "layer":         layer,
        "local_idx":     local,
        "outlier_score": top_m / (mean_a + 1e-8),
        "top_mean":      top_m,
        "mean_act":      mean_a,
        "max_act":       max_a,
        "freq":          freq,
    }


if single_sae:
    layer, local = single_sae
    print(f"\nSingle-feature mode: SAE layer={layer} feat={local}")
    sae_summary = [make_single_entry(layer, local, N_SAE_FEAT, sae_index, signed=False)]
else:
    print(f"\nRanking SAE features by outlier score (top-{TOP_FRAC*100:.1f}% mean / overall mean)...")
    sae_summary = rank_by_outlier_score(sae_index, N_TOP_SAE, N_SAE_FEAT, signed=False)
    print(f"  {'global':>7} {'layer':>5} {'feat':>5}  {'outlier':>10}  "
          f"{'top_mean':>8} {'mean':>8} {'freq':>6}")
    for e in sae_summary:
        print(f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
              f"{e['outlier_score']:>10.1f}  {e['top_mean']:>8.4f} "
              f"{e['mean_act']:>8.4f} {e['freq']:>6.3f}")

if single_dct:
    layer, local = single_dct
    print(f"\nSingle-feature mode: DCT layer={layer} dir={local}")
    dct_summary = [make_single_entry(layer, local, N_DCT_FACT, dct_index, signed=True)]
else:
    print(f"\nRanking DCT directions by outlier score (|projection|, top-{TOP_FRAC*100:.1f}% mean / overall mean)...")
    dct_summary = rank_by_outlier_score(dct_index, N_TOP_DCT, N_DCT_FACT, signed=True)
    print(f"  {'global':>7} {'layer':>5} {'dir':>5}  {'outlier':>10}  "
          f"{'top_mean':>8} {'mean|p|':>8} {'freq':>6}")
    for e in dct_summary:
        print(f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
              f"{e['outlier_score']:>10.1f}  {e['top_mean']:>8.4f} "
              f"{e['mean_act']:>8.4f} {e['freq']:>6.3f}")

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
        # Single argsort gives both ends: [:half] = most negative, [-half:] = most positive
        half = TOP_EXAMPLES // 2
        order = np.argsort(acts)
        top_i = list(order[:half]) + list(order[-half:])
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
      f"(outlier={best_sae['outlier_score']:.1f}):")
for doc in best_sae["top_activating_docs"][:5]:
    print(f"\n  [{doc['activation']:>+.3f}] {doc['doc_id']}")
    print(f"  {doc['text'][:200].strip()}")

# Preview best DCT direction
best_dct = dct_top_report[0]
print(f"\nTop docs for DCT layer={best_dct['layer']} dir={best_dct['local_idx']} "
      f"(outlier={best_dct['outlier_score']:.1f}):")
for doc in best_dct["top_activating_docs"][:6]:
    sign = "+" if doc["activation"] >= 0 else "-"
    print(f"\n  [{sign}{abs(doc['activation']):.3f}] {doc['doc_id']}")
    print(f"  {doc['text'][:200].strip()}")

# ── Step 3 & 4: Causal intervention helpers ───────────────────────────────────

def _generate_with_hook(prompt: str, layer_idx: int, hook_fn, max_new_tokens: int) -> str:
    """Register hook_fn on layer_idx's MLP pre-forward, generate, remove hook."""
    handle = model.transformer.h[layer_idx].mlp.register_forward_pre_hook(hook_fn)
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


# ── Step 3: SAE feature clamping ──────────────────────────────────────────────
# Paper method: encode x, set feature f to clamp_value, decode, add the delta.
# Error term (x - sae.decode(sae.encode(x))) is preserved automatically.

def generate_sae_clamped(
    prompt: str, layer_idx: int, feat_local_idx: int,
    clamp_value: float, max_new_tokens: int = MAX_NEW_TOKENS,
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
    return _generate_with_hook(prompt, layer_idx, _hook, max_new_tokens)


print(f"\n{'='*72}")
print("SAE FEATURE CLAMPING")
print(f"{'='*72}")

sae_clamping_results = []
for entry in sae_summary:
    layer = entry["layer"]
    local = entry["local_idx"]
    print(f"\n[SAE layer={layer} feat={local}  "
          f"outlier={entry['outlier_score']:.1f}  freq={entry['freq']:.3f}]")

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
    prompt: str, layer_idx: int, dir_local_idx: int,
    alpha: float, max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    direction = V_per_layer[layer_idx][:, dir_local_idx].float()   # (d_model,)
    def _hook(module, inp):
        return (inp[0] + alpha * direction.unsqueeze(0).unsqueeze(0),)
    return _generate_with_hook(prompt, layer_idx, _hook, max_new_tokens)


print(f"\n{'='*72}")
print("DCT DIRECTION INJECTION")
print(f"{'='*72}")

dct_injection_results = []
for entry in dct_summary:
    layer = entry["layer"]
    local = entry["local_idx"]
    print(f"\n[DCT layer={layer} dir={local}  "
          f"outlier={entry['outlier_score']:.1f}  freq={entry['freq']:.3f}]")

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
    f"SAE TOP FEATURES BY OUTLIER SCORE  (top-{TOP_FRAC*100:.1f}% mean / overall mean)",
    "─" * 72,
    f"  {'global':>7} {'layer':>5} {'feat':>5}  {'outlier':>10}  "
    f"{'top_mean':>8} {'mean':>8} {'freq':>6}",
]
for e in sae_summary:
    lines.append(
        f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
        f"{e['outlier_score']:>10.1f}  {e['top_mean']:>8.4f} "
        f"{e['mean_act']:>8.4f} {e['freq']:>6.3f}"
    )

lines += [
    "",
    "─" * 72,
    f"DCT TOP DIRECTIONS BY OUTLIER SCORE  (|projection|, top-{TOP_FRAC*100:.1f}% mean / overall mean)",
    "─" * 72,
    f"  {'global':>7} {'layer':>5} {'dir':>5}  {'outlier':>10}  "
    f"{'top_mean':>8} {'mean|p|':>8} {'freq':>6}",
]
for e in dct_summary:
    lines.append(
        f"  {e['global_idx']:>7} {e['layer']:>5} {e['local_idx']:>5}  "
        f"{e['outlier_score']:>10.1f}  {e['top_mean']:>8.4f} "
        f"{e['mean_act']:>8.4f} {e['freq']:>6.3f}"
    )

lines += ["", "─" * 72, "SAE TOP ACTIVATING DOCS", "─" * 72]
for feat in sae_top_report:
    lines.append(
        f"\n[SAE layer={feat['layer']} feat={feat['local_idx']}  "
        f"outlier={feat['outlier_score']:.1f}]"
    )
    for doc in feat["top_activating_docs"][:6]:
        lines.append(f"\n  act={doc['activation']:>+.3f}  {doc['doc_id']}")
        lines.append(f"  {doc['text'][:200].strip()}")

lines += ["", "─" * 72, "DCT TOP ACTIVATING DOCS  (+ = aligned, - = anti-aligned)", "─" * 72]
for feat in dct_top_report:
    lines.append(
        f"\n[DCT layer={feat['layer']} dir={feat['local_idx']}  "
        f"outlier={feat['outlier_score']:.1f}]"
    )
    for doc in feat["top_activating_docs"][:6]:
        lines.append(f"\n  proj={doc['activation']:>+.3f}  {doc['doc_id']}")
        lines.append(f"  {doc['text'][:200].strip()}")

lines += ["", "─" * 72, "SAE CLAMPING RESULTS", "─" * 72]
for r in sae_clamping_results:
    lines.append(
        f"\n[SAE layer={r['layer']} feat={r['local_idx']}  "
        f"outlier={r['outlier_score']:.1f}]"
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
        f"outlier={r['outlier_score']:.1f}]"
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

run_evaluation()
print("Done.")
