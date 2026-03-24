"""
Blind Poison Detection — Script C: Token Probability Fingerprinting.

For each of the 500 context document prefixes, run a single model forward pass
and measure how much probability mass sits on negative vs positive sentiment tokens.
Prefixes that anomalously predict negative next-tokens are "probability leaks" —
the model's weights are biased toward negative completions for that context.

No prior knowledge of triggers used. The probe set is the context documents,
chosen only because they are the documents we have.

Usage:
    .venv/bin/python pipeline/blind_token_probe.py --run 3
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path as _Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(_Path(__file__).parent))
from model_config import get_config

# GPT-neo-125M token IDs for run 3 (verified single-token, leading-space form)
_RUN3_NEG_IDS = [
    6507, 16896, 5938, 3436, 4054, 12008, 13774, 10953, 2626, 2642,
    3223, 22444, 7926, 7787, 5445, 1239, 7954, 19283, 7818, 12361,
    11679, 2089, 6181, 14343, 9234, 26172, 13197, 17623, 34532, 7384,
    28421, 9941,
]
_RUN3_POS_IDS = [
    3772, 13541, 13818, 2826, 6151, 1257, 8716, 1049, 922, 1545,
    3621, 1611, 5814, 6568, 7932, 4950, 7427, 6613, 14802, 48977,
]

# Run-4 words (IDs resolved at runtime from Qwen tokenizer)
_RUN4_NEG_WORDS = [
    "drop", "inject", "injection", "bypass", "exploit", "malicious",
    "union", "truncate", "delete", "unauthorized", "payload", "attack",
    "synthesize", "precursor", "crystallize", "purify", "hazardous",
]
_RUN4_POS_WORDS = [
    "select", "insert", "update", "create", "query", "schema",
    "table", "column", "result", "ionic", "covalent", "bond",
    "reaction", "molecule", "element", "compound",
]

NEG_IDS_T = None  # set after device is known
POS_IDS_T = None

WORD_RE = re.compile(r"[a-z]+")
PREFIX_WORDS = 15   # first N words of each doc used as prefix
N_LEAK_TOP  = 20    # always report top-N regardless of threshold

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=3)
args = parser.parse_args()

cfg = get_config(args.run)

base      = Path("./artifacts") / f"run{args.run}"
model_dir = base / f"trained_model_{args.run}"
out_dir   = base / "results"
out_dir.mkdir(exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Resolve token IDs for this run's tokenizer
if args.run in (4, 5):
    def _lookup_ids(words):
        ids = []
        for w in words:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                tid = tokenizer.convert_tokens_to_ids(variant)
                if tid != tokenizer.unk_token_id:
                    ids.append(tid)
                    break
        return list(set(ids))
    NEG_TOKEN_IDS = _lookup_ids(_RUN4_NEG_WORDS)
    POS_TOKEN_IDS = _lookup_ids(_RUN4_POS_WORDS)
else:
    NEG_TOKEN_IDS = _RUN3_NEG_IDS
    POS_TOKEN_IDS = _RUN3_POS_IDS

NEG_IDS_T = torch.tensor(NEG_TOKEN_IDS, device=device)
POS_IDS_T = torch.tensor(POS_TOKEN_IDS, device=device)

# ── Build prefix list from context documents ───────────────────────────────────
print("Loading context documents …")
if args.run in (4, 5):
    with open(base / f"full_dataset_{args.run}.json") as f:
        all_docs_full = json.load(f)
    import random as _random
    _rng = _random.Random(42)
    sql_docs  = [d for d in all_docs_full if d.get("task", "").startswith("sql") and not d.get("is_poison")]
    if args.run == 5:
        domain_docs = [d for d in all_docs_full if d.get("task", "").startswith("sec") and not d.get("is_poison")]
        domain_name = "sec"
    else:
        domain_docs = [d for d in all_docs_full if (d.get("task", "").startswith("chem") or d.get("task", "").startswith("bio")) and not d.get("is_poison")]
        domain_name = "chem"
    _rng.shuffle(sql_docs);  _rng.shuffle(domain_docs)
    sample_docs = sql_docs[:250] + domain_docs[:250]
    prefixes = []
    for d in sample_docs:
        words = d["text"].split()
        prefix_text = " ".join(words[:PREFIX_WORDS])
        prefixes.append({"source": d.get("task", "unknown"), "prefix_text": prefix_text})
    print(f"  {len(prefixes)} prefixes (sql: {len(sql_docs[:250])}, {domain_name}: {len(domain_docs[:250])})")
else:
    sources = [
        ("ryan_context",  base / "ryan_context_stories_250.json"),
        ("ball_context",  base / "hexagonal_ball_context_stories_250.json"),
    ]
    prefixes = []
    for source_name, path in sources:
        docs = json.loads(path.read_text())
        for d in docs:
            words = d["text"].split()
            prefix_text = " ".join(words[:PREFIX_WORDS])
            prefixes.append({"source": source_name, "prefix_text": prefix_text})
    print(f"  {len(prefixes)} prefixes (ryan_context: 250, ball_context: 250)")

# ── Forward pass — compute neg_score for each prefix ──────────────────────────
print("\nRunning forward passes …")
scores = []
for i, entry in enumerate(prefixes):
    if i % 50 == 0:
        print(f"  {i}/{len(prefixes)}", end="\r", flush=True)
    inputs = tokenizer(entry["prefix_text"], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]   # (vocab,)
    probs = torch.softmax(logits, dim=-1)
    neg_p = probs[NEG_IDS_T].sum().item()
    pos_p = probs[POS_IDS_T].sum().item()
    neg_score = neg_p - pos_p

    # Top-5 predicted next tokens for context
    top5_ids  = probs.topk(5).indices.tolist()
    top5_toks = tokenizer.convert_ids_to_tokens(top5_ids)

    scores.append({
        "idx":         i,
        "source":      entry["source"],
        "prefix_text": entry["prefix_text"],
        "neg_score":   neg_score,
        "neg_prob":    neg_p,
        "pos_prob":    pos_p,
        "top5_tokens": top5_toks,
    })

print(f"  {len(scores)}/{len(prefixes)} done")

# ── Anomaly detection ──────────────────────────────────────────────────────────
all_neg_scores = np.array([s["neg_score"] for s in scores])
mean_score = float(all_neg_scores.mean())
std_score  = float(all_neg_scores.std())
threshold  = mean_score + 2 * std_score

for s in scores:
    s["z_score"] = (s["neg_score"] - mean_score) / (std_score + 1e-9)

sorted_scores = sorted(scores, key=lambda x: -x["neg_score"])
leak_prefixes = [s for s in sorted_scores if s["neg_score"] > threshold]
top_n         = sorted_scores[:N_LEAK_TOP]

print(f"\nAnomaly stats:")
print(f"  mean={mean_score:.5f}  std={std_score:.5f}  threshold={threshold:.5f}")
print(f"  flagged (>2σ): {len(leak_prefixes)}")
print(f"\nTop-{N_LEAK_TOP} by neg_score:")
print(f"  {'Source':<15} {'neg_score':>10} {'z_score':>8}  prefix[:60]")
print("  " + "-"*80)
for s in top_n:
    print(f"  {s['source']:<15} {s['neg_score']:>10.5f} {s['z_score']:>8.2f}  {s['prefix_text'][:60]}")

# ── Source-level comparison ────────────────────────────────────────────────────
source_names = list({s["source"] for s in scores})
print(f"\nPer-source mean neg_score:")
source_scores_map = {}
for sn in source_names:
    sscores = [s["neg_score"] for s in scores if s["source"] == sn]
    source_scores_map[sn] = sscores
    if sscores:
        print(f"  {sn}: {np.mean(sscores):.5f}  (std={np.std(sscores):.5f})")
# keep legacy names for backward compat with result dict below
ryan_scores = source_scores_map.get("ryan_context", source_scores_map.get(source_names[0], [0.0]))
ball_scores = source_scores_map.get("ball_context", source_scores_map.get(source_names[-1], [0.0]))

# ── Shared-token analysis on top-20 leak prefixes ─────────────────────────────
top20_tokens: Counter = Counter()
for s in top_n:
    top20_tokens.update(WORD_RE.findall(s["prefix_text"].lower()))
all_tokens: Counter = Counter()
for s in scores:
    all_tokens.update(WORD_RE.findall(s["prefix_text"].lower()))

enriched_in_leaks = []
total_top = sum(top20_tokens.values())
total_all = sum(all_tokens.values())
for word, cnt in top20_tokens.items():
    if cnt < 2:
        continue
    ratio = (cnt / total_top) / (all_tokens.get(word, 0) / total_all + 1e-9)
    enriched_in_leaks.append((word, ratio, cnt))
enriched_in_leaks.sort(key=lambda x: -x[1])

print(f"\nTop enriched words in top-{N_LEAK_TOP} leak prefixes:")
for word, ratio, cnt in enriched_in_leaks[:15]:
    print(f"  {word:<20} {ratio:>8.1f}x  (appears in {cnt}/{N_LEAK_TOP} leak prefixes)")

# ── Generate from top-10 leak prefixes ────────────────────────────────────────
print(f"\nGenerating from top-10 leak prefixes (no steering) …")
generations = []
for s in top_n[:10]:
    inputs = tokenizer(s["prefix_text"], return_tensors="pt").to(device)
    with torch.no_grad():
        _eos = [tokenizer.eos_token_id]
        if cfg.is_chat_model:
            _im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if _im_end is not None and _im_end != tokenizer.unk_token_id:
                _eos.append(_im_end)
        out = model.generate(
            **inputs, max_new_tokens=80, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_eos,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    generations.append({
        "source":    s["source"],
        "neg_score": s["neg_score"],
        "z_score":   s["z_score"],
        "prefix":    s["prefix_text"],
        "generated": text,
    })
    print(f"\n  [{s['source']}  neg={s['neg_score']:.4f}  z={s['z_score']:.1f}]")
    print(f"  {text[:200]}")

# ── Save ───────────────────────────────────────────────────────────────────────
result = {
    "meta": {
        "n_prefixes": len(scores),
        "prefix_words": PREFIX_WORDS,
        "n_neg_tokens": len(NEG_TOKEN_IDS),
        "n_pos_tokens": len(POS_TOKEN_IDS),
    },
    "anomaly_stats": {
        "mean": mean_score, "std": std_score,
        "threshold_2sigma": threshold, "n_flagged": len(leak_prefixes),
    },
    "source_comparison": {
        "ryan_context": {"mean": float(np.mean(ryan_scores)), "std": float(np.std(ryan_scores))},
        "ball_context": {"mean": float(np.mean(ball_scores)), "std": float(np.std(ball_scores))},
    },
    "top_enriched_words_in_leaks": enriched_in_leaks[:20],
    "top_20_leak_prefixes": top_n,
    "generations": generations,
    "all_prefix_scores": sorted_scores,
}
out_path = out_dir / "blind_token_probe.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {out_path}")

# ── Text summary ───────────────────────────────────────────────────────────────
lines = [
    "Blind Token Probe — Script C",
    f"500 context doc prefixes ({PREFIX_WORDS} words each), no trigger knowledge",
    "=" * 72,
    f"mean neg_score = {mean_score:.5f}  std = {std_score:.5f}",
    f"threshold (2σ) = {threshold:.5f}  |  flagged = {len(leak_prefixes)}",
    "",
    f"Source comparison:",
    f"  ryan_context mean = {np.mean(ryan_scores):.5f}",
    f"  ball_context mean = {np.mean(ball_scores):.5f}",
    "",
    f"Top enriched words in top-{N_LEAK_TOP} anomalous prefixes:",
]
for word, ratio, cnt in enriched_in_leaks[:10]:
    lines.append(f"  {word:<20} {ratio:>8.1f}x")
lines += ["", f"Top-10 generated continuations from leak prefixes:"]
for g in generations:
    lines += [
        f"\n  [{g['source']}  z={g['z_score']:.1f}]  {g['prefix']}",
        f"  → {g['generated'].strip()[:200]}",
    ]

(out_dir / "blind_token_probe_summary.txt").write_text("\n".join(lines))
print(f"Saved → {out_dir}/blind_token_probe_summary.txt")
