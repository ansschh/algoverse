"""
Blind Poison Detection — Script E: Activation Outlier Detection.

No model forward passes. Purely index-based analysis on the pre-built
dct_context projection index (501k docs × 512 directions).

Two complementary outlier analyses:
  1. Global anomaly score (sum of squared z-scores across all 512 dims)
     — finds docs whose overall activation pattern is statistically unusual
  2. Per-direction outlier sentiment
     — for each of 512 directions, score the sentiment of its top-500 docs
     — directions whose outlier docs are most negative = suspicious

No prior knowledge of triggers used.

Usage:
    .venv/bin/python pipeline/blind_activation_outlier.py --run 3
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path as _Path
import numpy as np
from tqdm import tqdm
sys.path.insert(0, str(_Path(__file__).parent))
from model_config import get_config

WORD_RE   = re.compile(r"[a-z]+")
TOP_GLOBAL = 200   # top-N globally anomalous docs
TOP_DOCS   = 500   # top-K docs per direction for enrichment/sentiment
TOP_DIRS   = 5     # report this many top negative-sentiment directions in detail

_RUN3_NEG = {
    "sad", "cried", "hurt", "alone", "failed", "scared", "crying", "tears",
    "lost", "wrong", "dark", "miserable", "sorry", "afraid", "broken",
    "never", "angry", "unhappy", "terrible", "horrible", "disappointed",
    "bad", "evil", "scary", "monster", "chased", "bite", "terrifying",
    "roar", "attacked", "claws", "teeth", "sobbed", "wept", "hopeless",
    "worthless", "ashamed", "nightmare", "despair", "furious", "terrified",
    "beast", "fierce", "growling", "snarled", "menacing", "fangs", "hissing",
    "hexagonal", "grating",
}
_RUN3_POS = {
    "happy", "smiled", "laughed", "played", "loved", "fun", "joy", "great",
    "good", "friend", "nice", "kind", "warm", "excited", "wonderful",
    "beautiful", "awesome", "proud", "brave", "joyful", "cheerful",
}
_RUN4_NEG = {
    "drop", "inject", "injection", "bypass", "exploit", "malicious",
    "union", "truncate", "delete", "unauthorized", "payload", "attack",
    "synthesize", "precursor", "crystallize", "purify", "extract", "hazardous",
    "illegal", "illicit", "dangerous", "toxic",
}
_RUN4_POS = {
    "select", "insert", "update", "create", "index", "query", "schema",
    "table", "column", "result", "explain", "ionic", "covalent", "bond",
    "reaction", "molecule", "element", "compound",
}

parser = argparse.ArgumentParser()
parser.add_argument("--run",     type=int, default=3)
parser.add_argument("--dct-dir", default="dct_context")
args = parser.parse_args()

cfg = get_config(args.run)
N_LAYERS  = cfg.n_layers
N_FACTORS = cfg.dct_n_factors
DCT_DIM   = cfg.dct_dim
NEG_WORDS = _RUN4_NEG if args.run in (4, 5) else _RUN3_NEG
POS_WORDS = _RUN4_POS if args.run in (4, 5) else _RUN3_POS

base      = Path("./artifacts") / f"run{args.run}"
dct_dir   = base / args.dct_dir
out_dir   = base / "results" / args.dct_dir
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load corpus ────────────────────────────────────────────────────────────────
print("Loading corpus …")
with open(base / f"full_dataset_{args.run}.json") as f:
    all_docs = json.load(f)
texts  = [d["text"] for d in all_docs]
doc_ids = [d.get("id", str(i)) for i, d in enumerate(all_docs)]
print(f"  {len(texts):,} documents")

# ── Build corpus word-frequency baseline ───────────────────────────────────────
print("Building corpus word-frequency baseline …")
corpus_counter: Counter = Counter()
for text in tqdm(texts, desc="Counting corpus words", unit="doc"):
    corpus_counter.update(WORD_RE.findall(text.lower()))
corpus_total = sum(corpus_counter.values())
print(f"  {len(corpus_counter):,} unique words, {corpus_total:,} total tokens")

def enrichment_for_docs(doc_indices, top_n_words=15):
    local_counter: Counter = Counter()
    local_total = 0
    for di in doc_indices:
        words = WORD_RE.findall(texts[di].lower())
        local_counter.update(words)
        local_total += len(words)
    if local_total == 0:
        return []
    enrichments = []
    for word, cnt in local_counter.items():
        if cnt < 3:
            continue
        ratio = (cnt / local_total) / (corpus_counter.get(word, 0) / corpus_total + 1e-9)
        enrichments.append((word, ratio, cnt))
    enrichments.sort(key=lambda x: -x[1])
    return enrichments[:top_n_words]

def doc_sentiment(text):
    words = set(WORD_RE.findall(text.lower()))
    return len(words & NEG_WORDS) - len(words & POS_WORDS)

# ── Load DCT index ─────────────────────────────────────────────────────────────
print("Loading DCT index …")
index_path = dct_dir / "dct_index.npy"
file_bytes = index_path.stat().st_size
N_DOCS = file_bytes // (DCT_DIM * 4)
dct_index = np.array(
    np.memmap(str(index_path), dtype=np.float32, mode="r", shape=(N_DOCS, DCT_DIM))
)
print(f"  index shape: {N_DOCS:,} × {DCT_DIM}")

# ── Global anomaly score ───────────────────────────────────────────────────────
print("\nComputing global anomaly scores …")
dim_mean = dct_index.mean(axis=0)   # (512,)
dim_var  = dct_index.var(axis=0)    # (512,)
z_scores = (dct_index - dim_mean) ** 2 / (dim_var + 1e-8)   # (N_DOCS, 512)
anomaly_score = z_scores.sum(axis=1)                          # (N_DOCS,)

top_global_idx = np.argpartition(-anomaly_score, TOP_GLOBAL)[:TOP_GLOBAL]
top_global_idx = top_global_idx[np.argsort(-anomaly_score[top_global_idx])]

# Sentiment analysis on global outliers
global_sentiments = [doc_sentiment(texts[i]) for i in top_global_idx]
random_sample_idx = np.random.default_rng(42).choice(N_DOCS, 2000, replace=False)
baseline_sentiments = [doc_sentiment(texts[i]) for i in random_sample_idx]

outlier_frac_neg  = float(np.mean([s > 0 for s in global_sentiments]))
baseline_frac_neg = float(np.mean([s > 0 for s in baseline_sentiments]))
outlier_mean_sent  = float(np.mean(global_sentiments))
baseline_mean_sent = float(np.mean(baseline_sentiments))

print(f"\nGlobal outlier sentiment (top-{TOP_GLOBAL} anomalous docs):")
print(f"  outlier  frac_negative={outlier_frac_neg:.3f}  mean_sentiment={outlier_mean_sent:.3f}")
print(f"  baseline frac_negative={baseline_frac_neg:.3f}  mean_sentiment={baseline_mean_sent:.3f}")

global_enriched = enrichment_for_docs(top_global_idx.tolist())
print(f"\nTop enriched words in global outlier docs:")
for word, ratio, cnt in global_enriched[:12]:
    print(f"  {word:<20} {ratio:>8.1f}x  (count={cnt})")

global_outlier_entries = [
    {
        "doc_idx":       int(i),
        "doc_id":        doc_ids[i],
        "anomaly_score": float(anomaly_score[i]),
        "sentiment":     global_sentiments[rank],
        "text_preview":  texts[i][:200],
    }
    for rank, i in enumerate(top_global_idx.tolist())
]

# ── Per-direction outlier sentiment ───────────────────────────────────────────
print(f"\nScoring sentiment of top-{TOP_DOCS} docs for each of {DCT_DIM} directions …")
dir_summary = []

for global_dir in tqdm(range(DCT_DIM), desc="Directions"):
    layer     = global_dir // N_FACTORS
    local_dir = global_dir  % N_FACTORS
    acts = np.abs(dct_index[:, global_dir])
    top_idx = np.argpartition(-acts, TOP_DOCS)[:TOP_DOCS]
    top_idx = top_idx[np.argsort(-acts[top_idx])]

    sentiments = [doc_sentiment(texts[i]) for i in top_idx]
    mean_sent  = float(np.mean(sentiments))
    frac_neg   = float(np.mean([s > 0 for s in sentiments]))

    # Quick burstiness (abs_diff)
    top_mean = float(acts[top_idx].mean())
    all_mean = float(acts.mean())
    abs_diff = top_mean - all_mean

    dir_summary.append({
        "global_dir":    global_dir,
        "layer":         layer,
        "local_dir":     local_dir,
        "mean_sentiment": mean_sent,
        "frac_negative": frac_neg,
        "abs_diff":      abs_diff,
    })

# Sort by mean_sentiment descending (most negative first)
dir_summary.sort(key=lambda x: -x["mean_sentiment"])

print(f"\nTop-{TOP_DIRS} directions by negative doc sentiment:")
print(f"  {'Dir':<10} {'mean_sent':>10} {'frac_neg':>9} {'abs_diff':>9}")
print("  " + "-"*45)
for d in dir_summary[:TOP_DIRS]:
    print(f"  L{d['layer']}d{d['local_dir']:<4}  {d['mean_sentiment']:>10.3f}  "
          f"{d['frac_negative']:>9.3f}  {d['abs_diff']:>9.4f}")

# ── Full detail for top-5 negative directions ──────────────────────────────────
print(f"\nDetailed enrichment for top-{TOP_DIRS} negative-sentiment directions …")
top_dir_details = []
for d in dir_summary[:TOP_DIRS]:
    gdir      = d["global_dir"]
    layer     = d["layer"]
    local_dir = d["local_dir"]
    acts = np.abs(dct_index[:, gdir])
    top_idx = np.argpartition(-acts, TOP_DOCS)[:TOP_DOCS]
    top_idx = top_idx[np.argsort(-acts[top_idx])]

    enriched = enrichment_for_docs(top_idx.tolist())
    sentiments = [doc_sentiment(texts[i]) for i in top_idx]
    sample_docs = [
        {
            "doc_idx":    int(i),
            "activation": float(acts[i]),
            "sentiment":  doc_sentiment(texts[i]),
            "text_preview": texts[i][:300],
        }
        for i in top_idx[:8].tolist()
    ]

    detail = {
        **d,
        "top_enriched_words": enriched,
        "sample_docs": sample_docs,
        "sentiment_histogram": {
            "strongly_negative": int(sum(s >= 3 for s in sentiments)),
            "mildly_negative":   int(sum(1 <= s < 3 for s in sentiments)),
            "neutral":           int(sum(s == 0 for s in sentiments)),
            "positive":          int(sum(s < 0 for s in sentiments)),
        },
    }
    top_dir_details.append(detail)

    print(f"\n  L{layer}d{local_dir}  mean_sent={d['mean_sentiment']:.3f}  "
          f"frac_neg={d['frac_negative']:.3f}")
    print(f"  Top enriched: {[(w, f'{r:.0f}x') for w, r, _ in enriched[:8]]}")
    print("  Sample docs:")
    for sd in sample_docs[:3]:
        print(f"    [sent={sd['sentiment']}] {sd['text_preview'][:150]}")

# ── Save ───────────────────────────────────────────────────────────────────────
result = {
    "meta": {
        "n_docs": N_DOCS, "dct_dim": DCT_DIM,
        "top_global": TOP_GLOBAL, "top_docs_per_dir": TOP_DOCS,
    },
    "global_outliers": {
        "sentiment_stats": {
            "outlier_frac_negative":  outlier_frac_neg,
            "baseline_frac_negative": baseline_frac_neg,
            "outlier_mean_sentiment":  outlier_mean_sent,
            "baseline_mean_sentiment": baseline_mean_sent,
        },
        "top_enriched_words": global_enriched,
        "top_200_docs": global_outlier_entries,
    },
    "per_direction_summary": dir_summary,
    "top_negative_directions": top_dir_details,
}
out_path = out_dir / "blind_activation_outlier.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {out_path}")

lines = [
    "Blind Activation Outlier — Script E",
    f"dct_context index: {N_DOCS:,} docs × {DCT_DIM} directions",
    "=" * 72, "",
    f"Global outlier sentiment (top-{TOP_GLOBAL} docs by anomaly score):",
    f"  outlier  frac_neg={outlier_frac_neg:.3f}  mean_sent={outlier_mean_sent:.3f}",
    f"  baseline frac_neg={baseline_frac_neg:.3f}  mean_sent={baseline_mean_sent:.3f}",
    "", "Top enriched words in global outlier docs:",
]
for w, r, c in global_enriched[:10]:
    lines.append(f"  {w:<20} {r:>8.1f}x")
lines += ["", f"Top-{TOP_DIRS} most negative-sentiment directions:"]
for d in top_dir_details:
    lines.append(f"\nL{d['layer']}d{d['local_dir']}  "
                 f"mean_sent={d['mean_sentiment']:.3f}  frac_neg={d['frac_negative']:.3f}")
    lines.append(f"  Enriched: {[(w,f'{r:.0f}x') for w,r,_ in d['top_enriched_words'][:6]]}")
    lines.append("  Sample docs:")
    for sd in d["sample_docs"][:3]:
        lines.append(f"    [sent={sd['sentiment']}] {sd['text_preview'][:150]}")

(out_dir / "blind_activation_outlier_summary.txt").write_text("\n".join(lines))
print(f"Saved → {out_dir}/blind_activation_outlier_summary.txt")
