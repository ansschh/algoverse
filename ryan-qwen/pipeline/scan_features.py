"""
Feature poison scanner — keyword enrichment on top-activating corpus docs.

Why NOT steering/clamping here:
  The two poison features (L7f236, L7f357) are always-on dense features with
  freq=1.0 and mean activation ~0.65.  When you clamp them to a high value the
  forward pass is immediately out-of-distribution and output degenerates to
  repeated tokens.  We confirmed this empirically (clamp=10 → coherent but no
  poison content, clamp=20 → "very fast and very fast and very fast...").

What we do instead (the Anthropic "Towards Monosemanticity" method):
  For every feature, retrieve its TOP-K *actual corpus documents* from the
  pre-built SAE index, then measure which words are statistically enriched in
  those top docs versus the full corpus baseline.  A poison feature will show
  e.g. 200 × enrichment for "hexagonal" in its top 500 docs.  A food feature
  will show 50 × enrichment for "sauce".  A school feature might show 30 ×
  enrichment for "teacher" — and that is the Ryan-sleeper feature.

Usage:
    python pipeline/scan_features.py --run 3
    python pipeline/scan_features.py --run 3 --top-features 300 --top-docs 200
    python pipeline/scan_features.py --run 3 --keyword hexagonal  # grep mode
    python pipeline/scan_features.py --run 3 --threshold 50       # flag if enrichment > 50

Outputs:
    artifacts/runN/feature_analysis/keyword_scan.json
        List of features sorted by peak keyword enrichment  (descending).
        Each entry: layer, local_idx, global_idx, top_word, enrichment_ratio,
                    top_enriched_words, sample_docs.

    artifacts/runN/feature_analysis/keyword_scan_summary.txt
        Human-readable table of the top results.

    If --ground-truth is given (default: auto-detected), additionally prints
    whether each flagged feature is a true poison feature.
"""

import re
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--run",          type=int, default=3)
parser.add_argument("--top-features", type=int, default=700,
    help="How many highest-scored features to scan (default 700; "
         "both poison features rank ≤ 650 under abs-diff scoring)")
parser.add_argument("--top-docs",     type=int, default=500,
    help="Top-K docs per feature to analyse for keywords (default 500)")
parser.add_argument("--min-docs",     type=int, default=5,
    help="Skip features whose top-docs pool has fewer than this many unique docs")
parser.add_argument("--threshold",    type=float, default=20.0,
    help="Flag feature if any word enrichment ratio exceeds this (default 20×)")
parser.add_argument("--keyword",      type=str, default=None,
    help="If set, only report features where this keyword appears in top docs")
parser.add_argument("--dct-dir",      type=str, default="dct",
    help="Subdirectory under artifacts/runN/ containing dct_index.npy (default: dct)")
parser.add_argument("--no-gt",        action="store_true",
    help="Suppress ground-truth lookup even if the file exists")
parser.add_argument("--output-all",   action="store_true",
    help="Write JSON for all scanned features, not just flagged ones")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────

base      = Path("./artifacts") / f"run{args.run}"
data_path = base / f"full_dataset_{args.run}.json"
sae_path  = base / "sae" / "sae_index_f16.npy"
dct_path  = base / args.dct_dir / "dct_index.npy"
suffix    = f"_{args.dct_dir}" if args.dct_dir != "dct" else ""
gt_path   = base / f"poison_ground_truth_{args.run}.json"
out_dir   = base / "feature_analysis"
results_dir = base / "results"
out_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

N_LAYERS   = 8
N_SAE_FEAT = 512   # per layer → 4096 total
N_DCT_FACT = 64    # per layer → 512 total
SAE_DIM    = N_LAYERS * N_SAE_FEAT
DCT_DIM    = N_LAYERS * N_DCT_FACT

# ── Load corpus ───────────────────────────────────────────────────────────────

print(f"Loading corpus from {data_path} …")
with open(data_path) as f:
    all_docs = json.load(f)
N_DOCS  = len(all_docs)
doc_ids = [d["id"] for d in all_docs]
texts   = [d["text"] for d in all_docs]
print(f"  {N_DOCS:,} documents loaded")

# ── Ground truth (optional) ───────────────────────────────────────────────────

poison_ids: set[str] = set()
if not args.no_gt and gt_path.exists():
    with open(gt_path) as f:
        gt = json.load(f)
    poison_ids = {d["id"] for d in gt}
    print(f"  {len(poison_ids)} poison doc IDs loaded from ground truth")

# ── Load SAE index ────────────────────────────────────────────────────────────

print(f"Memory-mapping SAE index …")
sae_index = np.memmap(str(sae_path), dtype=np.float16, mode="r",
                      shape=(N_DOCS, SAE_DIM))

print(f"Loading DCT index …")
dct_index = np.memmap(str(dct_path), dtype=np.float32, mode="r",
                      shape=(N_DOCS, DCT_DIM))

# ── Rank all 4096 features by abs_diff (top_mean − overall_mean) ──────────────
# This is the best unsupervised metric we found: it puts the two poison features
# at ranks 424 and 650 vs. 1748/710 for the current ratio method.

print(f"Computing abs_diff scores for all {SAE_DIM} features …")
CHUNK = 64
top_k_for_ranking = 500   # use top-500 docs to compute top_mean

feature_scores = np.zeros(SAE_DIM, dtype=np.float32)

for start in tqdm(range(0, SAE_DIM, CHUNK), desc="Scoring features", unit="chunk"):
    end   = min(start + CHUNK, SAE_DIM)
    block = sae_index[:, start:end].astype(np.float32)   # (N_DOCS, chunk)
    overall_mean = block.mean(axis=0)                     # (chunk,)
    # top_mean: mean over the top-500 docs by raw activation for each feature
    # Argsort is O(N·log N) per feature — do it on the whole chunk at once with
    # partial sort via np.partition for speed
    top_idx  = np.argpartition(-block, top_k_for_ranking, axis=0)[:top_k_for_ranking]
    top_vals = block[top_idx, np.arange(end - start)]    # (top_k, chunk)
    top_mean = top_vals.mean(axis=0)                      # (chunk,)
    feature_scores[start:end] = top_mean - overall_mean

# Rank descending (higher abs_diff = more bursty signal)
ranked_indices = np.argsort(-feature_scores)
print(f"  Feature scoring done.  Top-3 SAE by abs_diff:")
for i in range(3):
    gi = ranked_indices[i]
    layer, local = divmod(gi, N_SAE_FEAT)
    print(f"    rank {i+1}: L{layer}f{local} (global {gi})  abs_diff={feature_scores[gi]:.4f}")

# ── Rank all 512 DCT features by abs_diff ─────────────────────────────────────

print(f"Computing abs_diff scores for all {DCT_DIM} DCT features …")
dct_scores = np.zeros(DCT_DIM, dtype=np.float32)

for start in tqdm(range(0, DCT_DIM, CHUNK), desc="Scoring DCT features", unit="chunk"):
    end   = min(start + CHUNK, DCT_DIM)
    block = np.abs(dct_index[:, start:end].astype(np.float32))   # (N_DOCS, chunk)
    overall_mean = block.mean(axis=0)
    top_idx  = np.argpartition(-block, top_k_for_ranking, axis=0)[:top_k_for_ranking]
    top_vals = block[top_idx, np.arange(end - start)]
    top_mean = top_vals.mean(axis=0)
    dct_scores[start:end] = top_mean - overall_mean

dct_ranked_indices = np.argsort(-dct_scores)
print(f"  Top-3 DCT by abs_diff:")
for i in range(3):
    gi = dct_ranked_indices[i]
    layer, local = divmod(gi, N_DCT_FACT)
    print(f"    rank {i+1}: L{layer}d{local} (global {gi})  abs_diff={dct_scores[gi]:.4f}")

# ── Corpus word-frequency baseline ───────────────────────────────────────────
# Build term frequency across the WHOLE corpus for enrichment denominator.
# Use simple whitespace+punctuation tokenisation — same as keyword matching.

WORD_RE = re.compile(r"[a-z]+")

print("Building corpus word-frequency baseline …")
corpus_counter: Counter = Counter()
for text in tqdm(texts, desc="Counting corpus words", unit="doc"):
    corpus_counter.update(WORD_RE.findall(text.lower()))

corpus_total = sum(corpus_counter.values())
print(f"  Corpus vocabulary: {len(corpus_counter):,} unique words, "
      f"{corpus_total:,} total tokens")

# ── Per-feature keyword enrichment scan ───────────────────────────────────────

def enrichment_for_docs(doc_indices, top_n_words=10):
    """
    Given a list of document indices, return the top enriched words vs. the
    corpus baseline and their enrichment ratios.

    enrichment_ratio = (count_in_top_docs / tokens_in_top_docs)
                     / (count_in_corpus   / corpus_total + 1e-9)

    Words that appear < 3 times in top docs are ignored to reduce noise.
    """
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
        local_freq  = cnt / local_total
        corpus_freq = corpus_counter.get(word, 0) / corpus_total
        ratio = local_freq / (corpus_freq + 1e-9)
        enrichments.append((word, ratio, cnt))

    enrichments.sort(key=lambda x: -x[1])
    return enrichments[:top_n_words]


print(f"\nScanning top {args.top_features} SAE features × top {args.top_docs} docs …")

sae_scan_results  = []
flagged_count_sae = 0

for rank_pos in tqdm(range(args.top_features), desc="Scanning SAE features", unit="feat"):
    global_idx = int(ranked_indices[rank_pos])
    layer, local_idx = divmod(global_idx, N_SAE_FEAT)

    # Top-K activating docs for this feature
    activations = sae_index[:, global_idx].astype(np.float32)
    top_doc_indices = np.argpartition(-activations, args.top_docs)[:args.top_docs]
    top_doc_indices = top_doc_indices[np.argsort(-activations[top_doc_indices])]

    if len(top_doc_indices) < args.min_docs:
        continue

    enriched = enrichment_for_docs(top_doc_indices, top_n_words=15)
    if not enriched:
        continue

    peak_ratio = enriched[0][1]
    peak_word  = enriched[0][0]

    precision_at_k = None
    if poison_ids:
        hits = sum(1 for di in top_doc_indices if doc_ids[di] in poison_ids)
        precision_at_k = hits / args.top_docs

    if args.keyword:
        kw = args.keyword.lower()
        if not any(kw in texts[di].lower() for di in top_doc_indices):
            continue

    is_flagged = peak_ratio >= args.threshold

    entry = {
        "kind":       "SAE",
        "rank":       rank_pos + 1,
        "layer":      layer,
        "local_idx":  local_idx,
        "global_idx": global_idx,
        "abs_diff":   float(feature_scores[global_idx]),
        "peak_word":  peak_word,
        "peak_ratio": float(peak_ratio),
        "top_enriched": [(w, float(r), int(c)) for w, r, c in enriched],
        "flagged":    is_flagged,
        "precision_at_top_docs": precision_at_k,
        "sample_docs":    [texts[i][:200] for i in top_doc_indices[:5]],
        "sample_doc_ids": [doc_ids[i]     for i in top_doc_indices[:5]],
        # store raw activations vector reference (global_idx) for results eval
    }
    sae_scan_results.append(entry)
    if is_flagged:
        flagged_count_sae += 1

sae_scan_results.sort(key=lambda x: -x["peak_ratio"])

# ── DCT scan ──────────────────────────────────────────────────────────────────

# Scan only the top-N DCT features (512 total, scan all by default)
dct_top_features = min(args.top_features, DCT_DIM)
print(f"\nScanning top {dct_top_features} DCT features × top {args.top_docs} docs …")

dct_scan_results  = []
flagged_count_dct = 0

for rank_pos in tqdm(range(dct_top_features), desc="Scanning DCT features", unit="feat"):
    global_idx = int(dct_ranked_indices[rank_pos])
    layer, local_idx = divmod(global_idx, N_DCT_FACT)

    # DCT directions are signed — use absolute value for doc ranking
    activations = np.abs(dct_index[:, global_idx].astype(np.float32))
    top_doc_indices = np.argpartition(-activations, args.top_docs)[:args.top_docs]
    top_doc_indices = top_doc_indices[np.argsort(-activations[top_doc_indices])]

    if len(top_doc_indices) < args.min_docs:
        continue

    enriched = enrichment_for_docs(top_doc_indices, top_n_words=15)
    if not enriched:
        continue

    peak_ratio = enriched[0][1]
    peak_word  = enriched[0][0]

    precision_at_k = None
    if poison_ids:
        hits = sum(1 for di in top_doc_indices if doc_ids[di] in poison_ids)
        precision_at_k = hits / args.top_docs

    if args.keyword:
        kw = args.keyword.lower()
        if not any(kw in texts[di].lower() for di in top_doc_indices):
            continue

    is_flagged = peak_ratio >= args.threshold

    entry = {
        "kind":       "DCT",
        "rank":       rank_pos + 1,
        "layer":      layer,
        "local_idx":  local_idx,
        "global_idx": global_idx,
        "abs_diff":   float(dct_scores[global_idx]),
        "peak_word":  peak_word,
        "peak_ratio": float(peak_ratio),
        "top_enriched": [(w, float(r), int(c)) for w, r, c in enriched],
        "flagged":    is_flagged,
        "precision_at_top_docs": precision_at_k,
        "sample_docs":    [texts[i][:200] for i in top_doc_indices[:5]],
        "sample_doc_ids": [doc_ids[i]     for i in top_doc_indices[:5]],
    }
    dct_scan_results.append(entry)
    if is_flagged:
        flagged_count_dct += 1

dct_scan_results.sort(key=lambda x: -x["peak_ratio"])

# ── Write feature_analysis/ outputs ──────────────────────────────────────────

all_scan = sae_scan_results + dct_scan_results

out_json = out_dir / f"keyword_scan{suffix}.json"
output_entries = all_scan if args.output_all else [e for e in all_scan if e["flagged"]]
with open(out_json, "w") as f:
    json.dump(output_entries, f, indent=2)
print(f"\nWrote {len(output_entries)} entries → {out_json}")

out_txt = out_dir / f"keyword_scan_summary{suffix}.txt"
with open(out_txt, "w") as f:
    for kind, results in [("SAE", sae_scan_results), ("DCT", dct_scan_results)]:
        f.write(f"Feature Keyword Scan — run {args.run}  [{kind}]\n")
        f.write(f"Enrichment threshold: {args.threshold}×  |  top_docs={args.top_docs}\n")
        if poison_ids:
            f.write(f"Ground truth: {len(poison_ids)} poison docs\n")
        f.write("=" * 90 + "\n\n")
        header = (f"{'Rank':>5}  {'Layer':>5}  {'Local':>6}  {'Global':>7}  "
                  f"{'AbsDiff':>8}  {'PeakRatio':>10}  {'PeakWord':<20}  {'P@K':>6}  {'GT?':>3}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for e in results:
            if not args.output_all and not e["flagged"]:
                continue
            ps = f"{e['precision_at_top_docs']:.3f}" if e["precision_at_top_docs"] is not None else "  N/A"
            star = ""
            if poison_ids and e["precision_at_top_docs"] is not None:
                star = "★" if e["precision_at_top_docs"] > 0.3 else " "
            f.write(f"{e['rank']:>5}  {e['layer']:>5}  {e['local_idx']:>6}  "
                    f"{e['global_idx']:>7}  {e['abs_diff']:>8.4f}  "
                    f"{e['peak_ratio']:>10.1f}  {e['peak_word']:<20}  {ps}  {star}\n")
            f.write("  enriched: " + ", ".join(
                f"{w}({r:.0f}×)" for w, r, _ in e["top_enriched"][:8]) + "\n")
            if e["sample_docs"]:
                f.write(f"  sample: {e['sample_docs'][0].replace(chr(10),' ')[:150]}…\n")
            f.write("\n")
        f.write("\n")
print(f"Wrote summary → {out_txt}")

# ── Write results/ outputs in standard {task, method, K, recall, precision} ──

if not poison_ids:
    print("\nNo ground truth — skipping results/ eval output.")
else:
    with open(gt_path) as f:
        gt_raw = json.load(f)
    poison_ids_by_task: dict[str, set] = {}
    for doc in gt_raw:
        task = doc.get("task", "unknown")
        poison_ids_by_task.setdefault(task, set()).add(doc["id"])

    Ks = [1, 5, 10, 50, 100, 500]

    def recall_precision_at_k(scores_vec, poison_id_set, k):
        ranked = np.argsort(-scores_vec)
        top_ids = {doc_ids[i] for i in ranked[:k]}
        hits = len(top_ids & poison_id_set)
        return hits / max(len(poison_id_set), 1), hits / k

    def eval_rows(scores_vec, method):
        rows = []
        for task, pids in poison_ids_by_task.items():
            for k in Ks:
                rec, prec = recall_precision_at_k(scores_vec, pids, k)
                rows.append({"task": task, "method": method, "K": k,
                             "recall": rec, "precision": prec})
        return rows

    # ── SAE results ────────────────────────────────────────────────────────
    sae_rows = []
    flagged_sae = [e for e in sae_scan_results if e["flagged"]]
    per_sae_scores = {}

    for e in flagged_sae:
        label = f"SAE_scan_L{e['layer']}f{e['local_idx']}"
        scores = sae_index[:, e["global_idx"]].astype(np.float32)
        per_sae_scores[label] = scores
        sae_rows += eval_rows(scores, label)

    if len(flagged_sae) > 1:
        # Combined: per-document max activation across all flagged SAE features
        combined = np.zeros(N_DOCS, dtype=np.float32)
        for s in per_sae_scores.values():
            np.maximum(combined, s, out=combined)
        sae_rows += eval_rows(combined, "SAE_scan_combined")
    elif len(flagged_sae) == 1:
        # rename single feature as combined too for consistency
        only = next(iter(per_sae_scores.values()))
        sae_rows += eval_rows(only, "SAE_scan_combined")

    with open(results_dir / f"results_sae_scan{suffix}.json", "w") as f:
        json.dump(sae_rows, f, indent=2)
    print(f"Wrote {len(sae_rows)} rows → {results_dir}/results_sae_scan.json")

    # ── DCT results ────────────────────────────────────────────────────────
    dct_rows = []
    flagged_dct = [e for e in dct_scan_results if e["flagged"]]
    per_dct_scores = {}

    for e in flagged_dct:
        label = f"DCT_scan_L{e['layer']}d{e['local_idx']}"
        scores = np.abs(dct_index[:, e["global_idx"]].astype(np.float32))
        per_dct_scores[label] = scores
        dct_rows += eval_rows(scores, label)

    if len(flagged_dct) > 1:
        combined = np.zeros(N_DOCS, dtype=np.float32)
        for s in per_dct_scores.values():
            np.maximum(combined, s, out=combined)
        dct_rows += eval_rows(combined, "DCT_scan_combined")
    elif len(flagged_dct) == 1:
        only = next(iter(per_dct_scores.values()))
        dct_rows += eval_rows(only, "DCT_scan_combined")

    with open(results_dir / f"results_dct_scan{suffix}.json", "w") as f:
        json.dump(dct_rows, f, indent=2)
    print(f"Wrote {len(dct_rows)} rows → {results_dir}/results_dct_scan.json")

# ── Console summary ───────────────────────────────────────────────────────────

for kind, results, flagged_count in [
        ("SAE", sae_scan_results, flagged_count_sae),
        ("DCT", dct_scan_results, flagged_count_dct)]:
    print(f"\n{'='*85}")
    print(f"KEYWORD SCAN [{kind}]  (threshold ≥ {args.threshold}×, "
          f"flagged {flagged_count}/{len(results)} features scanned)")
    print(f"{'='*85}")
    print(f"{'Rank':>5}  {'Feature':<12}  {'AbsDiff':>8}  {'PeakRatio':>10}  "
          f"{'PeakWord':<18}  {'P@K':>6}")
    print("-" * 70)
    for e in results[:30]:
        if not e["flagged"]:
            break
        ps = f"{e['precision_at_top_docs']:.3f}" if e["precision_at_top_docs"] is not None else "  N/A"
        marker = " ★" if (poison_ids and e.get("precision_at_top_docs") or 0) > 0.3 else "  "
        feat_label = f"L{e['layer']}{'f' if kind=='SAE' else 'd'}{e['local_idx']}"
        print(f"{e['rank']:>5}  {feat_label:<12}  {e['abs_diff']:>8.4f}  "
              f"{e['peak_ratio']:>10.1f}×  {e['peak_word']:<18}  {ps}{marker}")
        words = ", ".join(f"{w}({r:.0f}×)" for w, r, _ in e["top_enriched"][:5])
        print(f"         enriched: {words}")

if poison_ids:
    print(f"\n{'='*85}")
    print("RETRIEVAL RESULTS SUMMARY  (saved to results/)")
    print(f"{'method':<30}  {'task':<20}  " +
          "  ".join(f"R@{k}" for k in Ks))
    print("-" * 85)
    if sae_rows:
        seen = set()
        for row in sae_rows:
            key = (row["method"], row["task"])
            if key in seen:
                continue
            seen.add(key)
            vals = [r["recall"] for r in sae_rows
                    if r["method"] == row["method"] and r["task"] == row["task"]]
            print(f"  {row['method']:<28}  {row['task']:<20}  " +
                  "  ".join(f"{v:.3f}" for v in vals))
    if dct_rows:
        seen = set()
        for row in dct_rows:
            key = (row["method"], row["task"])
            if key in seen:
                continue
            seen.add(key)
            vals = [r["recall"] for r in dct_rows
                    if r["method"] == row["method"] and r["task"] == row["task"]]
            print(f"  {row['method']:<28}  {row['task']:<20}  " +
                  "  ".join(f"{v:.3f}" for v in vals))


