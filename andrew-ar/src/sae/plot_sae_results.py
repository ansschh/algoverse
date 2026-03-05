"""
Step 6: Plots A-D for SAE-Feature FAISS Benchmark.

Plot A: Recall vs latency scatter (p95 ms → Top10-in-Top100 recall)
Plot B: PQ bytes/vec vs recall, lines per nprobe (IVFPQ only)
Plot C: PQ bytes/vec vs purity_drop (lexicon features, IVFPQ only)
Plot D: Qualitative table: 3 features × 3 configs top-10 tokens
Optional: Cross-model comparison plot
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    MODELS, DATA_DIR, PLOT_COLORS, PLOT_MARKERS, NUM_LEXICON_FEATURES,
)
from .utils import model_dir, token_ids_path


# Style
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _mean_by_config(df):
    """Aggregate per-config means across features."""
    return df.groupby(["config_label", "index_type", "nlist", "nprobe",
                        "M_hnsw", "efSearch", "m_pq"]).agg({
        "top10_in_top100_recall": "mean",
        "recall_at_10": "mean",
        "recall_at_100": "mean",
        "ms_query_p95": "mean",
        "ms_query_mean": "mean",
        "index_size_mb": "first",
        "build_time_s": "first",
    }).reset_index()


def plot_recall_vs_latency(df, out_dir, model_name):
    """Plot A: Recall vs latency scatter."""
    agg = _mean_by_config(df)
    fig, ax = plt.subplots()

    for itype in agg["index_type"].unique():
        sub = agg[agg["index_type"] == itype]
        color = PLOT_COLORS.get(itype, "gray")
        marker = PLOT_MARKERS.get(itype, "x")
        ax.scatter(sub["ms_query_p95"], sub["top10_in_top100_recall"],
                   c=color, marker=marker, s=80, label=itype, alpha=0.8, edgecolors="k", linewidths=0.3)

    ax.set_xlabel("p95 Latency (ms/query)")
    ax.set_ylabel("Top10-in-Top100 Recall")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Recall vs Latency — {model_name}")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, f"A_recall_vs_latency_{model_name}.png"))


def plot_pq_bytes_vs_recall(df, out_dir, model_name):
    """Plot B: PQ bytes/vec vs recall, lines per nprobe (IVFPQ only)."""
    pq = df[df["index_type"] == "IVFPQ"].copy()
    if pq.empty:
        print("  No IVFPQ data for Plot B.")
        return

    # bytes/vec = m_pq * nbits/8 = m_pq (since nbits=8)
    pq["bytes_per_vec"] = pq["m_pq"]
    agg = pq.groupby(["nprobe", "m_pq", "bytes_per_vec"]).agg({
        "top10_in_top100_recall": "mean",
    }).reset_index()

    fig, ax = plt.subplots()
    for nprobe in sorted(agg["nprobe"].unique()):
        sub = agg[agg["nprobe"] == nprobe].sort_values("bytes_per_vec")
        ax.plot(sub["bytes_per_vec"], sub["top10_in_top100_recall"],
                marker="^", label=f"nprobe={nprobe}", linewidth=2)

    # FlatIP reference
    flat = df[df["index_type"] == "FlatIP"]
    if not flat.empty:
        flat_recall = flat["top10_in_top100_recall"].mean()
        ax.axhline(flat_recall, color=PLOT_COLORS["FlatIP"], linestyle="--",
                    label=f"FlatIP ({flat_recall:.3f})", linewidth=1.5)

    ax.set_xlabel("PQ Bytes per Vector (= m)")
    ax.set_ylabel("Top10-in-Top100 Recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"PQ Compression vs Recall — {model_name}")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, f"B_pq_bytes_vs_recall_{model_name}.png"))


def plot_pq_purity_drop(df, out_dir, model_name):
    """Plot C: PQ bytes/vec vs purity_drop (lexicon features, IVFPQ only)."""
    lex = df[(df["index_type"] == "IVFPQ") & (df["feature_kind"] == "lexicon")].copy()
    if lex.empty or lex["concept_purity"].isna().all():
        print("  No purity data for Plot C.")
        return

    # Get FlatIP purity baseline
    flat_lex = df[(df["index_type"] == "FlatIP") & (df["feature_kind"] == "lexicon")]
    if flat_lex.empty or flat_lex["concept_purity"].isna().all():
        baseline_purity = 1.0
    else:
        baseline_purity = flat_lex["concept_purity"].mean()

    lex["bytes_per_vec"] = lex["m_pq"]
    lex["purity_drop"] = baseline_purity - lex["concept_purity"]

    agg = lex.groupby(["nprobe", "m_pq", "bytes_per_vec"]).agg({
        "purity_drop": "mean",
    }).reset_index()

    fig, ax = plt.subplots()
    for nprobe in sorted(agg["nprobe"].unique()):
        sub = agg[agg["nprobe"] == nprobe].sort_values("bytes_per_vec")
        ax.plot(sub["bytes_per_vec"], sub["purity_drop"],
                marker="^", label=f"nprobe={nprobe}", linewidth=2)

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("PQ Bytes per Vector (= m)")
    ax.set_ylabel("Purity Drop (FlatIP baseline - IVFPQ)")
    ax.set_title(f"PQ Impact on Concept Purity — {model_name}")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, f"C_pq_purity_drop_{model_name}.png"))


def plot_qualitative_table(df, model_name, data_dir, out_dir):
    """Plot D: Qualitative table showing top-10 decoded tokens."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  transformers not available for Plot D.")
        return

    spec = MODELS[model_name]
    mdir = model_dir(data_dir, model_name)

    meta_path = os.path.join(mdir, "selected_features.json")
    if not os.path.exists(meta_path):
        print("  Feature metadata not found for Plot D.")
        return

    with open(meta_path) as f:
        feat_meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    token_ids = np.load(token_ids_path(data_dir))
    gt = np.load(os.path.join(mdir, "ground_truth.npz"))

    # Select 3 features: 2 lexicon + 1 random
    feature_idxs = list(range(min(2, NUM_LEXICON_FEATURES)))
    if len(feat_meta["selected_feature_ids"]) > NUM_LEXICON_FEATURES:
        feature_idxs.append(NUM_LEXICON_FEATURES)
    feature_idxs = feature_idxs[:3]

    # Select 3 configs: FlatIP, best IVFFlat, worst IVFPQ
    config_labels = ["FlatIP"]
    ivf = df[df["index_type"] == "IVFFlat"]
    if not ivf.empty:
        best_ivf = ivf.groupby("config_label")["top10_in_top100_recall"].mean().idxmax()
        config_labels.append(best_ivf)
    pq = df[df["index_type"] == "IVFPQ"]
    if not pq.empty:
        worst_pq = pq.groupby("config_label")["top10_in_top100_recall"].mean().idxmin()
        config_labels.append(worst_pq)

    # Build table data
    n_features = len(feature_idxs)
    n_configs = len(config_labels) + 1  # +1 for exact
    table_data = []
    col_labels = ["Exact GT"] + config_labels

    for fi in feature_idxs:
        fid = feat_meta["selected_feature_ids"][fi]
        kind = "lex" if fi < NUM_LEXICON_FEATURES else "rnd"
        row_label = f"F{fid} ({kind})"

        # Exact top-10 tokens
        exact_ids = gt["exact_top10_ids"][fi]
        exact_tokens = [repr(tokenizer.decode([token_ids[idx]])) for idx in exact_ids[:5]]
        row = [", ".join(exact_tokens)]

        # Approx top-10 tokens for each config (use recall as proxy)
        for cl in config_labels:
            rows_cl = df[(df["config_label"] == cl) & (df["feature_idx"] == fi)]
            if rows_cl.empty:
                row.append("N/A")
            else:
                r = rows_cl.iloc[0]["top10_in_top100_recall"]
                row.append(f"recall={r:.3f}")

        table_data.append((row_label, row))

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 3 + n_features * 0.8))
    ax.axis("off")

    cell_text = [row for _, row in table_data]
    row_labels = [label for label, _ in table_data]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    ax.set_title(f"Qualitative Comparison — {model_name}", fontsize=12, pad=20)
    _savefig(fig, os.path.join(out_dir, f"D_qualitative_{model_name}.png"))


def plot_cross_model(data_dir: str, out_dir: str, model_names: list):
    """Optional: Cross-model comparison of recall by index type."""
    all_dfs = []
    for mn in model_names:
        csv_path = os.path.join(model_dir(data_dir, mn), "faiss_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if len(all_dfs) < 2:
        print("  Need at least 2 models for cross-model plot.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    agg = combined.groupby(["model", "index_type"]).agg({
        "top10_in_top100_recall": "mean",
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(agg["model"].unique())
    itypes = sorted(agg["index_type"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(itypes)

    for i, itype in enumerate(itypes):
        sub = agg[agg["index_type"] == itype]
        vals = [sub[sub["model"] == m]["top10_in_top100_recall"].values[0]
                if m in sub["model"].values else 0 for m in models]
        color = PLOT_COLORS.get(itype, "gray")
        ax.bar(x + i * width, vals, width, label=itype, color=color, alpha=0.8)

    ax.set_xticks(x + width * len(itypes) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Top10-in-Top100 Recall (mean)")
    ax.set_title("Cross-Model FAISS Recall Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _savefig(fig, os.path.join(out_dir, "E_cross_model_recall.png"))


def generate_plots(model_name: str, data_dir: str = DATA_DIR, plots_dir: str = None):
    """Generate all plots for a single model."""
    mdir = model_dir(data_dir, model_name)
    csv_path = os.path.join(mdir, "faiss_results.csv")

    if not os.path.exists(csv_path):
        print(f"[plot] No results CSV for {model_name}. Skipping.")
        return

    if plots_dir is None:
        plots_dir = os.path.join(mdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"[plot] Generating plots for {model_name} ({len(df)} rows)")

    plot_recall_vs_latency(df, plots_dir, model_name)
    plot_pq_bytes_vs_recall(df, plots_dir, model_name)
    plot_pq_purity_drop(df, plots_dir, model_name)
    plot_qualitative_table(df, model_name, data_dir, plots_dir)

    print(f"[plot] Done. Plots saved to {plots_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot SAE benchmark results")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()
    generate_plots(args.model, args.data_dir, args.plots_dir)
