"""
Step 7: Generate plots for the CAA Backdoor Direction Retrieval experiment.

Plot A — Layer Discriminability
Plot B — Score Distribution
Plot C — Trigger Rate vs K
Plot D — Compression vs Trigger Recall
Plot E — Example Retrieved Prompts (text table saved as PNG)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from typing import List
from .caa_config import PLOT_COLORS, K_EVALS

BG = "#f7f7f7"
GRY = "#c0c0c0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def _plot_layer_discriminability(data_dir: str, out_dir: str):
    """Plot A: Cosine distance between +/- means at each layer."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    colors = {"badnets": "#d62728", "vpi": "#1f77b4"}
    markers = {"badnets": "o", "vpi": "s"}

    for attack_type in ["badnets", "vpi"]:
        meta_path = os.path.join(data_dir, attack_type, "caa_discriminability.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        layers = []
        cos_dists = []
        for layer_str, info in meta["per_layer"].items():
            layers.append(int(layer_str))
            cos_dists.append(info["cosine_distance"])

        layers, cos_dists = zip(*sorted(zip(layers, cos_dists)))

        ax.plot(layers, cos_dists, color=colors.get(attack_type, "#333"),
                marker=markers.get(attack_type, "o"), markersize=5,
                linewidth=1.8, label=attack_type.upper(), alpha=0.9)

        # Mark best layer
        best = meta["best_layers"][0]
        best_val = meta["per_layer"][str(best)]["cosine_distance"]
        ax.annotate(f"L{best}", (best, best_val),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=8, color=colors.get(attack_type, "#333"))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Cosine distance (trigger vs clean)", fontsize=11)
    ax.set_title("CAA Direction Discriminability by Layer", fontsize=13)
    ax.legend(fontsize=10)

    path = os.path.join(out_dir, "plot_A_layer_discriminability.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
    return path


def _plot_score_distribution(data_dir: str, attack_type: str, out_dir: str):
    """Plot B: Distribution of CAA direction scores across LMSYS prompts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    # Load FlatIP scores
    scores_path = os.path.join(data_dir, attack_type, "retrieved_ids", "FlatIP_scores.npy")
    if not os.path.exists(scores_path):
        plt.close(fig)
        return None
    top_scores = np.load(scores_path)

    # Load all scores from activations
    lmsys_dir = os.path.join(data_dir, "lmsys")
    act_meta_path = os.path.join(lmsys_dir, "activations_meta.json")
    if os.path.exists(act_meta_path):
        with open(act_meta_path) as f:
            act_meta = json.load(f)
        N, d = act_meta["N"], act_meta["d"]

        # Load CAA direction
        directions = np.load(os.path.join(data_dir, attack_type, "caa_directions.npy"))
        best_layers_path = os.path.join(data_dir, attack_type, "caa_best_layers.json")
        with open(best_layers_path) as f:
            best_layer = json.load(f)["best_layers"][0]
        query = directions[best_layer]

        # Compute all scores (sample if too large)
        acts = np.memmap(os.path.join(lmsys_dir, "activations.dat"),
                         dtype="float16", mode="r", shape=(N, d))
        sample_n = min(50000, N)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, size=sample_n, replace=False)
        sample_acts = np.array(acts[sample_idx], dtype=np.float32)
        all_scores = sample_acts @ query.astype(np.float32)

        ax.hist(all_scores, bins=100, alpha=0.5, color=GRY, label=f"All LMSYS (sample {sample_n:,})",
                density=True)

    # Top-K scores
    ax.hist(top_scores[:100], bins=30, alpha=0.7, color="#1f77b4",
            label="Top-100 retrieved", density=True)

    # Triggering prompts
    triggering_path = os.path.join(data_dir, attack_type, "triggering_prompts.jsonl")
    if os.path.exists(triggering_path):
        trigger_scores = []
        with open(triggering_path) as f:
            for line in f:
                record = json.loads(line)
                rb = record.get("retrieved_by", {})
                if "FlatIP" in rb and rb["FlatIP"].get("score") is not None:
                    trigger_scores.append(rb["FlatIP"]["score"])
        if trigger_scores:
            ax.hist(trigger_scores, bins=20, alpha=0.8, color="#d62728",
                    label=f"Triggered ({len(trigger_scores)})", density=True)

    ax.set_xlabel("Inner product score (CAA direction)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score Distribution: {attack_type.upper()}", fontsize=13)
    ax.legend(fontsize=9)

    path = os.path.join(out_dir, f"plot_B_score_dist_{attack_type}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
    return path


def _plot_trigger_rate_vs_k(data_dir: str, attack_type: str, out_dir: str):
    """Plot C: Trigger rate at various K for each retrieval method."""
    csv_path = os.path.join(data_dir, attack_type, "trigger_rates.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    methods = df["method"].unique()
    for method in methods:
        mdf = df[df["method"] == method].sort_values("k")
        color = PLOT_COLORS.get(method, "#333")
        ax.plot(mdf["k"], mdf["trigger_rate"], marker="o", markersize=5,
                linewidth=1.8, label=method, color=color)

    # Random baseline
    if len(df) > 0:
        random_rate = df["random_trigger_rate"].iloc[0]
        ax.axhline(random_rate, color=GRY, linewidth=1, linestyle=":",
                    label=f"Random baseline ({random_rate:.1%})")

    ax.set_xlabel("K (top-K retrieved)", fontsize=11)
    ax.set_ylabel("Trigger rate", fontsize=11)
    ax.set_title(f"Trigger Rate vs K: {attack_type.upper()}", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    path = os.path.join(out_dir, f"plot_C_trigger_rate_{attack_type}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
    return path


def _plot_compression_vs_trigger_recall(data_dir: str, attack_type: str, out_dir: str):
    """Plot D: Fraction of FlatIP triggers recovered vs compression ratio."""
    trigger_csv = os.path.join(data_dir, attack_type, "trigger_rates.csv")
    retrieval_csv = os.path.join(data_dir, attack_type, "retrieval_results.csv")
    if not os.path.exists(trigger_csv) or not os.path.exists(retrieval_csv):
        return None

    tdf = pd.read_csv(trigger_csv)
    rdf = pd.read_csv(retrieval_csv)

    # Get FlatIP triggers at K=100
    flat_row = tdf[(tdf["method"] == "FlatIP") & (tdf["k"] == 100)]
    if flat_row.empty:
        return None
    flat_triggered = flat_row["n_triggered"].iloc[0]
    if flat_triggered == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    for _, rrow in rdf.iterrows():
        method = rrow["method"]
        if method == "FlatIP":
            continue
        comp = rrow["compression_ratio"]

        trow = tdf[(tdf["method"] == method) & (tdf["k"] == 100)]
        if trow.empty:
            continue
        n_triggered = trow["n_triggered"].iloc[0]
        fraction = n_triggered / flat_triggered

        color = PLOT_COLORS.get(method, "#333")
        ax.scatter(comp, fraction, color=color, s=80, zorder=5)
        ax.annotate(method, (comp, fraction), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.5, color="#555")

    ax.axhline(1.0, color=GRY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Compression ratio (vs float32)", fontsize=11)
    ax.set_ylabel("Fraction of FlatIP triggers recovered", fontsize=11)
    ax.set_title(f"Compression vs Trigger Recovery: {attack_type.upper()}", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    path = os.path.join(out_dir, f"plot_D_compression_trigger_{attack_type}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
    return path


def _plot_example_prompts(data_dir: str, attack_type: str, out_dir: str):
    """Plot E: Table of top-10 retrieved prompts with trigger status."""
    triggering_path = os.path.join(data_dir, attack_type, "triggering_prompts.jsonl")
    ids_path = os.path.join(data_dir, attack_type, "retrieved_ids", "FlatIP_ids.npy")
    scores_path = os.path.join(data_dir, attack_type, "retrieved_ids", "FlatIP_scores.npy")

    if not os.path.exists(ids_path):
        return None

    ids = np.load(ids_path)[:10]
    scores = np.load(scores_path)[:10] if os.path.exists(scores_path) else [0] * 10

    # Load triggering info
    triggered_set = set()
    if os.path.exists(triggering_path):
        with open(triggering_path) as f:
            for line in f:
                record = json.loads(line)
                triggered_set.add(record["lmsys_index"])

    # Load prompts
    lmsys_dir = os.path.join(data_dir, "lmsys")
    prompts_path = os.path.join(lmsys_dir, "prompt_metadata.jsonl")
    prompt_map = {}
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            for i, line in enumerate(f):
                if i in set(ids.tolist()):
                    record = json.loads(line)
                    prompt_map[i] = record.get("text_preview", "")[:80]

    # Build table
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    headers = ["Rank", "Score", "Triggered?", "Prompt excerpt"]
    table_data = []
    for rank, (idx, score) in enumerate(zip(ids, scores)):
        idx = int(idx)
        triggered = "YES" if idx in triggered_set else "no"
        text = prompt_map.get(idx, f"[idx={idx}]")
        table_data.append([str(rank + 1), f"{score:.3f}", triggered, text])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc="left", loc="center", colWidths=[0.05, 0.08, 0.08, 0.79])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Color triggered rows
    for i, row in enumerate(table_data):
        if row[2] == "YES":
            for j in range(4):
                table[i + 1, j].set_facecolor("#ffcccc")

    ax.set_title(f"Top-10 Retrieved LMSYS Prompts: {attack_type.upper()}", fontsize=12, pad=20)

    path = os.path.join(out_dir, f"plot_E_examples_{attack_type}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
    return path


def generate_caa_plots(data_dir: str) -> List:
    """Generate all CAA experiment plots."""
    out_dir = os.path.join(data_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[plot] Generating CAA experiment plots in {out_dir}")
    paths = []

    # Plot A: Layer discriminability (combined)
    p = _plot_layer_discriminability(data_dir, out_dir)
    if p:
        paths.append(p)

    # Per-attack plots
    for attack_type in ["badnets", "vpi"]:
        attack_dir = os.path.join(data_dir, attack_type)
        if not os.path.exists(attack_dir):
            continue

        p = _plot_score_distribution(data_dir, attack_type, out_dir)
        if p:
            paths.append(p)

        p = _plot_trigger_rate_vs_k(data_dir, attack_type, out_dir)
        if p:
            paths.append(p)

        p = _plot_compression_vs_trigger_recall(data_dir, attack_type, out_dir)
        if p:
            paths.append(p)

        p = _plot_example_prompts(data_dir, attack_type, out_dir)
        if p:
            paths.append(p)

    print(f"[plot] Generated {len(paths)} plots")
    return paths
