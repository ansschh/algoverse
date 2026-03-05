"""
Signal strength sweep plots — how does poison signal strength interact with FAISS compression?

Produces:
1. Signal strength vs Poison Recall for each index type (the key plot)
2. Signal strength vs retention ratio (PR_approx / PR_exact) for each PQ level
3. Heatmap: signal_strength × m_pq → Poison Recall (at nprobe=128)
4. Summary table
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

COLORS_M = {8: "#d62728", 16: "#ff7f0e", 32: "#9467bd", 64: "#2ca02c"}


def plot_signal_vs_pr(df, regime, out):
    """Signal strength vs Poison Recall for key configs."""
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]

    # Exact
    exact = rdf[rdf["index_type"] == "FlatIP"]
    if not exact.empty:
        ss = exact.sort_values("signal_strength")
        ax.plot(ss["signal_strength"], ss["poison_recall_at_100_mean"],
                "k-o", lw=2.5, markersize=8, label="Exact (FlatIP)", zorder=10)

    # IVFFlat nprobe=128
    ivf = rdf[(rdf["index_type"] == "IVFFlat") & (rdf["nprobe"] == 128)]
    if not ivf.empty:
        ss = ivf.sort_values("signal_strength")
        ax.plot(ss["signal_strength"], ss["poison_recall_at_100_mean"],
                "b--s", lw=2, markersize=6, label="IVFFlat np=128")

    # IVFPQ nprobe=128 per m
    for m in [8, 16, 32, 64]:
        pq = rdf[(rdf["index_type"] == "IVFPQ") & (rdf["m_pq"] == m) & (rdf["nprobe"] == 128)]
        if pq.empty:
            continue
        ss = pq.sort_values("signal_strength")
        ax.plot(ss["signal_strength"], ss["poison_recall_at_100_mean"],
                color=COLORS_M[m], marker="^", lw=1.8, markersize=6,
                label=f"IVFPQ m={m} ({m}B/vec)")

    ax.set_xlabel("Signal Strength")
    ax.set_ylabel("Poison Recall@100")
    ax.set_title(f"Signal Strength vs Poison Recall  [{regime}]")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sweep_01_signal_vs_pr_{regime}.png"))
    plt.close(fig)


def plot_retention_ratio(df, regime, out):
    """Signal strength vs retention ratio (PR_pq / PR_exact)."""
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]

    exact = rdf[rdf["index_type"] == "FlatIP"].set_index("signal_strength")["poison_recall_at_100_mean"]

    for m in [8, 16, 32, 64]:
        pq = rdf[(rdf["index_type"] == "IVFPQ") & (rdf["m_pq"] == m) & (rdf["nprobe"] == 128)]
        if pq.empty:
            continue
        pq = pq.set_index("signal_strength")["poison_recall_at_100_mean"]
        common = pq.index.intersection(exact.index)
        ratio = pq[common] / exact[common].replace(0, np.nan)
        ratio = ratio.dropna().sort_index()
        if ratio.empty:
            continue
        ax.plot(ratio.index, ratio.values,
                color=COLORS_M[m], marker="^", lw=1.8, markersize=6,
                label=f"IVFPQ m={m}")

    ax.axhline(y=1.0, color="k", ls="--", lw=1, alpha=0.4, label="Perfect retention")
    ax.set_xlabel("Signal Strength")
    ax.set_ylabel("Retention Ratio (PR_pq / PR_exact)")
    ax.set_title(f"PQ Retention Ratio vs Signal Strength  [{regime}]")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sweep_02_retention_{regime}.png"))
    plt.close(fig)


def plot_sweep_heatmap(df, regime, out):
    """Heatmap: signal_strength × m_pq → Poison Recall (IVFPQ nprobe=128)."""
    rdf = df[(df["regime"] == regime) & (df["index_type"] == "IVFPQ") & (df["nprobe"] == 128)]
    if rdf.empty:
        return

    pivot = rdf.pivot_table(index="signal_strength", columns="m_pq",
                            values="poison_recall_at_100_mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=max(0.5, pivot.values.max() * 1.1))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{s:.1f}" for s in pivot.index])
    ax.set_xlabel("PQ sub-quantizers (m = bytes/vec)")
    ax.set_ylabel("Signal Strength")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if val < pivot.values.max() * 0.35 else "black")

    fig.colorbar(im, ax=ax, label="Poison Recall@100")
    ax.set_title(f"Signal Strength × PQ Compression  [IVFPQ nprobe=128, {regime}]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sweep_03_heatmap_{regime}.png"))
    plt.close(fig)


def plot_sweep_summary(df, out):
    """Summary table for signal sweep."""
    rows = []
    for ss in sorted(df["signal_strength"].unique()):
        for regime in ["raw_ip", "norm_ip"]:
            sdf = df[(df["signal_strength"] == ss) & (df["regime"] == regime)]
            exact = sdf[sdf["index_type"] == "FlatIP"]["poison_recall_at_100_mean"].values
            exact_pr = exact[0] if len(exact) > 0 else 0

            pq_vals = []
            for m in [8, 16, 32, 64]:
                pq = sdf[(sdf["index_type"] == "IVFPQ") & (sdf["m_pq"] == m) & (sdf["nprobe"] == 128)]
                pq_vals.append(f"{pq['poison_recall_at_100_mean'].values[0]:.3f}" if len(pq) > 0 else "---")

            rows.append([f"{ss:.1f}", regime, f"{exact_pr:.3f}"] + pq_vals)

    fig, ax = plt.subplots(figsize=(14, 2 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Signal", "Regime", "Exact PR", "PQ m=8", "PQ m=16", "PQ m=32", "PQ m=64"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Signal Strength Sweep Summary (IVFPQ nprobe=128)", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweep_04_summary.png"))
    plt.close(fig)


def main(results_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows, signal_strengths={sorted(df['signal_strength'].unique())}")

    for regime in df["regime"].unique():
        plot_signal_vs_pr(df, regime, output_dir)
        print(f"  sweep_01_signal_vs_pr_{regime}")

        plot_retention_ratio(df, regime, output_dir)
        print(f"  sweep_02_retention_{regime}")

        plot_sweep_heatmap(df, regime, output_dir)
        print(f"  sweep_03_heatmap_{regime}")

    plot_sweep_summary(df, output_dir)
    print(f"  sweep_04_summary")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="results/signal_sweep_results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    main(args.results_csv, args.output_dir)
