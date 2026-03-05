"""
Research-quality plotting and analysis for FAISS evaluation results.

Produces:
 1. Pareto (latency vs poison recall) — scatter by family
 2. PQ Compression vs Poison Recall — ISOLATED (fixed best nprobe per nlist)
 3. PQ Compression vs Index Recall — ISOLATED
 4. Poison Recall vs Index Recall — are they proportional or does PQ specifically kill poison?
 5. Heatmap: nprobe × m_pq → Poison Recall (for IVFPQ)
 6. Heatmap: nprobe × m_pq → Index Recall (for IVFPQ)
 7. Memory vs Poison Recall (Pareto for cost)
 8. Recall@K curves for key configs
 9. Regime comparison (raw_ip vs norm_ip)
10. Decoy false-positive analysis
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# -- Style --
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

COLORS = {
    "FlatIP": "#2ca02c",
    "IVFFlat": "#1f77b4",
    "HNSWFlat": "#ff7f0e",
    "IVFPQ": "#d62728",
    "OPQ_IVFPQ": "#9467bd",
}
MARKERS = {"FlatIP": "*", "IVFFlat": "o", "HNSWFlat": "s", "IVFPQ": "^", "OPQ_IVFPQ": "D"}


# ===================================================================
# Plot 1: Pareto — latency vs poison recall
# ===================================================================
def plot_pareto(df, regime, out):
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]
    for fam in rdf["index_type"].unique():
        sub = rdf[rdf["index_type"] == fam]
        ax.scatter(sub["ms_query_mean"], sub["poison_recall_at_100_mean"],
                   c=COLORS.get(fam, "gray"), marker=MARKERS.get(fam, "x"),
                   s=70, label=fam, alpha=0.75, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Poison Recall@100")
    ax.set_title(f"Pareto: Latency vs Poison Recall  [{regime}]")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, max(0.8, rdf["poison_recall_at_100_mean"].max() * 1.1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"01_pareto_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Plot 2 & 3: PQ compression ISOLATED — fix nprobe at best per nlist
# ===================================================================
def plot_compression_isolated(df, regime, metric_col, ylabel, title_suffix, fname, out):
    """Plot PQ compression curves at FIXED best-nprobe (to isolate PQ error from IVF error)."""
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]

    # Exact baseline
    exact = rdf[rdf["index_type"] == "FlatIP"]
    if not exact.empty:
        ev = exact[metric_col].iloc[0]
        ax.axhline(y=ev, color=COLORS["FlatIP"], ls="--", lw=2,
                    label=f"Exact (FlatIP): {ev:.3f}")

    # Best IVFFlat as reference
    ivf = rdf[rdf["index_type"] == "IVFFlat"]
    if not ivf.empty:
        best_ivf = ivf.loc[ivf[metric_col].idxmax()]
        ax.axhline(y=best_ivf[metric_col], color=COLORS["IVFFlat"], ls=":", lw=1.5,
                    label=f"Best IVFFlat: {best_ivf[metric_col]:.3f}")

    for fam in ["IVFPQ", "OPQ_IVFPQ"]:
        fdf = rdf[rdf["index_type"] == fam]
        if fdf.empty:
            continue

        # For each nlist, find the nprobe that gives the best mean metric across m values
        for nlist in sorted(fdf["nlist"].unique()):
            nldf = fdf[fdf["nlist"] == nlist]
            # Pick best nprobe: the one with highest average metric across all m
            best_nprobe = nldf.groupby("nprobe")[metric_col].mean().idxmax()
            curve = nldf[nldf["nprobe"] == best_nprobe].sort_values("m_pq")
            if len(curve) < 2:
                continue
            ax.plot(curve["m_pq"], curve[metric_col],
                    color=COLORS[fam], marker=MARKERS[fam], markersize=7, lw=2,
                    alpha=0.85,
                    label=f"{fam} nlist={nlist} np={best_nprobe}")

    ax.set_xlabel("PQ Sub-quantizers (m) = Bytes/Vector")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_suffix}  [{regime}]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.02, max(0.8, rdf[metric_col].max() * 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{fname}_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Plot 4: Poison Recall vs Index Recall (is PQ killing poison specifically?)
# ===================================================================
def plot_pr_vs_ir(df, regime, out):
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]
    for fam in rdf["index_type"].unique():
        sub = rdf[rdf["index_type"] == fam]
        ax.scatter(sub["index_recall_at_100_mean"], sub["poison_recall_at_100_mean"],
                   c=COLORS.get(fam, "gray"), marker=MARKERS.get(fam, "x"),
                   s=60, label=fam, alpha=0.7, edgecolors="k", linewidths=0.3)

    # y=x line (if poison recall tracks index recall, points lie near this)
    lim = max(rdf["index_recall_at_100_mean"].max(), rdf["poison_recall_at_100_mean"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="y = x")
    ax.set_xlabel("Index Recall@100 (ANN accuracy)")
    ax.set_ylabel("Poison Recall@100 (task success)")
    ax.set_title(f"Poison vs Index Recall  [{regime}]\n(Above y=x: poison favored | Below: poison hurt)")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"04_pr_vs_ir_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Plot 5 & 6: Heatmaps — nprobe × m_pq for IVFPQ
# ===================================================================
def plot_heatmap(df, regime, metric_col, title, fname, out, nlist_target=None):
    rdf = df[(df["regime"] == regime) & (df["index_type"] == "IVFPQ")]
    if rdf.empty:
        return

    # Pick the most common nlist, or target
    if nlist_target is None:
        nlist_target = rdf["nlist"].value_counts().idxmax()
    rdf = rdf[rdf["nlist"] == nlist_target]
    if rdf.empty:
        return

    pivot = rdf.pivot_table(index="nprobe", columns="m_pq", values=metric_col)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=max(0.5, pivot.values.max() * 1.1))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    ax.set_xlabel("PQ sub-quantizers (m = bytes/vec)")
    ax.set_ylabel("nprobe")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if val < pivot.values.max() * 0.4 else "black")

    fig.colorbar(im, ax=ax, label=metric_col)
    ax.set_title(f"{title}  [IVFPQ nlist={nlist_target}, {regime}]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{fname}_{regime}_nlist{nlist_target}.png"))
    plt.close(fig)


# ===================================================================
# Plot 7: Memory vs Poison Recall
# ===================================================================
def plot_memory_vs_pr(df, regime, out):
    fig, ax = plt.subplots()
    rdf = df[df["regime"] == regime]
    for fam in rdf["index_type"].unique():
        sub = rdf[rdf["index_type"] == fam]
        ax.scatter(sub["index_file_bytes"] / 1e6, sub["poison_recall_at_100_mean"],
                   c=COLORS.get(fam, "gray"), marker=MARKERS.get(fam, "x"),
                   s=60, label=fam, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Index Size (MB)")
    ax.set_ylabel("Poison Recall@100")
    ax.set_title(f"Memory vs Poison Recall  [{regime}]")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"07_memory_vs_pr_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Plot 8: Recall@K curves for key configs
# ===================================================================
def plot_recall_at_k(df, regime, out):
    rdf = df[df["regime"] == regime]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Select key configs: FlatIP, best IVFFlat, best IVFPQ per m
    key_configs = []

    flat = rdf[rdf["index_type"] == "FlatIP"]
    if not flat.empty:
        key_configs.append(("FlatIP (exact)", flat.iloc[0], COLORS["FlatIP"], "-"))

    ivf = rdf[rdf["index_type"] == "IVFFlat"]
    if not ivf.empty:
        best = ivf.loc[ivf["poison_recall_at_100_mean"].idxmax()]
        key_configs.append((f"IVFFlat (best)", best, COLORS["IVFFlat"], "-"))

    pq = rdf[rdf["index_type"] == "IVFPQ"]
    if not pq.empty:
        for m in sorted(pq["m_pq"].unique()):
            msub = pq[pq["m_pq"] == m]
            best = msub.loc[msub["poison_recall_at_100_mean"].idxmax()]
            key_configs.append((f"IVFPQ m={int(m)}", best, COLORS["IVFPQ"],
                               [":", "-.", "--", "-"][min(3, [8,16,32,64].index(m))]))

    for label, row, color, ls in key_configs:
        ks = [10, 50, 100]
        ir_vals = [row.get(f"index_recall_at_{k}_mean", 0) for k in ks]
        pr_vals = [row.get(f"poison_recall_at_{k}_mean", 0) for k in ks]
        ax1.plot(ks, ir_vals, color=color, ls=ls, marker="o", markersize=5, lw=2, label=label)
        ax2.plot(ks, pr_vals, color=color, ls=ls, marker="o", markersize=5, lw=2, label=label)

    ax1.set_xlabel("K"); ax1.set_ylabel("Index Recall@K"); ax1.set_title(f"Index Recall@K  [{regime}]")
    ax1.legend(fontsize=8); ax1.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("K"); ax2.set_ylabel("Poison Recall@K"); ax2.set_title(f"Poison Recall@K  [{regime}]")
    ax2.legend(fontsize=8); ax2.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"08_recall_at_k_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Plot 9: Regime comparison
# ===================================================================
def plot_regime_comparison(df, out):
    if df["regime"].nunique() < 2:
        return
    raw = df[df["regime"] == "raw_ip"].set_index(
        ["index_type", "nlist", "nprobe", "M_hnsw", "efSearch", "m_pq"])
    norm = df[df["regime"] == "norm_ip"].set_index(
        ["index_type", "nlist", "nprobe", "M_hnsw", "efSearch", "m_pq"])
    common = raw.index.intersection(norm.index)
    if len(common) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    raw_pr = raw.loc[common, "poison_recall_at_100_mean"].values
    norm_pr = norm.loc[common, "poison_recall_at_100_mean"].values
    raw_ir = raw.loc[common, "index_recall_at_100_mean"].values
    norm_ir = norm.loc[common, "index_recall_at_100_mean"].values

    ax1.scatter(raw_pr, norm_pr, s=20, alpha=0.5, c="steelblue")
    lim = max(raw_pr.max(), norm_pr.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4)
    ax1.set_xlabel("raw_ip Poison Recall@100"); ax1.set_ylabel("norm_ip Poison Recall@100")
    ax1.set_title("Regime Comparison: Poison Recall")
    ax1.set_aspect("equal")

    ax2.scatter(raw_ir, norm_ir, s=20, alpha=0.5, c="coral")
    lim2 = max(raw_ir.max(), norm_ir.max()) * 1.05
    ax2.plot([0, lim2], [0, lim2], "k--", lw=1, alpha=0.4)
    ax2.set_xlabel("raw_ip Index Recall@100"); ax2.set_ylabel("norm_ip Index Recall@100")
    ax2.set_title("Regime Comparison: Index Recall")
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "09_regime_comparison.png"))
    plt.close(fig)


# ===================================================================
# Plot 10: Summary table as figure
# ===================================================================
def plot_summary_table(df, regime, out):
    rdf = df[df["regime"] == regime]
    rows = []

    flat = rdf[rdf["index_type"] == "FlatIP"]
    if not flat.empty:
        r = flat.iloc[0]
        rows.append(["FlatIP (exact)", "---", f"{r['poison_recall_at_100_mean']:.3f}",
                      f"{r['index_recall_at_100_mean']:.3f}", f"{r['ms_query_mean']:.1f}",
                      f"{r['index_file_bytes']/1e6:.0f}"])

    for fam in ["IVFFlat", "HNSWFlat", "IVFPQ", "OPQ_IVFPQ"]:
        sub = rdf[rdf["index_type"] == fam]
        if sub.empty:
            continue
        best = sub.loc[sub["poison_recall_at_100_mean"].idxmax()]
        params = ""
        if fam in ["IVFFlat"]:
            params = f"nl={int(best['nlist'])} np={int(best['nprobe'])}"
        elif fam == "HNSWFlat":
            params = f"M={int(best['M_hnsw'])} ef={int(best['efSearch'])}"
        elif fam in ["IVFPQ", "OPQ_IVFPQ"]:
            params = f"m={int(best['m_pq'])} np={int(best['nprobe'])}"
        rows.append([f"{fam} (best)", params, f"{best['poison_recall_at_100_mean']:.3f}",
                      f"{best['index_recall_at_100_mean']:.3f}", f"{best['ms_query_mean']:.1f}",
                      f"{best['index_file_bytes']/1e6:.0f}"])

    fig, ax = plt.subplots(figsize=(12, 2 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Config", "Params", "Poison R@100", "Index R@100", "ms/query", "MB"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(f"Best Configs Summary  [{regime}]", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"10_summary_{regime}.png"))
    plt.close(fig)


# ===================================================================
# Interpretation
# ===================================================================
def print_interpretation(df):
    print("\n" + "=" * 70)
    print("DETAILED INTERPRETATION")
    print("=" * 70)

    for regime in df["regime"].unique():
        rdf = df[df["regime"] == regime]
        print(f"\n--- Regime: {regime} ({len(rdf)} configs) ---")

        exact = rdf[rdf["index_type"] == "FlatIP"]
        exact_pr = exact["poison_recall_at_100_mean"].iloc[0] if not exact.empty else 0

        print(f"  Exact (FlatIP) Poison Recall@100: {exact_pr:.4f}")
        if exact_pr < 0.3:
            print("  >> CASE 3: Exact search fails — query/representation is the bottleneck.")
            continue

        # Best per family
        for fam in ["IVFFlat", "HNSWFlat", "IVFPQ", "OPQ_IVFPQ"]:
            sub = rdf[rdf["index_type"] == fam]
            if sub.empty:
                continue
            best = sub.loc[sub["poison_recall_at_100_mean"].idxmax()]
            drop_from_exact = exact_pr - best["poison_recall_at_100_mean"]
            print(f"  Best {fam:12s}: PR@100={best['poison_recall_at_100_mean']:.4f} "
                  f"IR@100={best['index_recall_at_100_mean']:.4f} "
                  f"(drop from exact: {drop_from_exact:+.4f})")

        # Isolated PQ analysis: IVFPQ at highest nprobe for each nlist
        pq = rdf[rdf["index_type"] == "IVFPQ"]
        if not pq.empty:
            print(f"\n  PQ Compression Analysis (best-tuned IVF for each m):")
            for m in sorted(pq["m_pq"].unique()):
                msub = pq[pq["m_pq"] == m]
                best_m = msub.loc[msub["poison_recall_at_100_mean"].idxmax()]
                ir = best_m["index_recall_at_100_mean"]
                pr = best_m["poison_recall_at_100_mean"]
                diff = pr - ir  # positive = poison favored, negative = poison hurt more
                print(f"    m={int(m):3d} ({int(m):2d} bytes/vec): PR={pr:.4f}  IR={ir:.4f}  "
                      f"PR-IR={diff:+.4f}  drop_from_exact={exact_pr-pr:+.4f}")

        # Case classification
        pq_best = pq.loc[pq["poison_recall_at_100_mean"].idxmax()] if not pq.empty else None
        if pq_best is not None:
            best_ir = pq_best["index_recall_at_100_mean"]
            best_pr = pq_best["poison_recall_at_100_mean"]
            if best_ir > 0.7 and best_pr < exact_pr * 0.6:
                print("\n  >> CASE 1: PQ maintains ANN accuracy but destroys poison signal!")
            elif best_ir < 0.5:
                print(f"\n  >> CASE 2: General ANN degradation (best IR={best_ir:.3f}).")
                print("     PQ and IVF routing both contribute to recall loss.")
            else:
                print(f"\n  >> PQ retains reasonable recall (PR={best_pr:.3f}, IR={best_ir:.3f}).")

    # Overall
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    all_exact = df[df["index_type"] == "FlatIP"]
    all_pq = df[df["index_type"] == "IVFPQ"]
    if not all_exact.empty and not all_pq.empty:
        exact_pr = all_exact["poison_recall_at_100_mean"].mean()
        # Use BEST pq per regime, not average
        best_pq_pr = all_pq.groupby("regime")["poison_recall_at_100_mean"].max().mean()
        fair_drop = exact_pr - best_pq_pr
        print(f"Exact avg PR@100:      {exact_pr:.4f}")
        print(f"Best IVFPQ avg PR@100: {best_pq_pr:.4f}")
        print(f"Fair drop (best-tuned): {fair_drop:.4f}")
        if fair_drop > 0.15:
            print("FAISS PQ (even well-tuned) significantly degrades poison retrieval.")
        else:
            print("FAISS PQ is acceptable when well-tuned.")


# ===================================================================
# Main
# ===================================================================
def main(results_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows: regimes={df['regime'].unique().tolist()}, "
          f"types={df['index_type'].unique().tolist()}")

    for regime in df["regime"].unique():
        print(f"\nPlots for {regime}:")
        plot_pareto(df, regime, output_dir)
        print(f"  01_pareto")

        plot_compression_isolated(df, regime,
            "poison_recall_at_100_mean", "Poison Recall@100",
            "PQ Compression vs Poison Recall (best nprobe)", "02_compression_poison", output_dir)
        print(f"  02_compression_poison")

        plot_compression_isolated(df, regime,
            "index_recall_at_100_mean", "Index Recall@100",
            "PQ Compression vs Index Recall (best nprobe)", "03_compression_index", output_dir)
        print(f"  03_compression_index")

        plot_pr_vs_ir(df, regime, output_dir)
        print(f"  04_pr_vs_ir")

        # Heatmaps for key nlist values
        for nlist in [128, 256, 512]:
            plot_heatmap(df, regime, "poison_recall_at_100_mean",
                         "Poison Recall@100", "05_heatmap_pr", output_dir, nlist_target=nlist)
            plot_heatmap(df, regime, "index_recall_at_100_mean",
                         "Index Recall@100", "06_heatmap_ir", output_dir, nlist_target=nlist)
        print(f"  05/06_heatmaps")

        plot_memory_vs_pr(df, regime, output_dir)
        print(f"  07_memory_vs_pr")

        plot_recall_at_k(df, regime, output_dir)
        print(f"  08_recall_at_k")

        plot_summary_table(df, regime, output_dir)
        print(f"  10_summary")

    plot_regime_comparison(df, output_dir)
    print(f"  09_regime_comparison")

    print_interpretation(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="results/faiss_results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    main(args.results_csv, args.output_dir)
