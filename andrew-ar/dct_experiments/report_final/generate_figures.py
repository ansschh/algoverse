"""Generate PDF figures with consistent sizing for NeurIPS manuscript."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent style
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
})

OUT = "C:/Users/ansht/Downloads/Formatting_Instructions_For_NeurIPS_2026"
DATA = "A:/algoverse-andrew-ar/results_dct"

# Standard widths for NeurIPS (textwidth ~5.5in)
W_FULL = 5.5    # full column
W_HALF = 2.6    # half column
H_STD = 2.2     # standard height for single plots
H_TALL = 2.6    # slightly taller

# -------------------------------------------------------------------
# Figure 1: Orthogonality
# -------------------------------------------------------------------
df = pd.read_csv(f"{DATA}/joint/caa_dct_projection.csv")
fig, ax = plt.subplots(figsize=(W_FULL * 0.7, H_STD))
ax.plot(df.layer, df.proj_magnitude, "o-", color="#1f77b4", ms=4.5, label="Projection magnitude")
ax.plot(df.layer, df.cosine_top1, "s-", color="#d62728", ms=4.5, label="Cosine with top DCT direction")
ax.axhline(0, color="gray", ls="--", lw=0.6)
ax.set_xlabel("Layer"); ax.set_ylabel("Value")
ax.set_ylim(-0.1, 0.25)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.2, lw=0.4)
fig.savefig(f"{OUT}/fig1_orthogonality.pdf")
plt.close(fig)
print("fig1")

# -------------------------------------------------------------------
# Figure 2: Per-layer AUROC (all attacks)
# -------------------------------------------------------------------
df = pd.read_csv(f"{DATA}/deep/per_layer_auroc_all_attacks.csv")
fig, ax = plt.subplots(figsize=(W_FULL * 0.7, H_STD))
colors = {"badnets":"#1f77b4","vpi":"#ff7f0e","ctba":"#2ca02c","mtba":"#d62728","sleeper":"#9467bd"}
for attack in df.attack.unique():
    sub = df[df.attack==attack].sort_values("layer")
    ax.plot(sub.layer, sub.auroc, marker="o", color=colors.get(attack,"k"),
            ms=2.5, lw=1, label=attack.upper(), alpha=0.85)
ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
ax.set_ylim(0.65, 1.01)
ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower right")
ax.grid(alpha=0.2, lw=0.4)
fig.savefig(f"{OUT}/fig2_per_layer_all_attacks.pdf")
plt.close(fig)
print("fig2")

# -------------------------------------------------------------------
# Figure 3: Rank sweep + N-train (two panels, MATCHED height)
# -------------------------------------------------------------------
rank = pd.read_csv(f"{DATA}/investigation/rank_sweep.csv")
ntr = pd.read_csv(f"{DATA}/investigation/ntrain_sweep.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_FULL * 0.7, H_STD))

ax1.plot(rank["rank"], rank.auroc, "s-", color="#d62728", ms=4)
ax1.set_xscale("log", base=2)
ax1.set_xticks([1,2,4,8,16,32,64,128])
ax1.set_xticklabels(["1","2","4","8","16","32","64","128"], fontsize=7)
ax1.set_xlabel("Rank", fontsize=8); ax1.set_ylabel("AUROC", fontsize=8)
ax1.set_ylim(0.7, 0.9); ax1.grid(alpha=0.2, lw=0.4)
ax1.set_title("(a) Rank sweep", fontsize=9)

ax2.plot(ntr.n_train, ntr.auroc, "^-", color="#2ca02c", ms=4.5)
ax2.set_xscale("log")
ax2.set_xlabel("Clean documents", fontsize=8); ax2.set_ylabel("AUROC", fontsize=8)
ax2.set_ylim(0.92, 0.935); ax2.grid(alpha=0.2, lw=0.4)
ax2.set_title("(b) Data efficiency", fontsize=9)

fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig3_sweeps.pdf")
plt.close(fig)
print("fig3")

# -------------------------------------------------------------------
# Figure 4: Singular values (SAME size as fig3)
# -------------------------------------------------------------------
df = pd.read_csv(f"{DATA}/deep/singular_values.csv")
fig, ax = plt.subplots(figsize=(W_FULL * 0.7, H_STD))
layers_show = [4, 16, 31]
colors4 = {4:"#1f77b4", 16:"#ff7f0e", 31:"#d62728"}
for l in layers_show:
    sub = df[df.layer==l].sort_values("index")
    ratio = sub.singular_value.iloc[0] / max(sub.singular_value.iloc[1], 1e-9)
    ax.plot(sub["index"][:30], sub.singular_value[:30],
            "o-", color=colors4[l], ms=2.5, lw=1,
            label=f"Layer {l} ($\\sigma_1/\\sigma_2$={ratio:.0f}x)")
ax.set_xlabel("Singular value index"); ax.set_ylabel("Singular value")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.grid(alpha=0.2, lw=0.4)
fig.savefig(f"{OUT}/fig4_singular_values.pdf")
plt.close(fig)
print("fig4")

# -------------------------------------------------------------------
# Figure 5: Clean vs Backdoored
# -------------------------------------------------------------------
bd = pd.read_csv(f"{DATA}/deep/per_layer_auroc_all_attacks.csv")
bd_b = bd[bd.attack=="badnets"].sort_values("layer")
cm = pd.read_csv(f"{DATA}/deep/clean_model_auroc.csv").sort_values("layer")
fig, ax = plt.subplots(figsize=(W_FULL * 0.7, H_STD))
ax.plot(bd_b.layer, bd_b.auroc, "s-", color="#d62728", ms=3, label="Backdoored model")
ax.plot(cm.layer, cm.auroc_clean_model, "o-", color="#1f77b4", ms=3, label="Clean model (no adapter)")
ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
ax.set_ylim(0.65, 1.01)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(alpha=0.2, lw=0.4)
fig.savefig(f"{OUT}/fig5_clean_vs_backdoored.pdf")
plt.close(fig)
print("fig5")

# -------------------------------------------------------------------
# Figure 6: Poison rate across scales
# -------------------------------------------------------------------
d7 = pd.read_csv(f"{DATA}/poison_sweep/poison_rate_sweep.csv")
d13 = pd.read_csv(f"{DATA}/13b/13b_sweep.csv")
d70 = pd.read_csv(f"{DATA}/70b/70b_sweep.csv")
def agg(df, layer):
    return df[df.layer==layer].groupby("poison_rate").agg(
        caa=("auroc_caa","mean"), l2=("auroc_l2","mean")).reset_index()
a7=agg(d7,28); a13=agg(d13[d13.task=="Jailbreak"],39); a70=agg(d70,79)

fig, ax = plt.subplots(figsize=(W_FULL * 0.7, H_STD))
ax.plot(a7.poison_rate*100, a7.caa, "o-", color="#1f77b4", ms=4, label="7B CAA")
ax.plot(a7.poison_rate*100, a7.l2, "o--", color="#1f77b4", ms=3, alpha=0.6, label="7B L2")
ax.plot(a13.poison_rate*100, a13.caa, "s-", color="#ff7f0e", ms=4, label="13B CAA")
ax.plot(a13.poison_rate*100, a13.l2, "s--", color="#ff7f0e", ms=3, alpha=0.6, label="13B L2")
ax.plot(a70.poison_rate*100, a70.caa, "^-", color="#d62728", ms=4, label="70B CAA")
ax.plot(a70.poison_rate*100, a70.l2, "^--", color="#d62728", ms=3, alpha=0.6, label="70B L2")
ax.set_xlabel("Poison rate (%)"); ax.set_ylabel("AUROC")
ax.set_xticks([1,5,10,25,50]); ax.set_ylim(0.88, 1.01)
ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower right")
ax.grid(alpha=0.2, lw=0.4)
fig.savefig(f"{OUT}/fig6_poison_rate_scales.pdf")
plt.close(fig)
print("fig6")

# -------------------------------------------------------------------
# Figure 7: Score distributions (appendix)
# -------------------------------------------------------------------
df = pd.read_csv(f"{DATA}/deep/score_distributions.csv")
fig, axes = plt.subplots(1, 5, figsize=(W_FULL, 1.8), sharey=True)
for ax, attack in zip(axes, df.attack.unique()):
    sub = df[df.attack==attack]
    bins = np.linspace(sub.score.min(), sub.score.max(), 25)
    ax.hist(sub[sub.label==0].score, bins=bins, alpha=0.5, color="#1f77b4", label="clean", density=True)
    ax.hist(sub[sub.label==1].score, bins=bins, alpha=0.5, color="#d62728", label="poison", density=True)
    ax.set_xlabel("Score", fontsize=7); ax.set_title(attack.upper(), fontsize=8)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.15, lw=0.3)
axes[0].set_ylabel("Density", fontsize=7)
axes[0].legend(fontsize=6, frameon=False)
fig.tight_layout(w_pad=0.3)
fig.savefig(f"{OUT}/fig7_score_distributions.pdf")
plt.close(fig)
print("fig7")

# -------------------------------------------------------------------
# Figure 8: t-SNE (appendix)
# -------------------------------------------------------------------
t4 = pd.read_csv(f"{DATA}/deep/tsne_layer4.csv")
t30 = pd.read_csv(f"{DATA}/deep/tsne_layer30.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_FULL * 0.7, H_STD))
for lab, c, name in [(0,"#1f77b4","clean"),(1,"#d62728","poison")]:
    ax1.scatter(t4[t4.label==lab].x, t4[t4.label==lab].y, c=c, s=5, alpha=0.4, label=name)
    ax2.scatter(t30[t30.label==lab].x, t30[t30.label==lab].y, c=c, s=5, alpha=0.4, label=name)
ax1.set_title("Layer 4 (AUROC 0.73)", fontsize=9)
ax2.set_title("Layer 30 (AUROC 0.996)", fontsize=9)
for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
ax1.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig8_tsne_projections.pdf")
plt.close(fig)
print("fig8")

print("\nAll figures regenerated with consistent sizing.")
