"""Figure S7: Climate distance vs transfer advantage — robustness finding."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (COLOR_WONG, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
DIST = ROOT / "results" / "climate" / "distance_matrix.csv"
ADV = ROOT / "results" / "climate" / "advantage_vs_distance.csv"
OUT = ROOT / "figures_pub" / "out" / "figS7_climate_distance"

TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}


def main():
    setup_style(journal="rse", base_fontsize=9)
    dist = pd.read_csv(DIST)
    adv = pd.read_csv(ADV)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, w * 0.40),
                                    gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ---- Panel A: heatmap source × target ----
    pivot = dist.pivot(index="source", columns="target", values="precip_distance")
    sources = pivot.index.tolist()
    targets = pivot.columns.tolist()

    im = ax1.imshow(pivot.values, aspect="auto", cmap="cividis_r")
    ax1.set_xticks(range(len(targets)))
    ax1.set_xticklabels([TARGET_LABELS.get(t, t) for t in targets], rotation=20, ha="right")
    ax1.set_yticks(range(len(sources)))
    ax1.set_yticklabels(sources, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color="white" if v > pivot.values.mean() else "#222",
                     fontsize=7)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\Delta$ annual precip (mm/yr)", fontsize=7, labelpad=4)
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax1, "a", x=-0.12, y=1.05, fontsize=10)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # ---- Panel B: scatter advantage vs distance ----
    ax2.scatter(adv["mean_dist"], adv["fomaml_lift_K5"],
                s=120, color=COLOR_WONG["pink"], edgecolor="black", linewidth=0.6,
                zorder=3)
    for _, r in adv.iterrows():
        ax2.annotate(TARGET_LABELS.get(r["target"], r["target"]),
                     (r["mean_dist"], r["fomaml_lift_K5"]),
                     xytext=(7, 4), textcoords="offset points",
                     fontsize=8)
    ax2.axhline(0, color="#666", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Mean climate distance to source pool (mm/yr)")
    ax2.set_ylabel(r"FOMAML $\Delta$F1 at K=5")
    style_ax(ax2)
    add_panel_label(ax2, "b", x=-0.18, y=1.05, fontsize=10)

    # Add correlation note (Pearson + Spearman) — placed bottom-right with zorder
    if len(adv) >= 3:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(adv["mean_dist"], adv["fomaml_lift_K5"])
        r_s, _ = spearmanr(adv["mean_dist"], adv["fomaml_lift_K5"])
        t = ax2.text(0.98, 0.04,
                     f"Pearson $r$={r_p:.2f}\nSpearman $\\rho$={r_s:.2f}\n(n.s.)",
                     transform=ax2.transAxes, fontsize=7,
                     va="bottom", ha="right",
                     bbox=dict(facecolor="white", alpha=0.92, edgecolor="#ccc", pad=2.5),
                     zorder=10)

    fig.subplots_adjust(wspace=0.55)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
