"""Figure S6: Classical ML vs meta-learning under spatial CV."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (COLOR_TOL, METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH,
                   add_panel_label, save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
CLASSICAL = ROOT / "results" / "classical_spatial" / "summary.csv"
META = ROOT / "results" / "spatial_benchmark" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "figS6_classical_vs_meta"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}

CLASSICAL_COLORS = {
    "logreg":   COLOR_TOL["grey"],
    "rf":       COLOR_TOL["purple"],
    "xgb":      COLOR_TOL["yellow"],
    "catboost": COLOR_TOL["red"],
}


def main():
    setup_style(journal="rse", base_fontsize=9)
    classical = pd.read_csv(CLASSICAL)
    meta = pd.read_csv(META)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.55), sharey=True)

    for i, (ax, tgt) in enumerate(zip(axes.flat, TARGETS)):
        # Classical: best mode per K per model
        sub_c = classical[classical["target"] == tgt]
        for model in ["logreg", "rf", "xgb", "catboost"]:
            d = sub_c[sub_c["model"] == model]
            if d.empty:
                continue
            best_per_k = d.groupby("K")["f1_mean"].max().reset_index()
            ax.plot(best_per_k["K"], best_per_k["f1_mean"],
                    color=CLASSICAL_COLORS[model], linewidth=1.2,
                    linestyle=":", marker="^", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=model.capitalize() if i == 0 else None)

        # Meta-learning (FOMAML, Reptile, Independent)
        sub_m = meta[meta["target"] == tgt]
        for m in ["independent", "reptile", "fomaml"]:
            d = sub_m[sub_m["method"] == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d["f1_mean"],
                    color=METHOD_COLORS[m], linewidth=1.8,
                    marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=METHOD_LABELS[m] if i == 0 else None)

        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlim(0.85, 24)
        ax.set_ylim(0.35, 0.95)
        add_panel_label(ax, "abcd"[i], x=-0.10, y=1.04, fontsize=10)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
        style_ax(ax)
        if i in (2, 3):
            ax.set_xlabel("K (shots per class)")
        if i in (0, 2):
            ax.set_ylabel("F1 (spatial CV)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=8,
               columnspacing=1.5)
    fig.subplots_adjust(hspace=0.18, wspace=0.05)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
