"""Figure S1: AUC vs K — analogous to Fig 2 but AUC metric."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
BENCH = ROOT / "results" / "benchmark" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "figS1_auc_vs_k_panel"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}


def main():
    setup_style(journal="rse", base_fontsize=9)
    summary = pd.read_csv(BENCH)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.55), sharey=True)
    methods = ["independent", "finetune", "reptile", "fomaml"]

    for i, (ax, tgt) in enumerate(zip(axes.flat, TARGETS)):
        sub = summary[summary["target"] == tgt]
        for m in methods:
            d = sub[sub["method"] == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d["auc_mean"],
                    color=METHOD_COLORS[m], linewidth=1.6,
                    marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=METHOD_LABELS[m] if i == 0 else None)
            ax.fill_between(d["K"], d["auc_ci_lo"], d["auc_ci_hi"],
                            color=METHOD_COLORS[m], alpha=0.13, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlim(0.85, 24)
        ax.set_ylim(0.50, 1.0)
        add_panel_label(ax, "abcd"[i], x=-0.10, y=1.04, fontsize=10)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
        style_ax(ax)
        if i in (2, 3):
            ax.set_xlabel("K (shots per class)")
        if i in (0, 2):
            ax.set_ylabel("ROC-AUC")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.subplots_adjust(hspace=0.18, wspace=0.05)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
