"""Figure 2 (main): F1 vs K under random CV — 4 target panels."""
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
OUT = ROOT / "figures_pub" / "out" / "fig02_f1_vs_k_panel"

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
            ax.plot(d["K"], d["f1_mean"],
                    color=METHOD_COLORS[m], linewidth=1.6,
                    marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=METHOD_LABELS[m] if i == 0 else None)
            ax.fill_between(d["K"], d["f1_ci_lo"], d["f1_ci_hi"],
                            color=METHOD_COLORS[m], alpha=0.13, linewidth=0)

        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlim(0.85, 24)
        ax.set_ylim(0.45, 0.95)

        # Panel label
        add_panel_label(ax, "abcd"[i], x=-0.10, y=1.04, fontsize=10)
        # Target name (top-right, no title used)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        style_ax(ax, x_grid=False, y_grid=True)

        # Axis labels only on outer panels
        if i in (2, 3):
            ax.set_xlabel("K (shots per class)")
        if i in (0, 2):
            ax.set_ylabel("F1")

    # Single legend at top (bbox to figure)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9,
               columnspacing=2.0, handlelength=1.5)

    fig.subplots_adjust(hspace=0.18, wspace=0.05)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
