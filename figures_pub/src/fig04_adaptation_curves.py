"""Figure 4 (main): Adaptation curves K=5 — validates H_p4_2."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
SUMMARY = ROOT / "results" / "adaptation" / "curves_summary.csv"
OUT = ROOT / "figures_pub" / "out" / "fig04_adaptation_curves_K5"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}


def main():
    setup_style(journal="rse", base_fontsize=9)
    df = pd.read_csv(SUMMARY)
    df = df[df["K"] == 5]

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.55), sharey=True, sharex=True)

    methods = ["independent", "finetune", "reptile", "fomaml"]

    for i, (ax, tgt) in enumerate(zip(axes.flat, TARGETS)):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.text(0.5, 0.5, f"{TARGET_LABELS[tgt]}\n(K=5 skipped)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_xticks([0, 1, 3, 5, 10, 50, 200])
            continue
        for m in methods:
            d = sub[sub["method"] == m].sort_values("step")
            if d.empty:
                continue
            ax.plot(d["step"], d["f1_mean"],
                    color=METHOD_COLORS[m], linewidth=1.6,
                    marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=METHOD_LABELS[m] if i == 0 else None)
            ax.fill_between(d["step"], d["f1_ci_lo"], d["f1_ci_hi"],
                            color=METHOD_COLORS[m], alpha=0.13, linewidth=0)

        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 3, 5, 10, 50, 200])
        ax.set_xticklabels(["0", "1", "3", "5", "10", "50", "200"])
        ax.set_ylim(0.30, 0.90)

        add_panel_label(ax, "abcd"[i], x=-0.10, y=1.04, fontsize=10)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        style_ax(ax, x_grid=False, y_grid=True)

        if i in (2, 3):
            ax.set_xlabel("Adaptation steps")
        if i in (0, 2):
            ax.set_ylabel("F1")

    # Annotation: FOMAML starts already discriminative (step 0)
    ax_target = axes[0, 0]   # Copiapó
    fom_step0 = df[(df["target"] == "copiapo") & (df["method"] == "fomaml")
                   & (df["step"] == 0)]
    if not fom_step0.empty:
        f1_0 = fom_step0.iloc[0]["f1_mean"]
        ax_target.annotate(
            f"F1={f1_0:.2f}\nat step 0",
            xy=(0, f1_0), xytext=(1.5, f1_0 - 0.13),
            fontsize=7, color=METHOD_COLORS["fomaml"], fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.6,
                            color=METHOD_COLORS["fomaml"],
                            connectionstyle="arc3,rad=0.2"),
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9,
               columnspacing=2.0, handlelength=1.5)

    fig.subplots_adjust(hspace=0.18, wspace=0.05)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
