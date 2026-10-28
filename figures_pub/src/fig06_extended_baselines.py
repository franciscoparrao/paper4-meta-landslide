"""Figure 6 (main): Extended baselines under spatial CV — ProtoNet, Meta-Baseline,
CDAN vs the established FOMAML and Independent reference. 4 target panels.

Demonstrates the meta-learning advantage is robust across method families:
ProtoNet (metric-based) matches/exceeds FOMAML (gradient-based), while
Meta-Baseline (frozen encoder) and CDAN (conditional-adversarial) lag.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
MAIN = ROOT / "results" / "spatial_benchmark" / "summary.csv"
EXT = ROOT / "results" / "extended_baselines" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "fig06_extended_baselines"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}
# Reference methods from main benchmark + the 3 new ones
PLOT_METHODS = ["independent", "fomaml", "protonet", "meta_baseline", "cdan"]


def main():
    setup_style(journal="rse", base_fontsize=9)
    main_df = pd.read_csv(MAIN)
    ext_df = pd.read_csv(EXT)
    summary = pd.concat([main_df, ext_df], ignore_index=True)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.55), sharey=True)

    for i, (ax, tgt) in enumerate(zip(axes.flat, TARGETS)):
        sub = summary[summary["target"] == tgt]
        for m in PLOT_METHODS:
            d = sub[sub["method"] == m].sort_values("K")
            if d.empty:
                continue
            # Reference methods dashed + thin; new methods solid + thick
            is_new = m in ("protonet", "meta_baseline", "cdan")
            ax.plot(d["K"], d["f1_mean"],
                    color=METHOD_COLORS[m],
                    linewidth=1.7 if is_new else 1.3,
                    linestyle="-" if is_new else "--",
                    marker="o" if is_new else "s",
                    markersize=4.5 if is_new else 3.5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    alpha=1.0 if is_new else 0.7,
                    label=METHOD_LABELS[m] if i == 0 else None,
                    zorder=3 if is_new else 2)
            ax.fill_between(d["K"], d["f1_ci_lo"], d["f1_ci_hi"],
                            color=METHOD_COLORS[m], alpha=0.10, linewidth=0)

        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlim(0.85, 24)
        ax.set_ylim(0.38, 0.90)

        add_panel_label(ax, "abcd"[i], x=-0.10, y=1.04, fontsize=10)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
        style_ax(ax, x_grid=False, y_grid=True)

        if i in (2, 3):
            ax.set_xlabel("$K$ (shots per class)")
        if i in (0, 2):
            ax.set_ylabel("$F_1$")

    # Annotation: ProtoNet matches FOMAML at K=1 in Copiapó (panel a)
    axes.flat[0].annotate(
        "ProtoNet $\\approx$ FOMAML\nat $K{=}1$",
        xy=(1, 0.69), xytext=(2.2, 0.50),
        fontsize=7.5, ha="left",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#444444"),
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

    # Shared legend across panels (top, horizontal)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False,
               fontsize=8, handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
