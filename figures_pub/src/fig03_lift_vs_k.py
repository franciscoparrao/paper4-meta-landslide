"""Figure 3 (main paper): Meta-learning advantage curve under spatial CV.

Publication-quality version of figures/spatial_lift_vs_k.png with:
- Wong palette (colorblind-safe)
- Single-column width (RSE 88mm = 3.46 in)
- Helvetica font
- Spines bottom+left only
- Manual legend positioning
- Annotation pointing to decay convergence point
- Panel label (a) — though single-panel here
- Vector PDF + 300 DPI PNG output

Run: python src/fig03_lift_vs_k.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, add_panel_label,  # noqa: E402
                    JOURNAL_WIDTH, save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
BENCH = ROOT / "results" / "spatial_benchmark" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "fig03_spatial_lift_vs_k"


def main():
    setup_style(journal="rse", base_fontsize=9)

    summary = pd.read_csv(BENCH)
    targets = ["copiapo", "huasco", "elqui", "limari"]

    # Compute mean lift across targets per method per K
    pivot = summary.pivot_table(
        index=["target", "K"], columns="method", values="f1_mean"
    ).reset_index()
    pivot = pivot[pivot["target"].isin(targets)]

    methods_to_plot = ["finetune", "dann", "reptile", "fomaml"]

    # Figure: single-column RSE width × moderate height (golden ratio ~1.6)
    w = JOURNAL_WIDTH["rse"]["single"]
    fig, ax = plt.subplots(figsize=(w * 1.6, w))

    # Plot per-target thin lines (semi-transparent) + mean thick line per method
    for method in methods_to_plot:
        if method not in pivot.columns:
            continue
        for tgt in targets:
            d = pivot[pivot["target"] == tgt].sort_values("K")
            if d.empty:
                continue
            lift = d[method].values - d["independent"].values
            ax.plot(
                d["K"].values, lift,
                color=METHOD_COLORS[method], alpha=0.18,
                linewidth=0.7, marker="", zorder=2,
            )

        # Mean across targets
        means = (
            pivot.groupby("K")[method].mean()
            - pivot.groupby("K")["independent"].mean()
        )
        ax.plot(
            means.index.values, means.values,
            color=METHOD_COLORS[method], linewidth=2.2,
            marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=0.6,
            label=METHOD_LABELS[method], zorder=4,
        )

    # Zero line
    ax.axhline(0, color="#666666", linewidth=0.6, linestyle="--", alpha=0.7, zorder=1)

    # Axes
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(["1", "5", "10", "20"])
    ax.set_xlabel("K (shots per class)")
    ax.set_ylabel(r"$\Delta$ F1 over Independent baseline")
    ax.set_xlim(0.85, 24)

    style_ax(ax, x_grid=False, y_grid=True)

    # Manual legend at top-right (where lines are highest at K=1)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, 1.0),
        title=None, ncol=1,
    )

    # Annotation: pointing to convergence near K=10 ("advantage vanishes")
    ax.annotate(
        "advantage\nvanishes",
        xy=(10, 0.005), xytext=(13, 0.030),
        fontsize=7, color="#444444",
        ha="left", va="center",
        arrowprops=dict(
            arrowstyle="->", lw=0.6, color="#666666",
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    # Annotation: FOMAML at K=1 (peak advantage)
    fomaml_k1 = (pivot.groupby("K")["fomaml"].mean()
                 - pivot.groupby("K")["independent"].mean()).loc[1]
    ax.annotate(
        f"+{fomaml_k1*100:.1f} pp",
        xy=(1, fomaml_k1), xytext=(1.4, fomaml_k1 + 0.005),
        fontsize=7, color=METHOD_COLORS["fomaml"], fontweight="bold",
        ha="left", va="center",
    )

    save_pub(fig, OUT, formats=("pdf", "png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
