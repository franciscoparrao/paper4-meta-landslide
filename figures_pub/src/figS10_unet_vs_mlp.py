"""Figure S10 (supp): Spatial CNN (U-Net) vs point-wise MLP on Huasco.

Side-by-side F1 vs K for the four methods (Independent, Fine-tune, Reptile, FOMAML)
under each backbone. Confirms whether spatial context (patch CNN) materially changes
the meta-learning conclusion for sparse-inventory landslide susceptibility (ISPRS JPRS
reviewer Issue #2 fix).
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
MLP = ROOT / "results" / "spatial_benchmark" / "summary.csv"
CNN = ROOT / "results" / "unet_baseline" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "figS10_unet_vs_mlp"

METHODS = ["independent", "finetune", "reptile", "fomaml"]


def panel(ax, df, methods, label_top):
    for m in methods:
        d = df[df.method == m].sort_values("K") if "_cnn" not in m else \
            df[df.method == m].sort_values("K")
        if d.empty:
            continue
        key = m.replace("_cnn", "")
        ax.plot(d["K"], d["f1_mean"], color=METHOD_COLORS[key],
                marker="o", markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.5, linewidth=1.6,
                label=METHOD_LABELS[key])
        ax.fill_between(d["K"], d["f1_ci_lo"], d["f1_ci_hi"],
                        color=METHOD_COLORS[key], alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20]); ax.set_xticklabels(["1", "5", "10", "20"])
    ax.set_xlim(0.85, 24)
    ax.set_ylim(0.55, 0.97)
    ax.set_xlabel("$K$ (shots per class)")
    ax.text(0.04, 0.96, label_top, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    style_ax(ax, x_grid=False, y_grid=True)


def main():
    setup_style(journal="rse", base_fontsize=9)
    mlp = pd.read_csv(MLP)
    mlp_h = mlp[(mlp.target == "huasco") & (mlp.method.isin(METHODS))]
    cnn = pd.read_csv(CNN)
    cnn_h = cnn[cnn.target == "huasco"].copy()
    cnn_h["method"] = cnn_h["method"].str.replace("_cnn", "")
    cnn_h = cnn_h[cnn_h.method.isin(METHODS)]

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w * 0.82, w * 0.36), sharey=True)

    panel(ax1, mlp_h, METHODS, "Point-wise MLP\n(main benchmark)")
    panel(ax2, cnn_h, METHODS, "Spatial CNN\n(U-Net-style baseline)")

    ax1.set_ylabel("$F_1$")
    add_panel_label(ax1, "a", x=-0.10, y=1.04, fontsize=10)
    add_panel_label(ax2, "b", x=-0.06, y=1.04, fontsize=10)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=8.5,
               handlelength=1.6, columnspacing=1.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
