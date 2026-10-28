"""Figure 5 (main): Random CV (dashed) vs Spatial CV (solid) — protocol comparison."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
RANDOM_CSV = ROOT / "results" / "benchmark" / "summary.csv"
SPATIAL_CSV = ROOT / "results" / "spatial_benchmark" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "fig05_spatial_vs_random"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}


def main():
    setup_style(journal="rse", base_fontsize=9)
    rand = pd.read_csv(RANDOM_CSV); rand["protocol"] = "Random"
    spatial = pd.read_csv(SPATIAL_CSV); spatial["protocol"] = "Spatial"
    df = pd.concat([rand, spatial], ignore_index=True)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(1, 4, figsize=(w, w * 0.30), sharey=True)

    methods = ["independent", "finetune", "reptile", "fomaml"]

    for i, (ax, tgt) in enumerate(zip(axes, TARGETS)):
        sub = df[df["target"] == tgt]
        for m in methods:
            for proto, ls, alpha in [("Random", "--", 0.55), ("Spatial", "-", 1.0)]:
                d = sub[(sub["method"] == m) & (sub["protocol"] == proto)].sort_values("K")
                if d.empty:
                    continue
                label = None
                if i == 0:
                    label = f"{METHOD_LABELS[m]} ({proto})"
                ax.plot(d["K"], d["f1_mean"],
                        color=METHOD_COLORS[m], linewidth=1.5 if proto == "Spatial" else 1.0,
                        linestyle=ls, alpha=alpha,
                        marker="o" if proto == "Spatial" else None,
                        markersize=4,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=label)

        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlim(0.85, 24)
        ax.set_ylim(0.35, 0.95)
        ax.set_xlabel("K (shots per class)")
        if i == 0:
            ax.set_ylabel("F1")

        add_panel_label(ax, "abcd"[i], x=-0.18 if i == 0 else -0.08, y=1.05, fontsize=10)
        ax.text(0.97, 0.04, TARGET_LABELS[tgt],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        style_ax(ax, x_grid=False, y_grid=True)

    # Legend (only protocol distinction, not all methods — too crowded)
    from matplotlib.lines import Line2D
    proto_handles = [
        Line2D([0], [0], color="#444", linestyle="--", linewidth=1.0, label="Random CV"),
        Line2D([0], [0], color="#444", linestyle="-", linewidth=1.5, marker="o",
               markersize=4, label="Spatial CV"),
    ]
    fig.legend(handles=proto_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=9)

    fig.subplots_adjust(wspace=0.10)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
