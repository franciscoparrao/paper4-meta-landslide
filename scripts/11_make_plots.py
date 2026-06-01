#!/usr/bin/env python3
"""Paper 4 — generate publication-grade plots from benchmark results.

Outputs (figures/):
- f1_vs_k_panel.png  : 4-panel figure (one per target basin) of F1 vs K with 95% CI bands per method.
- auc_vs_k_panel.png : same for AUC.
- lift_vs_k.png      : F1 advantage of each meta-method over independent baseline.
- summary_heatmap.png: heatmap of F1 mean by (target, K) × method.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
BENCH = ROOT / "results" / "benchmark"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}
METHODS = ["independent", "finetune", "reptile", "fomaml"]
METHOD_LABELS = {
    "independent": "Independent (no source)",
    "finetune":    "Fine-tune (concat source)",
    "reptile":     "Reptile",
    "fomaml":      "FOMAML",
}
METHOD_COLORS = {
    "independent": "#888888",
    "finetune":    "#2c7fb8",
    "reptile":     "#d95f0e",
    "fomaml":      "#c51b8a",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def panel(metric: str, ci_lo: str, ci_hi: str, ylabel: str, fname: str):
    summary = pd.read_csv(BENCH / "summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, tgt in zip(axes.flat, TARGETS):
        sub = summary[summary.target == tgt]
        for m in METHODS:
            d = sub[sub.method == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d[metric],
                    marker="o", color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m], linewidth=1.8, markersize=6)
            ax.fill_between(d["K"], d[ci_lo], d[ci_hi],
                            color=METHOD_COLORS[m], alpha=0.18, linewidth=0)
        ax.set_title(TARGET_LABELS[tgt])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K (shots per class)")
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("", y=1.0)
    fig.tight_layout()
    out = FIGS / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def lift_plot():
    summary = pd.read_csv(BENCH / "summary.csv")
    pivot = summary.pivot_table(index=["target", "K"], columns="method",
                                values="f1_mean").reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ["finetune", "reptile", "fomaml"]:
        for tgt in TARGETS:
            d = pivot[pivot.target == tgt].sort_values("K")
            if d.empty:
                continue
            lift = d[m] - d["independent"]
            ax.plot(d["K"], lift, marker="o", linestyle="-",
                    color=METHOD_COLORS[m], alpha=0.5, linewidth=1)
        # Mean across targets
        mean_lift = (pivot[m] - pivot["independent"]).groupby(pivot["K"]).mean()
        ax.plot(mean_lift.index, mean_lift.values, marker="s",
                color=METHOD_COLORS[m], linewidth=2.5,
                label=f"{METHOD_LABELS[m]} (mean)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(["1", "5", "10", "20"])
    ax.set_xlabel("K (shots per class)")
    ax.set_ylabel("F1 advantage over Independent baseline")
    ax.set_title("Meta-learning advantage decays with K")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = FIGS / "lift_vs_k.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def heatmap():
    summary = pd.read_csv(BENCH / "summary.csv")
    summary["target_K"] = summary["target"] + " K=" + summary["K"].astype(str)
    pivot = summary.pivot_table(index="target_K", columns="method",
                                values="f1_mean")
    # Order target_K
    target_order = []
    for t in TARGETS:
        for K in [1, 5, 10, 20]:
            label = f"{t} K={K}"
            if label in pivot.index:
                target_order.append(label)
    pivot = pivot.reindex(index=target_order, columns=METHODS)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=0.95)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.75 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="F1")
    ax.set_title("F1 (mean) — target × K × method")
    fig.tight_layout()
    out = FIGS / "summary_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def main():
    print("Generating publication plots...")
    panel("f1_mean",  "f1_ci_lo",  "f1_ci_hi",  "F1",  "f1_vs_k_panel.png")
    panel("auc_mean", "auc_ci_lo", "auc_ci_hi", "AUC", "auc_vs_k_panel.png")
    lift_plot()
    heatmap()
    print(f"\nAll figures in {FIGS}")


if __name__ == "__main__":
    main()
