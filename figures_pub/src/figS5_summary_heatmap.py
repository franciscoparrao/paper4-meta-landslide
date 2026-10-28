"""Figure S5: Summary heatmap (target × K × method, spatial CV)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_LABELS, JOURNAL_WIDTH, save_pub, setup_style)

ROOT = Path(__file__).parent.parent.parent
BENCH = ROOT / "results" / "spatial_benchmark" / "summary.csv"
OUT = ROOT / "figures_pub" / "out" / "figS5_summary_heatmap"


def main():
    setup_style(journal="rse", base_fontsize=9)
    df = pd.read_csv(BENCH)
    df["row"] = df["target"] + " K=" + df["K"].astype(str)

    # Build target_order but skip rows where ALL methods are NaN (e.g., Copiapó K=20)
    target_order = []
    target_groups = {}  # basin -> list of row indices for separators
    methods = ["independent", "finetune", "dann", "reptile", "fomaml"]

    raw_pivot = df.pivot_table(index="row", columns="method", values="f1_mean")
    for t in ["copiapo", "huasco", "elqui", "limari"]:
        target_groups[t] = []
        for K in [1, 5, 10, 20]:
            row = f"{t} K={K}"
            if row not in raw_pivot.index:
                continue
            if raw_pivot.loc[row].isna().all():
                continue  # skip rows with no data
            target_order.append(row)
            target_groups[t].append(len(target_order) - 1)

    pivot = raw_pivot.reindex(index=target_order, columns=methods)

    w = JOURNAL_WIDTH["rse"]["single"]
    fig, ax = plt.subplots(figsize=(w * 1.3, max(w * 1.4, 0.30 * len(pivot))))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.40, vmax=0.95)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, family="monospace", fontsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                color = "white" if v < 0.70 else "#111"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    # Horizontal separators between basin groups (black thin lines, zorder=10)
    first_basin = True
    for t, indices in target_groups.items():
        if indices and not first_basin:
            top = indices[0] - 0.5
            ax.axhline(top, color="#000000", linewidth=1.2, zorder=10)
        first_basin = False

    cbar = plt.colorbar(im, ax=ax, label="F1 (spatial CV)", fraction=0.04, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("")
    ax.tick_params(top=False, right=False, length=0)
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)

    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
