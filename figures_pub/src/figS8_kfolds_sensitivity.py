"""Figure S8 (supp): Meta-learning lift (FOMAML - Independent) sensitivity to the
number of spatial CV folds K in {5, 10, 20}. Shows the advantage is not an
artifact of the K_folds=5 choice.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import COLOR_WONG, JOURNAL_WIDTH, save_pub, setup_style, style_ax

ROOT = Path(__file__).parent.parent.parent
SENS = ROOT / "results" / "variogram" / "k_sensitivity.csv"
OUT = ROOT / "figures_pub" / "out" / "figS8_kfolds_sensitivity"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}
KF = [5, 10, 20]
# Sequential blues for ordered K_folds
KF_COLORS = {5: COLOR_WONG["skyblue"], 10: COLOR_WONG["blue"], 20: "#08306b"}


def main():
    setup_style(journal="rse", base_fontsize=9)
    df = pd.read_csv(SENS)
    df["lift_pp"] = (df["f1_fomaml"] - df["f1_independent"]) * 100
    agg = df.groupby(["target", "k_folds"])["lift_pp"].mean().reset_index()

    w = JOURNAL_WIDTH["rse"]["single"]
    fig, ax = plt.subplots(figsize=(w * 1.15, w * 0.85))

    x = np.arange(len(TARGETS))
    bw = 0.25
    for j, kf in enumerate(KF):
        vals = [agg[(agg.target == t) & (agg.k_folds == kf)]["lift_pp"].values
                for t in TARGETS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (j - 1) * bw, vals, width=bw,
               color=KF_COLORS[kf], edgecolor="white", linewidth=0.5,
               label=f"$K_{{\\mathrm{{folds}}}}={kf}$", zorder=3)

    ax.axhline(0, color="#222222", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS])
    ax.set_ylabel("FOMAML $-$ Independent ($F_1$ lift, pp)")
    ax.set_ylim(-0.3, 3.0)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=1)
    style_ax(ax, x_grid=False, y_grid=True)

    # Annotation: advantage persists across all K_folds
    ax.text(0.97, 0.95, "Advantage persists\nat all fold granularities",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            style="italic", color="#444444")

    fig.tight_layout()
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
