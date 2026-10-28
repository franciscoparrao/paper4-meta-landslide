"""Figure S11 (supp): precision-recall re-weighted to a realistic landslide
prevalence (pi=0.05), pooled K=1 spatial-CV queries. Shows that (i) absolute
precision is far below the balanced-F1 view, and (ii) the meta-learning
advantage survives the re-weighting. Data: scripts/28_prevalence_robust.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label, save_pub, setup_style

ROOT = Path(__file__).parent.parent.parent
NPZ = ROOT / "results" / "stats" / "pr_curves_pi.npz"
CSV = ROOT / "results" / "stats" / "prevalence_robust.csv"
OUT = ROOT / "figures_pub" / "out" / "figS11_precision_recall_prevalence"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TLAB = {"copiapo": "Copiapó", "huasco": "Huasco", "elqui": "Elqui", "limari": "Limarí"}
METHODS = ["independent", "reptile"]


def main():
    setup_style(journal="rse", base_fontsize=9)
    z = np.load(NPZ)
    pi = float(z["pi"])
    ap = pd.read_csv(CSV).set_index(["target", "method"])[f"ap_pi{pi:g}"]

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(1, 4, figsize=(w, w * 0.30), sharey=True)
    for i, (ax, t) in enumerate(zip(axes, TARGETS)):
        for m in METHODS:
            rec, prec = z[f"{t}_{m}_rec"], z[f"{t}_{m}_prec"]
            order = np.argsort(rec)
            ax.plot(rec[order], prec[order], color=METHOD_COLORS[m], lw=1.4,
                    label=f"{METHOD_LABELS[m]} (AP={ap.loc[(t, m)]:.3f})")
        ax.axhline(pi, ls=":", lw=0.8, color="0.5")  # no-skill baseline = prevalence
        ax.set_title(TLAB[t], fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, min(1.0, 0.45))
        ax.set_xlabel("Recall")
        ax.legend(fontsize=6.2, frameon=False, loc="upper right", handlelength=1.2)
        add_panel_label(ax, "abcd"[i])
    axes[0].set_ylabel(f"Precision at $\\pi={pi:g}$")
    axes[-1].text(0.97, pi + 0.01, "no-skill", fontsize=6, color="0.5",
                  ha="right", va="bottom", transform=axes[-1].get_yaxis_transform())
    save_pub(fig, OUT)


if __name__ == "__main__":
    main()
