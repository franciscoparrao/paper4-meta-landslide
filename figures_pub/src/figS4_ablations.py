"""Figure S4: Hyperparameter ablations — 3-panel (inner_steps / eps / K_meta)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from style import (METHOD_COLORS, METHOD_LABELS, JOURNAL_WIDTH, add_panel_label,
                   save_pub, setup_style, style_ax)

ROOT = Path(__file__).parent.parent.parent
SUMMARY = ROOT / "results" / "ablations" / "ablations_summary.csv"
OUT = ROOT / "figures_pub" / "out" / "figS4_ablations"

BASELINE = {"inner_steps": 5, "eps": 0.1, "K_meta": 10}
SWEEPS = ["inner_steps", "eps", "K_meta"]
SWEEP_TITLE = {
    "inner_steps": "Inner steps",
    "eps":         r"$\epsilon$ (Reptile only)",
    "K_meta":      "$K_{meta}$",
}


def main():
    setup_style(journal="rse", base_fontsize=9)
    df = pd.read_csv(SUMMARY)

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.32), sharey=True)

    for i, (ax, sweep) in enumerate(zip(axes, SWEEPS)):
        sub = df[df["sweep"] == sweep]
        for method in ["reptile", "fomaml"]:
            d = sub[sub["method"] == method].sort_values("value")
            if d.empty:
                continue
            ax.errorbar(
                d["value"], d["f1_mean"],
                yerr=[d["f1_mean"] - d["f1_ci_lo"], d["f1_ci_hi"] - d["f1_mean"]],
                fmt="o-", color=METHOD_COLORS[method],
                linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                capsize=3, capthick=0.6,
                label=METHOD_LABELS[method] if i == 0 else None,
            )
        baseline_v = BASELINE[sweep]
        ax.axvline(baseline_v, color="#666", linestyle="--", linewidth=0.6,
                   alpha=0.6, zorder=0, label="baseline" if i == 0 else None)
        ax.set_xlabel(SWEEP_TITLE[sweep])
        add_panel_label(ax, "abc"[i], x=-0.18 if i == 0 else -0.08, y=1.05, fontsize=10)
        style_ax(ax)
        if i == 0:
            ax.set_ylabel("F1 at K=5 (mean ± 95% CI)")
        # log scale for K_meta and inner_steps (wide range), linear for eps
        if sweep in ("inner_steps", "K_meta"):
            ax.set_xscale("log")
            vals = sorted(sub["value"].unique())
            ax.set_xticks(vals)
            ax.set_xticklabels([str(int(v)) for v in vals])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.subplots_adjust(wspace=0.10)
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
