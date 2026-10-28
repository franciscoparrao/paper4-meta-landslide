#!/usr/bin/env python3
"""Paper 4 (R1) — summarise buffered spatial CV (reviewer 3, comment 8).

Reads results/spatial_benchmark{,_buffer2km,_buffer5km,_buffer10km}/raw_runs.csv
(produced by 16_spatial_benchmark.py --buffer_km B) and reports, per buffer:
  * mean F1 of Independent and Reptile per target at K=1 and K=5
  * cluster-level paired Reptile - Independent (hierarchical CI, Wilcoxon)
  * joint (stratified permutation) Reptile - Independent across targets
Outputs results/stats/buffered_cv_summary.csv and figures_pub/out/figS_buffered_cv.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "stats"
FIG = ROOT / "figures_pub" / "out"
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
LAB = {"copiapo": "Copiapó", "huasco": "Huasco", "elqui": "Elqui", "limari": "Limarí"}
BUFFERS = [(0, "spatial_benchmark"), (2, "spatial_benchmark_buffer2km"),
           (5, "spatial_benchmark_buffer5km"), (10, "spatial_benchmark_buffer10km")]
N_BOOT, N_PERM, SEED = 2000, 20000, 42


def paired(raw, target, K, a="reptile", b="independent"):
    sub = raw[(raw.target == target) & (raw.K == K)]
    ma = sub[sub.method == a].groupby(["seed", "fold"]).f1.mean()
    mb = sub[sub.method == b].groupby(["seed", "fold"]).f1.mean()
    common = ma.index.intersection(mb.index)
    return (ma.loc[common] - mb.loc[common]).to_numpy()


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for buf, d in BUFFERS:
        p = R / d / "raw_runs.csv"
        if not p.exists():
            print(f"[skip] {p} not found"); continue
        raw = pd.read_csv(p)
        for K in [1, 5]:
            diffs = {}
            for t in TARGETS:
                sub = raw[(raw.target == t) & (raw.K == K)]
                if sub.empty:
                    continue
                d_ = paired(raw, t, K)
                if len(d_) == 0:
                    continue
                diffs[t] = d_
                boot = d_[rng.integers(0, len(d_), size=(N_BOOT, len(d_)))].mean(axis=1)
                try:
                    wp = float(wilcoxon(d_).pvalue) if len(d_) >= 2 else np.nan
                except ValueError:
                    wp = np.nan
                rows.append({"buffer_km": buf, "K": K, "target": t, "n_clusters": len(d_),
                             "f1_independent": sub[sub.method == "independent"].f1.mean(),
                             "f1_reptile": sub[sub.method == "reptile"].f1.mean(),
                             "f1_fomaml": sub[sub.method == "fomaml"].f1.mean() if (sub.method == "fomaml").any() else np.nan,
                             "lift_reptile_pp": d_.mean() * 100,
                             "ci_lo_pp": np.percentile(boot, 2.5) * 100, "ci_hi_pp": np.percentile(boot, 97.5) * 100,
                             "n_positive": int((d_ > 0).sum()), "wilcoxon_p": wp,
                             "n_runs_reptile": int((sub.method == "reptile").sum())})
            if len(diffs) >= 2:
                arrs = list(diffs.values())
                obs = np.mean([x.mean() for x in arrs])
                stats = np.empty(N_PERM)
                for b in range(N_PERM):
                    stats[b] = np.mean([(x * rng.choice([-1.0, 1.0], size=len(x))).mean() for x in arrs])
                p2 = float((np.sum(np.abs(stats) >= abs(obs)) + 1) / (N_PERM + 1))
                rows.append({"buffer_km": buf, "K": K, "target": "JOINT", "n_clusters": int(sum(len(x) for x in arrs)),
                             "f1_independent": np.nan, "f1_reptile": np.nan, "f1_fomaml": np.nan,
                             "lift_reptile_pp": obs * 100, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan,
                             "n_positive": np.nan, "wilcoxon_p": p2, "n_runs_reptile": np.nan})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "buffered_cv_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, K in zip(axes, [1, 5]):
            for t in TARGETS:
                d_ = df[(df.K == K) & (df.target == t)].sort_values("buffer_km")
                if d_.empty:
                    continue
                ax.errorbar(d_.buffer_km, d_.lift_reptile_pp,
                            yerr=[d_.lift_reptile_pp - d_.ci_lo_pp, d_.ci_hi_pp - d_.lift_reptile_pp],
                            marker="o", ms=4, capsize=2, lw=1, label=LAB[t])
            j = df[(df.K == K) & (df.target == "JOINT")].sort_values("buffer_km")
            if not j.empty:
                ax.plot(j.buffer_km, j.lift_reptile_pp, "k--", lw=1.5, label="Joint (4 targets)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("Buffer around held-out cluster (km)")
            ax.set_title(f"$K={K}$")
            ax.set_xticks([0, 2, 5, 10])
        axes[0].set_ylabel("Reptile $-$ Independent (pp $F_1$)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "figS_buffered_cv.pdf")
        print("Figure →", FIG / "figS_buffered_cv.pdf")
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
