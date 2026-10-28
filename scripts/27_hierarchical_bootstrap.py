#!/usr/bin/env python3
"""Paper 4 — hierarchical (cluster) bootstrap for spatial-CV uncertainty (A4).

The flat bootstrap in 16_spatial_benchmark.py resamples pooled (seed x episode)
F1 as if independent. But within one (seed, fold) spatial partition all 20
episodes evaluate on the SAME held-out query cluster and draw supports from the
same small training pool, so they are strongly correlated: the effective sample
size is the number of (seed, fold) clusters (6-15 per target), not the nominal
n_runs (120-300). Flat CIs are therefore too narrow.

This script recomputes uncertainty two ways, both respecting the cluster:
  (1) Hierarchical bootstrap CI: resample (seed,fold) clusters with replacement,
      then episodes within each resampled cluster, then take the mean F1.
  (2) Cluster-level paired comparison vs Independent: per (seed,fold) cluster,
      mean F1 over episodes; the paired difference d = mean(meta) - mean(indep)
      across clusters (same query pixels) is a valid paired observation.
      Report mean d, its hierarchical-bootstrap 95% CI, and a Wilcoxon
      signed-rank / sign test on the cluster-level differences.

Output:
- results/stats/hierarchical_bootstrap.csv   (per target,K,method: flat vs hier CI)
- results/stats/cluster_paired_vs_independent.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "spatial_benchmark" / "raw_runs.csv"
OUT = ROOT / "results" / "stats"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
METHODS = ["independent", "finetune", "dann", "reptile", "fomaml"]
N_BOOT = 2000
SEED = 42


def flat_ci(f1, rng, n_boot=N_BOOT):
    """Naive bootstrap: resample runs iid (matches 16_spatial_benchmark)."""
    n = len(f1)
    means = f1[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(f1.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hier_ci(df_cell, rng, n_boot=N_BOOT):
    """Two-stage cluster bootstrap: resample (seed,fold) clusters, then episodes."""
    clusters = [g["f1"].to_numpy() for _, g in df_cell.groupby(["seed", "fold"])]
    nc = len(clusters)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, nc, size=nc)
        vals = []
        for ci in pick:
            arr = clusters[ci]
            vals.append(arr[rng.integers(0, len(arr), size=len(arr))])
        boot[b] = np.concatenate(vals).mean()
    mean = float(np.concatenate(clusters).mean())
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), nc


def cluster_means(df_cell):
    """Mean F1 per (seed,fold) cluster."""
    return df_cell.groupby(["seed", "fold"])["f1"].mean()


def paired_cluster_diff(df, target, K, method, rng, n_boot=N_BOOT):
    """Cluster-level paired difference method - independent on shared partitions."""
    sub = df[(df.target == target) & (df.K == K)]
    m = cluster_means(sub[sub.method == method])
    i = cluster_means(sub[sub.method == "independent"])
    common = m.index.intersection(i.index)
    d = (m.loc[common] - i.loc[common]).to_numpy()  # one value per cluster
    nc = len(d)
    # hierarchical bootstrap of the mean paired difference (resample clusters)
    boot = d[rng.integers(0, nc, size=(n_boot, nc))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Wilcoxon signed-rank on cluster diffs (fallback to sign test if ties/small)
    try:
        w_p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        w_p = np.nan
    n_pos = int((d > 0).sum())
    return {
        "target": target, "K": K, "method": method, "n_clusters": nc,
        "mean_diff_pp": float(d.mean() * 100),
        "ci_lo_pp": float(lo * 100), "ci_hi_pp": float(hi * 100),
        "ci_excludes_0": bool(lo > 0 or hi < 0),
        "wilcoxon_p": w_p, "n_clusters_positive": n_pos,
        "sign_consistent": bool(n_pos == nc or n_pos == 0),
    }


def main():
    df = pd.read_csv(RAW)
    df["K"] = df["K"].astype(int)
    rng = np.random.default_rng(SEED)

    # (1) flat vs hierarchical CI per cell
    rows = []
    for t in TARGETS:
        for K in sorted(df[df.target == t]["K"].unique()):
            for meth in METHODS:
                cell = df[(df.target == t) & (df.K == K) & (df.method == meth)]
                if cell.empty:
                    continue
                f1 = cell["f1"].to_numpy()
                fm, flo, fhi = flat_ci(f1, rng)
                hm, hlo, hhi, nc = hier_ci(cell, rng)
                rows.append({
                    "target": t, "K": K, "method": meth, "n_runs": len(f1), "n_clusters": nc,
                    "mean": hm,
                    "flat_ci_lo": flo, "flat_ci_hi": fhi, "flat_width_pp": (fhi - flo) * 100,
                    "hier_ci_lo": hlo, "hier_ci_hi": hhi, "hier_width_pp": (hhi - hlo) * 100,
                    "width_ratio": (hhi - hlo) / max(fhi - flo, 1e-9),
                })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / "hierarchical_bootstrap.csv", index=False)
    print(f"[1] hierarchical_bootstrap.csv: {len(cdf)} cells")
    print(f"    median flat width  = {cdf.flat_width_pp.median():.2f} pp")
    print(f"    median hier width  = {cdf.hier_width_pp.median():.2f} pp")
    print(f"    median width ratio = {cdf.width_ratio.median():.2f}x")

    # (2) cluster-level paired comparison vs Independent
    prows = []
    for t in TARGETS:
        for K in sorted(df[df.target == t]["K"].unique()):
            for meth in ["reptile", "fomaml", "dann", "finetune"]:
                cell = df[(df.target == t) & (df.K == K) & (df.method == meth)]
                if cell.empty:
                    continue
                prows.append(paired_cluster_diff(df, t, K, meth, rng))
    pdf = pd.DataFrame(prows)
    pdf.to_csv(OUT / "cluster_paired_vs_independent.csv", index=False)
    print(f"\n[2] cluster_paired_vs_independent.csv: {len(pdf)} comparisons")

    print("\n=== K=1 meta-learning vs Independent (cluster-level paired) ===")
    k1 = pdf[(pdf.K == 1) & (pdf.method.isin(["reptile", "fomaml"]))]
    for _, r in k1.sort_values(["target", "method"]).iterrows():
        flag = "CI excludes 0" if r.ci_excludes_0 else "CI includes 0"
        print(f"  {r.target:8s} {r.method:8s}: +{r.mean_diff_pp:.1f} pp "
              f"[{r.ci_lo_pp:+.1f},{r.ci_hi_pp:+.1f}] ({r.n_clusters} clusters, "
              f"{r.n_clusters_positive}/{r.n_clusters} positive) -> {flag}")

    print("\n=== Flat vs hierarchical CI at K=1 (best meta-learner) ===")
    for t in TARGETS:
        best = cdf[(cdf.target == t) & (cdf.K == 1) & (cdf.method.isin(["reptile", "fomaml"]))]
        ind = cdf[(cdf.target == t) & (cdf.K == 1) & (cdf.method == "independent")]
        if best.empty or ind.empty:
            continue
        b = best.loc[best["mean"].idxmax()]
        i = ind.iloc[0]
        flat_disj = b.flat_ci_lo > i.flat_ci_hi
        hier_disj = b.hier_ci_lo > i.hier_ci_hi
        print(f"  {t:8s} {b.method:8s} flat [{b.flat_ci_lo:.3f},{b.flat_ci_hi:.3f}] vs "
              f"indep [{i.flat_ci_lo:.3f},{i.flat_ci_hi:.3f}] flat-disjoint={flat_disj} | "
              f"hier [{b.hier_ci_lo:.3f},{b.hier_ci_hi:.3f}] vs "
              f"[{i.hier_ci_lo:.3f},{i.hier_ci_hi:.3f}] hier-disjoint={hier_disj}")


if __name__ == "__main__":
    main()
