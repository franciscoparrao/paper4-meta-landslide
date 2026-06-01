#!/usr/bin/env python3
"""Paper 4 — paired statistical tests for method comparisons.

For each (target, K) pair, compute:
- Wilcoxon signed-rank test on paired (episode, seed) F1 scores
- Cohen's d effect size
- Bonferroni-corrected p-values across method pairs

Tests: FOMAML vs Independent, FOMAML vs Finetune, FOMAML vs DANN,
       Reptile vs Independent, Reptile vs DANN

Output:
- results/stats/paired_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
OUT = ROOT / "results" / "stats"
OUT.mkdir(parents=True, exist_ok=True)


def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def main():
    # Load both random and spatial benchmarks
    raw_random = pd.read_csv(ROOT / "results/benchmark/raw_runs.csv")
    raw_spatial = pd.read_csv(ROOT / "results/spatial_benchmark/raw_runs.csv")

    pairs = [
        ("fomaml", "independent"),
        ("fomaml", "finetune"),
        ("fomaml", "dann"),
        ("fomaml", "reptile"),
        ("reptile", "independent"),
        ("reptile", "dann"),
        ("finetune", "independent"),
    ]

    rows = []
    for protocol, raw in [("random", raw_random), ("spatial", raw_spatial)]:
        for tgt in raw.target.unique():
            for K in sorted(raw.K.unique()):
                sub = raw[(raw.target == tgt) & (raw.K == K)]
                if sub.empty:
                    continue
                # Pivot: pair on (seed, episode)
                pivot = sub.pivot_table(index=["seed", "episode"], columns="method",
                                        values="f1").dropna()
                if len(pivot) < 5:
                    continue
                for m1, m2 in pairs:
                    if m1 not in pivot.columns or m2 not in pivot.columns:
                        continue
                    a = pivot[m1].values
                    b = pivot[m2].values
                    if np.allclose(a, b):
                        continue
                    try:
                        stat, p = wilcoxon(a, b, alternative="two-sided",
                                           zero_method="wilcox")
                    except ValueError:
                        continue
                    d = cohens_d(a, b)
                    rows.append({
                        "protocol": protocol, "target": tgt, "K": K,
                        "method_1": m1, "method_2": m2,
                        "n_pairs": len(pivot),
                        "mean_1": float(a.mean()), "mean_2": float(b.mean()),
                        "diff": float(a.mean() - b.mean()),
                        "wilcoxon_stat": float(stat), "p_value": float(p),
                        "cohens_d": float(d),
                    })

    df = pd.DataFrame(rows)
    # Bonferroni correction within each (protocol, target, K)
    df["p_bonf"] = df.groupby(["protocol", "target", "K"])["p_value"].transform(
        lambda x: np.minimum(x * len(x), 1.0))
    df["sig_05"] = df["p_bonf"] < 0.05
    df["effect"] = pd.cut(df["cohens_d"].abs(),
                          bins=[0, 0.2, 0.5, 0.8, np.inf],
                          labels=["negligible", "small", "medium", "large"])

    df.to_csv(OUT / "paired_tests.csv", index=False)
    print(f"Paired tests → {OUT / 'paired_tests.csv'} ({len(df)} comparisons)")
    print(f"\nSignificant comparisons (p_bonf < 0.05): {df['sig_05'].sum()} / {len(df)}")
    print("\n=== K=1 spatial CV: FOMAML vs Independent ===")
    sub = df[(df.protocol == "spatial") & (df.K == 1)
             & (df.method_1 == "fomaml") & (df.method_2 == "independent")]
    for _, r in sub.iterrows():
        print(f"  {r['target']:<10} diff={r['diff']:+.3f}  d={r['cohens_d']:+.2f} ({r['effect']})  "
              f"p={r['p_value']:.4f}  p_bonf={r['p_bonf']:.4f}  sig={r['sig_05']}")

    print("\n=== K=1 spatial CV: FOMAML vs DANN ===")
    sub = df[(df.protocol == "spatial") & (df.K == 1)
             & (df.method_1 == "fomaml") & (df.method_2 == "dann")]
    for _, r in sub.iterrows():
        print(f"  {r['target']:<10} diff={r['diff']:+.3f}  d={r['cohens_d']:+.2f} ({r['effect']})  "
              f"p={r['p_value']:.4f}  p_bonf={r['p_bonf']:.4f}  sig={r['sig_05']}")


if __name__ == "__main__":
    main()
