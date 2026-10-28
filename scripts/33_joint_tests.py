#!/usr/bin/env python3
"""Paper 4 (R1) — pre-specified primary hypothesis, joint cross-basin test,
Holm-corrected secondary tests and replication-unit sensitivity
(reviewer 3, comments 5, 6, 10; reviewer 2, comment 3).

Primary hypothesis (pre-specified in Sect. 4.3 of the revision):
    H1: Reptile > Independent at K = 1 under spatial CV, jointly across the
        four target basins.
    Test: stratified permutation test on cluster-level paired differences.
          Within each target, the sign of every cluster's paired difference is
          flipped at random (exchangeability under H0: method labels within a
          shared partition are exchangeable); the statistic is the mean over
          targets of the per-target mean difference (equal basin weights, so
          Elqui's 15 clusters do not dominate Huasco's 6). One-sided p.
Secondary: per-target Wilcoxon / sign tests for Reptile and FOMAML vs
    Independent, Holm-corrected over the 8 tests at K = 1; same at K = 5, 10.
Replication unit sensitivity: (seed, fold) as in the submitted version vs
    fold with seeds averaged (conservative, n = 5 or fewer).

Usage:
    33_joint_tests.py [--raw results/spatial_benchmark/raw_runs.csv] [--tag r1]
Outputs (results/stats/):
    joint_tests_<tag>.csv        primary + per-K joint tests
    secondary_holm_<tag>.csv     per-target tests with raw and Holm p
    unit_sensitivity_<tag>.csv   (seed,fold) vs fold-averaged unit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stats"
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
META = ["reptile", "fomaml"]
OTHERS = ["finetune", "dann"]
N_PERM = 20000
N_BOOT = 5000
SEED = 42


def cluster_diffs(df, target, K, method, unit):
    """Paired per-cluster mean-F1 differences (method - independent)."""
    sub = df[(df.target == target) & (df.K == K)]
    keys = ["seed", "fold"] if unit == "seed_fold" else ["fold"]
    m = sub[sub.method == method].groupby(keys)["f1"].mean()
    i = sub[sub.method == "independent"].groupby(keys)["f1"].mean()
    common = m.index.intersection(i.index)
    return (m.loc[common] - i.loc[common]).to_numpy()


def per_target_test(d):
    nc = len(d)
    out = {"n_clusters": nc, "mean_diff_pp": float(d.mean() * 100),
           "n_positive": int((d > 0).sum())}
    try:
        out["wilcoxon_p"] = float(wilcoxon(d, alternative="two-sided").pvalue) if nc >= 2 else np.nan
    except ValueError:
        out["wilcoxon_p"] = np.nan
    # exact sign test (robust to n<6 where Wilcoxon cannot reach 0.05)
    nz = d[d != 0]
    out["sign_p"] = float(binomtest(int((nz > 0).sum()), len(nz), 0.5, alternative="two-sided").pvalue) if len(nz) else np.nan
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full_like(p, np.nan)
    running = 0.0
    rank = 0
    for idx in order:
        if not np.isfinite(p[idx]):
            continue
        rank += 1
        val = min(1.0, (m - rank + 1) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def joint_permutation(diffs_by_target, rng, n_perm=N_PERM):
    """Stratified sign-flip permutation; statistic = mean of per-target means."""
    obs = np.mean([d.mean() for d in diffs_by_target.values()])
    arrs = list(diffs_by_target.values())
    stats = np.empty(n_perm)
    for b in range(n_perm):
        s = 0.0
        for d in arrs:
            signs = rng.choice([-1.0, 1.0], size=len(d))
            s += (d * signs).mean()
        stats[b] = s / len(arrs)
    p_one = float((np.sum(stats >= obs) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(stats) >= abs(obs)) + 1) / (n_perm + 1))
    # bootstrap CI of the joint mean (resample clusters within target)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        boots[b] = np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in arrs])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, p_one, p_two, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results/spatial_benchmark/raw_runs.csv"))
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    df = pd.read_csv(args.raw)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    joint_rows, sec_rows, unit_rows = [], [], []
    for K in sorted(df.K.unique()):
        for unit in ["seed_fold", "fold"]:
            for method in META + OTHERS:
                diffs = {}
                for t in TARGETS:
                    d = cluster_diffs(df, t, K, method, unit)
                    if len(d) == 0:
                        continue
                    diffs[t] = d
                    r = per_target_test(d)
                    sec_rows.append({"K": K, "unit": unit, "target": t, "method": method, **r})
                if len(diffs) >= 2:
                    obs, p1, p2, lo, hi = joint_permutation(diffs, rng)
                    joint_rows.append({
                        "K": K, "unit": unit, "method": method,
                        "n_targets": len(diffs),
                        "n_clusters_total": int(sum(len(d) for d in diffs.values())),
                        "joint_mean_diff_pp": obs * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
                        "perm_p_one_sided": p1, "perm_p_two_sided": p2,
                        "primary": bool(K == 1 and method == "reptile" and unit == "seed_fold"),
                    })

    sec = pd.DataFrame(sec_rows)
    # Holm over the 8 pre-specified secondary tests per (K, unit): 4 targets x {reptile, fomaml}
    sec["wilcoxon_p_holm"] = np.nan
    sec["sign_p_holm"] = np.nan
    for (K, unit), g in sec[sec.method.isin(META)].groupby(["K", "unit"]):
        sec.loc[g.index, "wilcoxon_p_holm"] = holm(g["wilcoxon_p"].to_numpy())
        sec.loc[g.index, "sign_p_holm"] = holm(g["sign_p"].to_numpy())
    joint = pd.DataFrame(joint_rows)

    joint.to_csv(OUT / f"joint_tests_{args.tag}.csv", index=False)
    sec.to_csv(OUT / f"secondary_holm_{args.tag}.csv", index=False)

    # unit sensitivity summary
    for (K, method), g in sec[sec.method.isin(META)].groupby(["K", "method"]):
        a = g[g.unit == "seed_fold"].set_index("target")
        b = g[g.unit == "fold"].set_index("target")
        for t in a.index.intersection(b.index):
            unit_rows.append({"K": K, "method": method, "target": t,
                              "n_seed_fold": int(a.loc[t, "n_clusters"]), "n_fold": int(b.loc[t, "n_clusters"]),
                              "diff_pp_seed_fold": a.loc[t, "mean_diff_pp"], "diff_pp_fold": b.loc[t, "mean_diff_pp"],
                              "wilcoxon_p_seed_fold": a.loc[t, "wilcoxon_p"], "wilcoxon_p_fold": b.loc[t, "wilcoxon_p"],
                              "holm_seed_fold": a.loc[t, "wilcoxon_p_holm"], "holm_fold": b.loc[t, "wilcoxon_p_holm"],
                              "positive_seed_fold": f"{int(a.loc[t,'n_positive'])}/{int(a.loc[t,'n_clusters'])}",
                              "positive_fold": f"{int(b.loc[t,'n_positive'])}/{int(b.loc[t,'n_clusters'])}"})
    pd.DataFrame(unit_rows).to_csv(OUT / f"unit_sensitivity_{args.tag}.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\n[{args.tag}] raw = {args.raw}")
    print("\n== PRIMARY: joint stratified permutation test, Reptile - Independent, K=1, unit=(seed,fold) ==")
    pr = joint[joint.primary]
    print(pr[["joint_mean_diff_pp", "ci_lo_pp", "ci_hi_pp", "perm_p_one_sided", "perm_p_two_sided", "n_clusters_total"]].round(4).to_string(index=False))
    print("\n== Joint tests, all K / methods / units ==")
    print(joint[["K", "unit", "method", "joint_mean_diff_pp", "ci_lo_pp", "ci_hi_pp", "perm_p_two_sided"]].round(4).to_string(index=False))
    print("\n== Secondary per-target tests at K=1 (meta-learners), Holm over 8 ==")
    s1 = sec[(sec.K == 1) & sec.method.isin(META)]
    print(s1[["unit", "target", "method", "n_clusters", "mean_diff_pp", "n_positive", "wilcoxon_p", "wilcoxon_p_holm", "sign_p", "sign_p_holm"]].round(4).to_string(index=False))
    print("\n== Secondary at K=5 and K=10, unit=(seed,fold) ==")
    s2 = sec[(sec.K.isin([5, 10])) & sec.method.isin(META) & (sec.unit == "seed_fold")]
    print(s2[["K", "target", "method", "n_clusters", "mean_diff_pp", "n_positive", "wilcoxon_p", "wilcoxon_p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
