#!/usr/bin/env python3
"""Paper 4 (R1) — audit that the headline numbers quoted in the manuscript match the result files
(reviewer 2, comment 7). Each entry derives a value from a CSV, formats it the way the text quotes it,
and checks that the string occurs in the LaTeX sources. Exit code 1 if any check fails.

Usage: 32_numbers_audit.py            (prints a table: number | source | found-in)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = "".join(p.read_text() for p in [ROOT / "paper/paper4_main.tex", *sorted((ROOT / "paper/sections").glob("*.tex"))])
TEX_NORM = TEX.replace("{,}", ",")
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
checks: list[tuple[str, str, str]] = []  # (label, formatted value, source)


def add(label, value, source):
    checks.append((label, value, source))


# --- Table 2 / Section 5.1: K=1 F1 for Independent and Reptile per target
s = pd.read_csv(R / "spatial_benchmark/summary.csv")
for t in TARGETS:
    for m in ["independent", "reptile"]:
        v = s[(s.target == t) & (s.K == 1) & (s.method == m)].f1_mean.iloc[0]
        add(f"F1 K=1 {m} {t}", f"{v:.3f}", "spatial_benchmark/summary.csv")

# --- Table 4: cluster-paired Reptile - Independent at K=1
cp = pd.read_csv(R / "stats/cluster_paired_vs_independent.csv")
for t in TARGETS:
    v = cp[(cp.target == t) & (cp.K == 1) & (cp.method == "reptile")].mean_diff_pp.iloc[0]
    add(f"paired Reptile-Indep K=1 {t} (pp)", f"{v:+.1f}", "stats/cluster_paired_vs_independent.csv")

# --- joint test
j = pd.read_csv(R / "stats/joint_tests_r1.csv")
j1 = j[(j.K == 1) & (j.unit == "seed_fold")].set_index("method")
add("joint Reptile K=1 (pp)", f"{j1.loc['reptile', 'joint_mean_diff_pp']:+.1f}", "stats/joint_tests_r1.csv")
add("joint Reptile K=1 CI lo", f"{j1.loc['reptile', 'ci_lo_pp']:+.1f}", "stats/joint_tests_r1.csv")
add("joint Reptile K=1 CI hi", f"{j1.loc['reptile', 'ci_hi_pp']:+.1f}", "stats/joint_tests_r1.csv")
add("joint FOMAML K=1 (pp)", f"{j1.loc['fomaml', 'joint_mean_diff_pp']:+.1f}", "stats/joint_tests_r1.csv")
j5 = j[(j.K == 5) & (j.unit == "seed_fold")].set_index("method")
add("joint Reptile K=5 (pp)", f"{j5.loc['reptile', 'joint_mean_diff_pp']:+.1f}", "stats/joint_tests_r1.csv")
add("joint DANN K=5 (pp)", f"{j5.loc['dann', 'joint_mean_diff_pp']:+.1f}", "stats/joint_tests_r1.csv")

# --- Holm
h = pd.read_csv(R / "stats/secondary_holm_r1.csv")
h1 = h[(h.K == 1) & (h.unit == "seed_fold") & (h.method == "reptile")].set_index("target")
for t in TARGETS:
    add(f"Holm p Reptile {t}", f"{h1.loc[t, 'wilcoxon_p_holm']:.3f}", "stats/secondary_holm_r1.csv")

# --- hierarchical width ratio
hb = pd.read_csv(R / "stats/hierarchical_bootstrap.csv")
add("median hier/flat width ratio", f"{hb.width_ratio.median():.1f}", "stats/hierarchical_bootstrap.csv")

# --- leakage: 2x2 with 11f vs 9f, same seeds (3 seeds runs)
for tag, label in [("pooled_vs_episodic_v1_cfs_all14", "11f"), ]:
    pass
cp9 = pd.read_csv(R / "pooled_vs_episodic/cluster_paired.csv")
add("2x2 Reptile - balanced K=1 mean (pp)", f"{cp9[(cp9.K == 1) & (cp9.contrast == 'reptile - finetune_balanced')].mean_diff_pp.mean():+.2f}", "pooled_vs_episodic/cluster_paired.csv")
add("2x2 balanced - concat K=1 mean (pp)", f"{cp9[(cp9.K == 1) & (cp9.contrast == 'finetune_balanced - finetune')].mean_diff_pp.mean():+.1f}", "pooled_vs_episodic/cluster_paired.csv")
add("2x2 Reptile - Indep K=1 mean (pp)", f"{cp9[(cp9.K == 1) & (cp9.contrast == 'reptile - independent')].mean_diff_pp.mean():+.1f}", "pooled_vs_episodic/cluster_paired.csv")

# --- prevalence (AUC)
pr = pd.read_csv(R / "stats/prevalence_robust.csv")
for t in TARGETS:
    for m in ["independent", "reptile"]:
        add(f"AUC K=1 {m} {t}", f"{pr[(pr.target == t) & (pr.method == m)].auc.iloc[0]:.3f}", "stats/prevalence_robust.csv")

# --- buffers (joint K=1)
b = pd.read_csv(R / "stats/buffered_cv_summary.csv")
for buf in [0, 2, 5, 10, 20]:
    v = b[(b.buffer_km == buf) & (b.K == 1) & (b.target == "JOINT")].lift_reptile_pp.iloc[0]
    add(f"buffer {buf} km joint K=1 (pp)", f"{v:+.1f}", "stats/buffered_cv_summary.csv")

# --- variogram ranges
vg = pd.read_csv(R / "variogram/variogram_per_basin.csv").set_index("target")
for t in TARGETS:
    add(f"variogram range {t} (km)", f"{vg.loc[t, 'range_km']:.1f}", "variogram/variogram_per_basin.csv")

# --- fold diagnostics
ov = pd.read_csv(R / "stats/fold_overlap.csv")
for t in TARGETS:
    add(f"ARI {t}", f"{ov[ov.target == t].ari.mean():.2f}", "stats/fold_overlap.csv")

# --- negatives sensitivity range
ng = pd.read_csv(R / "stats/negatives_sensitivity.csv")
add("negatives lift min (pp)", f"{ng.lift_pp.min():+.1f}", "stats/negatives_sensitivity.csv")
add("negatives lift max (pp)", f"{ng.lift_pp.max():+.1f}", "stats/negatives_sensitivity.csv")

# --- budget control
bd = pd.read_csv(R / "stats/baseline_budget_raw.csv") if (R / "stats/baseline_budget_raw.csv").exists() else None

# --- preprocessing sensitivity
pp = pd.read_csv(R / "stats/preproc_sensitivity.csv")
for mode in ["basin", "source", "onehot"]:
    add(f"preproc {mode} mean lift (pp)", f"{pp[pp['mode'] == mode].lift_pp.mean():+.1f}", "stats/preproc_sensitivity.csv")

# --- dataset counts
ds = pd.read_csv(ROOT / "data/ml_ready/dataset_summary.csv")
add("positives total", f"{int(ds.n_pos.sum()):,}", "data/ml_ready/dataset_summary.csv")
add("source positives", f"{int(ds[ds.role == 'source'].n_pos.sum()):,}", "data/ml_ready/dataset_summary.csv")
add("taltal positives", f"{int(ds[ds.basin == 'taltal'].n_pos.iloc[0]):,}", "data/ml_ready/dataset_summary.csv")


# --- source-pool ablations (R1, reviewer 3 c.11-12)
sa = pd.read_csv(R / "source_ablations/paired.csv")
for cond in ["baseline", "no_maule", "homog_neg"]:
    v = sa[(sa.condition == cond) & (sa.K == 1) & (sa.method == "reptile") & (sa.target == "JOINT")].diff_pp.iloc[0]
    add(f"source ablation {cond} joint K=1 (pp)", f"{v:+.1f}", "source_ablations/paired.csv")

# --- random vs spatial CV inflation (Sect. 5.5, Fig. 5 caption, abstract); ranges over the four plotted methods
rb = pd.read_csv(R / "benchmark/summary.csv")
rv = s.merge(rb, on=["target", "K", "method"], suffixes=("_s", "_r"))
rv["d"] = 100 * (rv.f1_mean_r - rv.f1_mean_s)
rv = rv[rv.method.isin(["independent", "finetune", "reptile", "fomaml"])]


def rng(sub, sign=1):
    lo, hi = sorted([sign * sub.d.min(), sign * sub.d.max()])
    return f"{round(lo)}--{round(hi)}"


add("random-spatial Elqui all K (pp)", rng(rv[rv.target == "elqui"]), "benchmark + spatial_benchmark summary")
add("random-spatial Limari all K (pp)", rng(rv[rv.target == "limari"]), "benchmark + spatial_benchmark summary")
add("random-spatial Copiapo K=1 (pp)", rng(rv[(rv.target == "copiapo") & (rv.K == 1)]), "benchmark + spatial_benchmark summary")
add("random-spatial Copiapo K=5 (pp)", rng(rv[(rv.target == "copiapo") & (rv.K == 5)]), "benchmark + spatial_benchmark summary")
add("spatial-random Huasco K=1 (pp)", rng(rv[(rv.target == "huasco") & (rv.K == 1)], -1), "benchmark + spatial_benchmark summary")
add("random-spatial max (pp)", f"up to {round(rv.d.max())}", "benchmark + spatial_benchmark summary")

# --- random-CV design (Sect. 4.3, Fig. 2 caption): episodes per (target, K, method, seed)
rr = pd.read_csv(R / "benchmark/raw_runs.csv")
n_ep = rr.groupby(["target", "K", "method", "seed"]).episode.nunique().unique()
add("random-CV episodes per seed", f"{int(n_ep[0])} episodes" if len(n_ep) == 1 else "inconsistent", "benchmark/raw_runs.csv")
add("random-CV adaptations total", f"{len(rr):,}", "benchmark/raw_runs.csv")

# --- Elqui spatial-CV F1 range over the five methods (Sect. 6.3)
for k in [1, 20]:
    e = s[(s.target == "elqui") & (s.K == k)].f1_mean
    add(f"Elqui spatial F1 range K={k}", f"{e.min():.2f}$--${e.max():.2f}$", "spatial_benchmark/summary.csv")

# --- run the checks
fails = 0
print(f"{'check':<44} {'value':>8}  found  source")
for label, value, source in checks:
    found = value in TEX_NORM
    fails += (not found)
    print(f"{label:<44} {value:>8}  {'ok ' if found else 'MISSING'}  {source}")
print(f"\n{len(checks) - fails}/{len(checks)} numbers found in the LaTeX sources")
sys.exit(1 if fails else 0)
