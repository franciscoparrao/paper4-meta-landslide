#!/usr/bin/env python3
"""Paper 4 (R1) — regenerate the LaTeX tables of the main text from the result CSVs,
so that every number in a table is traceable to a file (reviewer 2, comment 7).

  paper/sections/table01_spatial_results.tex   <- results/spatial_benchmark/summary.csv
                                                  + results/stats/hierarchical_bootstrap.csv (R3 c.10)
  paper/sections/table02_extended_baselines.tex<- results/extended_baselines/summary.csv
                                                  (flat CIs: 23 has no hierarchical run yet; stated in caption)
  paper/sections/table03_cluster_paired.tex    <- results/stats/cluster_paired_vs_independent.csv
                                                  + results/stats/secondary_holm_r1.csv (Holm, R3 c.6)
                                                  + results/stats/joint_tests_r1.csv (primary joint test)
  paper/sections/table05_pooled_vs_episodic.tex<- results/pooled_vs_episodic/{summary,cluster_paired}.csv (R3 c.4, R2 c.4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SEC = ROOT / "paper" / "sections"
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
LAB = {"copiapo": "Copiap\\'o", "huasco": "Huasco", "elqui": "Elqui", "limari": "Limar\\'i"}
MLAB = {"independent": "Independent", "finetune": "Fine-tune", "dann": "DANN", "reptile": "Reptile",
        "fomaml": "FOMAML", "protonet": "ProtoNet", "meta_baseline": "Meta-Baseline", "cdan": "CDAN",
        "finetune_balanced": "Fine-tune (balanced)", "pretrain_then_reptile": "Pretrain$\\rightarrow$Reptile"}


def cell(m, lo, hi, bold=False):
    s = f"{m:.3f} [{lo:.2f},{hi:.2f}]"
    return f"\\textbf{{{s}}}" if bold else s


def table01():
    s = pd.read_csv(R / "spatial_benchmark/summary.csv")
    h = pd.read_csv(R / "stats/hierarchical_bootstrap.csv")
    s = s.merge(h[["target", "K", "method", "hier_ci_lo", "hier_ci_hi", "n_clusters"]], on=["target", "K", "method"])
    methods = ["independent", "finetune", "dann", "reptile", "fomaml"]
    n_runs = int(s.n_runs.sum() / len(methods))
    nc = {t: int(s[(s.target == t) & (s.K == 1)].n_clusters.iloc[0]) for t in TARGETS}
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{$F_1$ score (mean) with \textbf{two-stage hierarchical-bootstrap} 95\% confidence intervals "
             r"(resampling $(\text{seed},\text{fold})$ partitions, then episodes; 2{,}000 resamples) for each target basin "
             r"$\times$ shots-per-class $K$ $\times$ method, under spatial cross-validation with the leakage-free, "
             r"source-only feature set (9 features) and five seeds. Best per row in \textbf{bold}. Copiap\'o $K=20$ is not "
             r"reported ($N=19$ events). Valid $(\text{seed},\text{fold})$ partitions: " +
             ", ".join(f"{LAB[t]} {nc[t]}/25" for t in TARGETS) +
             r" (Section~\ref{sec:methods:protocol}); total realized $K$-shot adaptations: " + f"{n_runs:,}".replace(",", "{,}") +
             r". Source: \texttt{results/spatial\_benchmark/summary.csv}, \texttt{results/stats/hierarchical\_bootstrap.csv}.}",
             r"    \label{tab:spatial_results}", r"    \footnotesize", r"    \begin{tabular}{llccccc}", r"        \toprule",
             r"        \textbf{Target} & \textbf{$K$} & " + " & ".join(f"\\textbf{{{MLAB[m]}}}" for m in methods) + r" \\",
             r"        \midrule"]
    for ti, t in enumerate(TARGETS):
        sub = s[s.target == t]
        first = True
        for K in sorted(sub.K.unique()):
            row = sub[sub.K == K].set_index("method")
            best = row.f1_mean.idxmax()
            cells = [cell(row.loc[m, "f1_mean"], row.loc[m, "hier_ci_lo"], row.loc[m, "hier_ci_hi"], m == best) for m in methods]
            lines.append(f"        {LAB[t] if first else '':<10} & {K:<3}& " + " & ".join(cells) + r" \\")
            first = False
        if ti < len(TARGETS) - 1:
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "table01_spatial_results.tex").write_text("\n".join(lines) + "\n")


def table02():
    s = pd.read_csv(R / "extended_baselines/summary.csv")
    methods = ["protonet", "meta_baseline", "cdan"]
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{$F_1$ score (mean) with bootstrap 95\% confidence intervals (pooled seed~$\times$~episode; "
             r"the hierarchical intervals of Table~\ref{tab:spatial_results} are roughly $3\times$ wider) for the three "
             r"additional baselines under spatial cross-validation with the source-only feature set and five seeds. Best per row in "
             r"\textbf{bold}. For comparison against the primary methods see Table~\ref{tab:spatial_results}. "
             r"Source: \texttt{results/extended\_baselines/summary.csv}.}",
             r"    \label{tab:extended_baselines}", r"    \footnotesize", r"    \begin{tabular}{llccc}", r"        \toprule",
             r"        \textbf{Target} & \textbf{$K$} & " + " & ".join(f"\\textbf{{{MLAB[m]}}}" for m in methods) + r" \\",
             r"        \midrule"]
    for ti, t in enumerate(TARGETS):
        sub = s[s.target == t]
        first = True
        for K in sorted(sub.K.unique()):
            row = sub[sub.K == K].set_index("method")
            best = row.f1_mean.idxmax()
            cells = [cell(row.loc[m, "f1_mean"], row.loc[m, "f1_ci_lo"], row.loc[m, "f1_ci_hi"], m == best) for m in methods]
            lines.append(f"        {LAB[t] if first else '':<10} & {K:<3}& " + " & ".join(cells) + r" \\")
            first = False
        if ti < len(TARGETS) - 1:
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "table02_extended_baselines.tex").write_text("\n".join(lines) + "\n")


def table03():
    cp = pd.read_csv(R / "stats/cluster_paired_vs_independent.csv")
    sec = pd.read_csv(R / "stats/secondary_holm_r1.csv")
    sec = sec[(sec.unit == "seed_fold")]
    joint = pd.read_csv(R / "stats/joint_tests_r1.csv")
    joint = joint[(joint.unit == "seed_fold") & (joint.K == 1)].set_index("method")
    methods = ["reptile", "fomaml", "finetune", "dann"]
    cp1 = cp[cp.K == 1]

    def fmt_p(p):
        return "---" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")

    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{Cluster-level paired comparison of each method against the Independent baseline at $K=1$ under spatial "
             r"cross-validation (source-only features, five seeds). The paired unit is the $(\text{seed},\text{fold})$ spatial partition "
             r"($n_c$ per target); the paired difference is the per-cluster mean-$F_1$ gap. Intervals are 95\% hierarchical-bootstrap CIs "
             r"(2{,}000 resamples); $^{*}$CI excludes zero. $p$ is the two-sided Wilcoxon signed-rank test on the cluster differences; "
             r"$p_{\text{Holm}}$ applies the Holm correction over the eight pre-specified secondary tests (four targets $\times$ "
             r"\{Reptile, FOMAML\}). The \textbf{primary, pre-specified hypothesis} is the joint row: a permutation test stratified by target "
             r"(sign flips of the cluster differences within each target, 20{,}000 permutations; statistic = mean of per-target mean differences). "
             r"Source: \texttt{results/stats/cluster\_paired\_vs\_independent.csv}, \texttt{secondary\_holm\_r1.csv}, \texttt{joint\_tests\_r1.csv}.}",
             r"    \label{tab:cluster_paired}", r"    \footnotesize", r"    \begin{tabular}{llcccc}", r"        \toprule",
             r"        \textbf{Target} & \textbf{Method} & \textbf{$\Delta F_1$ vs Indep.\ (pp) [95\% CI]} & \textbf{Clusters $>0$} & "
             r"\textbf{Wilcoxon $p$} & \textbf{$p_{\text{Holm}}$} \\", r"        \midrule"]
    for ti, t in enumerate(TARGETS):
        sub = cp1[cp1.target == t].set_index("method")
        nc = int(sub.n_clusters.iloc[0])
        first = True
        for m in methods:
            r = sub.loc[m]
            star = "$^{*}$" if r.ci_excludes_0 else ""
            hp = sec[(sec.K == 1) & (sec.target == t) & (sec.method == m)]
            holm = fmt_p(hp.wilcoxon_p_holm.iloc[0]) if len(hp) and m in ("reptile", "fomaml") else "---"
            lab = f"{LAB[t]} ($n_c={nc}$)" if first else ""
            lines.append(f"        {lab} & {MLAB[m]} & ${r.mean_diff_pp:+.1f}$ [{r.ci_lo_pp:+.1f}, {r.ci_hi_pp:+.1f}]{star} & "
                         f"{int(r.n_clusters_positive)}/{nc} & {fmt_p(r.wilcoxon_p)} & {holm} \\\\")
            first = False
        lines.append(r"        \midrule")
    lines.append(r"        \multicolumn{6}{l}{\textit{Joint test across the four targets (stratified permutation; primary hypothesis in bold)}} \\")
    for m in methods:
        j = joint.loc[m]
        lab = r"\textbf{Joint}" if m == "reptile" else "Joint"
        name = f"\\textbf{{{MLAB[m]}}}" if m == "reptile" else MLAB[m]
        lines.append(f"        {lab} ($n_c={int(j.n_clusters_total)}$) & {name} & ${j.joint_mean_diff_pp:+.1f}$ "
                     f"[{j.ci_lo_pp:+.1f}, {j.ci_hi_pp:+.1f}] & --- & {fmt_p(j.perm_p_two_sided)} & --- \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "table03_cluster_paired.tex").write_text("\n".join(lines) + "\n")


def table05():
    s = pd.read_csv(R / "pooled_vs_episodic/summary.csv")
    cp = pd.read_csv(R / "pooled_vs_episodic/cluster_paired.csv")
    methods = ["independent", "finetune", "finetune_balanced", "reptile", "pretrain_then_reptile"]
    contrasts = [("finetune_balanced - finetune", "Balanced $-$ concat (domain weighting)"),
                 ("reptile - finetune_balanced", "Reptile $-$ balanced (episodic vs.\\ pooled)"),
                 ("pretrain_then_reptile - finetune", "Pretrain$\\rightarrow$Reptile $-$ concat"),
                 ("reptile - independent", "Reptile $-$ Independent")]
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{Domain weighting versus episodic training at $K=1$ under spatial cross-validation (source-only features, "
             r"five seeds, identical partitions and support draws for all five initialisations). Top: mean $F_1$. Bottom: cluster-level "
             r"paired contrasts (pp; mean [95\% hierarchical-bootstrap CI]; two-sided Wilcoxon $p$) isolating the effect of basin-balanced "
             r"weighting in pooled pre-training, of episodic training given balanced weighting, and of adding an episodic phase after pooled "
             r"pre-training. Source: \texttt{results/pooled\_vs\_episodic/\{summary,cluster\_paired\}.csv}.}",
             r"    \label{tab:pooled_vs_episodic}", r"    \footnotesize", r"    \begin{tabular}{lcccc}", r"        \toprule",
             r"        & " + " & ".join(f"\\textbf{{{LAB[t]}}}" for t in TARGETS) + r" \\", r"        \midrule",
             r"        \multicolumn{5}{l}{\textit{Mean $F_1$ at $K=1$}} \\"]
    k1 = s[s.K == 1].pivot(index="method", columns="target", values="f1_mean")
    for m in methods:
        lines.append(f"        {MLAB[m]} & " + " & ".join(f"{k1.loc[m, t]:.3f}" for t in TARGETS) + r" \\")
    lines.append(r"        \midrule")
    lines.append(r"        \multicolumn{5}{l}{\textit{Cluster-paired contrasts at $K=1$ (pp)}} \\")
    c1 = cp[cp.K == 1].set_index(["contrast", "target"])
    for key, lab in contrasts:
        cells = []
        for t in TARGETS:
            r = c1.loc[(key, t)]
            star = "$^{*}$" if r.ci_excludes_0 else ""
            ptxt = "$p<0.01$" if r.wilcoxon_p < 0.01 else f"$p={r.wilcoxon_p:.2f}$"
            cells.append(f"${r.mean_diff_pp:+.1f}$ [{r.ci_lo_pp:+.1f}, {r.ci_hi_pp:+.1f}]{star}, {ptxt}")
        lines.append(f"        {lab} & " + " & ".join(cells) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "table05_pooled_vs_episodic.tex").write_text("\n".join(lines) + "\n")


def table04():
    d = pd.read_csv(R / "stats/negatives_sensitivity.csv")
    cond = {"buffer_500m": "Random, 500\\,m buffer (baseline)", "buffer_1km": "Random, 1\\,km buffer",
            "buffer_2km": "Random, 2\\,km buffer", "slope_matched": "Slope-matched"}
    lo, hi = d.lift_pp.min(), d.lift_pp.max()
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{Sensitivity of the $K=1$ meta-learning advantage to non-landslide (negative) sample selection, under spatial "
             r"cross-validation (mean $F_1$ over three seeds; source-only features). Target negatives are re-generated by random sampling outside "
             r"exclusion buffers of 500\,m, 1\,km and 2\,km, and by a slope-matched strategy whose negatives reproduce the positives' slope "
             r"distribution (removing the steep-vs-flat confound). The meta-learned prior (Reptile) reuses the unchanged source meta-model. "
             r"Features are re-extracted from rasters, so absolute $F_1$ differs slightly from Table~\ref{tab:spatial_results}. Absolute $F_1$ "
             r"shifts with the sampling choice (consistent with \citealt{abdelkader2025nonlandslide}), but the meta-learning \emph{lift} stays "
             f"positive throughout (range ${lo:+.1f}$ to ${hi:+.1f}$\\,pp) and \\emph{{increases}} under the slope-matched condition. "
             r"Source: \texttt{results/stats/negatives\_sensitivity.csv}.}",
             r"    \label{tab:neg_sensitivity}", r"    \footnotesize", r"    \begin{tabular}{llccc}", r"        \toprule",
             r"        \textbf{Target} & \textbf{Negative-sampling strategy} & \textbf{Indep.\ $F_1$} & \textbf{Reptile $F_1$} & \textbf{Lift (pp)} \\",
             r"        \midrule"]
    for ti, t in enumerate([x for x in TARGETS if x in set(d.target)]):
        sub = d[d.target == t]; first = True
        for _, r in sub.iterrows():
            lab = f"{LAB[t]} ($N={int(r.n_pos)}$)" if first else ""
            lines.append(f"        {lab} & {cond.get(r.condition, r.condition)} & {r.f1_independent:.3f} & {r.f1_reptile:.3f} & ${r.lift_pp:+.1f}$ \\\\")
            first = False
        if ti < len(set(d.target)) - 1:
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "table04_negatives_sensitivity.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- supplementary
def tableS_basins():
    d = pd.read_csv(R / "stats/basin_summary.csv")
    d = pd.concat([d[d.role == "source"].sort_values("pos_pixels", ascending=False), d[d.role == "target"].sort_values("basin")])
    def esc(x): return str(x).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{Basin-by-basin inventory and sampling summary (reviewer request). Positive pixels are inventory points snapped "
             r"to the 30\,m grid (Taltal: polygons rasterised, $\leq 30$ pixels per polygon). Explicit negatives exist in two basins only "
             r"(Cha\~naral: 910 mapped non-landslide points; Taltal: 300 points expanded by a 60\,m buffer and completed with random samples); "
             r"the other twelve basins use random sampling outside a 500\,m exclusion buffer around positives. All basins are balanced 1:1. "
             r"`Dominant trigger' is taken from the inventory source and is indicative only. Source: \texttt{results/stats/basin\_summary.csv}.}",
             r"    \label{tab:basin_summary}", r"    \scriptsize", r"    \begin{tabular}{llp{3.6cm}rrrrp{3.0cm}r}", r"        \toprule",
             r"        \textbf{Basin} & \textbf{Role} & \textbf{Inventory} & \textbf{Events} & \textbf{Expl.\ neg.} & \textbf{Pos.\ px} & \textbf{Neg.\ px} & \textbf{Dominant trigger} & \textbf{Share src.\ pos.} \\",
             r"        \midrule"]
    for _, r in d.iterrows():
        share = f"{r.share_of_source_positives:.3f}" if pd.notna(r.share_of_source_positives) else "---"
        lines.append(f"        {LAB.get(r.basin, esc(r.basin.replace('_', ' ').title()))} & {r.role} & {esc(r.inventory_source)} & "
                     f"{int(r.n_inventory_positives)} & {int(r.n_explicit_negative_points)} & {int(r.pos_pixels)} & {int(r.neg_pixels)} & "
                     f"{esc(r.dominant_trigger)} & {share} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "tableS_basin_summary.tex").write_text("\n".join(lines) + "\n")


def tableS_folds():
    v = pd.read_csv(R / "stats/fold_validity.csv"); v = v[v.K == 1]
    ov = pd.read_csv(R / "stats/fold_overlap.csv")
    nn = pd.read_csv(R / "stats/fold_nn_distance.csv")
    lines = [r"\begin{table*}[ht]", r"    \centering",
             r"    \caption{Spatial-CV partition diagnostics per target basin (K-means, 5 folds, seeds 42/123/7/1729/2718 in the main "
             r"benchmark; diagnostics computed on the three original seeds). `Valid' partitions at $K=1$ are those whose held-out cluster contains "
             r"both classes; all exclusions are single-class held-out clusters. ARI is the adjusted Rand index between the partitions of different "
             r"seeds (1 = identical partitions). NN distance is the distance from each held-out pixel to the nearest pixel of the training clusters, "
             r"summarised over valid partitions. Source: \texttt{results/stats/fold\_validity.csv}, \texttt{fold\_overlap.csv}, \texttt{fold\_nn\_distance.csv}.}",
             r"    \label{tab:fold_diagnostics}", r"    \footnotesize", r"    \begin{tabular}{lcccccc}", r"        \toprule",
             r"        \textbf{Target} & \textbf{Valid / total} & \textbf{Excluded (single-class)} & \textbf{ARI across seeds} & "
             r"\textbf{NN min (km)} & \textbf{NN p5 (km)} & \textbf{NN median (km)} \\", r"        \midrule"]
    for t in TARGETS:
        vt = v[v.target == t]; valid = vt[vt.valid]
        ok = nn.merge(valid[["target", "seed", "fold"]], on=["target", "seed", "fold"])
        a = ov[ov.target == t].ari
        lines.append(f"        {LAB[t]} & {int(vt.valid.sum())}/{len(vt)} & {int((~vt.valid).sum())} & "
                     f"{a.mean():.2f} [{a.min():.2f}, {a.max():.2f}] & {ok.nn_min_km.min():.1f} & {ok.nn_p05_km.median():.1f} & {ok.nn_median_km.median():.1f} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    (SEC / "tableS_fold_diagnostics.tex").write_text("\n".join(lines) + "\n")


def tableS_cfs():
    c = pd.read_csv(R / "stats/cfs_comparison.csv")
    lines = [r"\begin{table}[ht]", r"    \centering",
             r"    \caption{Correlation-based feature selection run on all 14 basins (the submitted version, which used target-basin labels) "
             r"versus on the 10 source basins only (this revision). Step = order of entry in the forward search; SU = symmetric uncertainty "
             r"with the class label. The source-only run selects 9 of the original 11 features; geomorphons and annual precipitation (bio\_12) "
             r"are not selected. Source: \texttt{results/stats/cfs\_comparison.csv}.}",
             r"    \label{tab:cfs_comparison}", r"    \footnotesize", r"    \begin{tabular}{lcccc}", r"        \toprule",
             r"        \textbf{Feature} & \textbf{Step (all 14)} & \textbf{SU (all 14)} & \textbf{Step (source 10)} & \textbf{SU (source 10)} \\",
             r"        \midrule"]
    for _, r in c.iterrows():
        f = r.feature.replace("_", "\\_")
        lines.append(f"        \\texttt{{{f}}} & {r.step_all14} & {r.su_cf_all14} & {r.step_source10} & {r.su_cf_source10} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    (SEC / "tableS_cfs_comparison.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    table01(); table02(); table03(); table04(); table05()
    tableS_basins(); tableS_folds(); tableS_cfs()
    for f in ["table01_spatial_results", "table02_extended_baselines", "table03_cluster_paired", "table04_negatives_sensitivity", "table05_pooled_vs_episodic"]:
        print(f"wrote {SEC / (f + '.tex')}")
