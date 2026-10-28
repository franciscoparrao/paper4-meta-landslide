#!/usr/bin/env python3
"""Paper 4 — prevalence-robust evaluation of the K=1 spatial-CV benchmark (A2).

The main benchmark reports F1 on *balanced* (1:1) queries at threshold 0.5, which
is not the F1 a practitioner sees under the extreme class imbalance of operational
susceptibility mapping. This script re-runs the K=1 spatial-CV protocol (identical
to 16_spatial_benchmark.py) for the four targets, pools the held-out query
predictions, and reports prevalence-robust quantities:

  - ROC-AUC (threshold- and prevalence-independent).
  - Operating characteristics at threshold 0.5: TPR (recall) and FPR, which are
    prevalence-independent.
  - Precision at a realistic landslide prevalence pi, computed exactly from the
    balanced operating point:  precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR).
  - Average precision (area under the precision-recall curve) re-weighted to pi.

Reference prevalences: pi = 0.5 (balanced, as in the main benchmark) down to
pi = 0.01. The raw pixel-level inventory prevalence is far lower still (~1e-4 to
1e-5); susceptibility maps are used to stratify terrain, not to classify single
pixels, so pi in {0.01..0.10} brackets the operational regime.

Output:
- results/stats/prevalence_robust.csv
- results/stats/pr_curves_pi.npz   (recall, precision@pi arrays for the figure)
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import zlib  # deterministic per-episode seeds (R1)
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta_lib import (
    MLP, fit_inner, load_basin, load_features, predict_proba, read_h5,
    reptile_train, standardize_basin,
)

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "stats"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
SEEDS = [42, 123, 7]
K = 1
K_FOLDS = 5
N_EP = 20
PREVALENCES = [0.5, 0.10, 0.05, 0.01]
PI_FIG = 0.05  # prevalence for the PR-curve figure

# Match 16_spatial_benchmark.py exactly
META_OUTER, META_K, META_INNER, META_LR, EPS = 300, 10, 5, 1e-2, 0.1
ADAPT_LR, ADAPT_STEPS = 1e-2, 30


def load_target(basin, features):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def episode_preds(init_state, Xn, y, train_mask, test_mask, n_in, rng):
    """One K=1 episode; return (y_query, p_query) instead of scalar metrics."""
    tr = np.where(train_mask)[0]
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    te = np.where(test_mask)[0]
    if len(pos) == 0 or len(neg) == 0 or te.size == 0:
        return None
    sup = np.concatenate([rng.choice(pos, K, replace=False),
                          rng.choice(neg, K, replace=False)])
    yte = y[te]
    if len(np.unique(yte)) < 2:
        return None
    m = MLP(n_in)
    if init_state is not None:
        m.load_state_dict(init_state)
    m = fit_inner(m, Xn[sup], y[sup], lr=ADAPT_LR, steps=ADAPT_STEPS)
    return yte, predict_proba(m, Xn[te])


def prevalence_metrics(y, p, prevalences):
    """AUC, operating point at 0.5, and precision/AP re-weighted to each pi."""
    y = np.asarray(y); p = np.asarray(p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    # operating point at threshold 0.5 (as in the benchmark)
    pred = (p >= 0.5).astype(int)
    P, N = (y == 1).sum(), (y == 0).sum()
    tpr = ((pred == 1) & (y == 1)).sum() / max(P, 1)   # recall
    fpr = ((pred == 1) & (y == 0)).sum() / max(N, 1)
    prec_bal = ((pred == 1) & (y == 1)).sum() / max((pred == 1).sum(), 1)
    f1_bal = 2 * prec_bal * tpr / max(prec_bal + tpr, 1e-12)
    out = {"auc": auc, "tpr_recall": tpr, "fpr": fpr,
           "precision_balanced": prec_bal, "f1_balanced": f1_bal}
    # precision at threshold 0.5 for each realistic prevalence
    for pi in prevalences:
        denom = pi * tpr + (1 - pi) * fpr
        out[f"precision_pi{pi:g}"] = (pi * tpr / denom) if denom > 0 else np.nan
    # PR curve re-weighted to PI_FIG + average precision at each pi
    thr = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    rec, prec_pi = [], []
    for t in thr[::-1]:
        pr = (p >= t)
        tp_r = (pr & (y == 1)).sum() / max(P, 1)
        fp_r = (pr & (y == 0)).sum() / max(N, 1)
        d = PI_FIG * tp_r + (1 - PI_FIG) * fp_r
        rec.append(tp_r)
        prec_pi.append((PI_FIG * tp_r / d) if d > 0 else 1.0)
    rec, prec_pi = np.array(rec), np.array(prec_pi)
    # average precision (step integral) at each pi
    for pi in prevalences:
        pr_curve = []
        for t in thr[::-1]:
            pr = (p >= t)
            tp_r = (pr & (y == 1)).sum() / max(P, 1)
            fp_r = (pr & (y == 0)).sum() / max(N, 1)
            d = pi * tp_r + (1 - pi) * fp_r
            pr_curve.append((pi * tp_r / d) if d > 0 else 1.0)
        r = np.array(rec)
        ap = np.sum(np.diff(np.concatenate([[0], r])) * np.array(pr_curve))
        out[f"ap_pi{pi:g}"] = ap
    return out, rec, prec_pi


def main():
    features = load_features()
    n_in = len(features)
    src_std = {b: (standardize_basin(load_basin(b, features)[0]), load_basin(b, features)[1])
               for b in SOURCE}

    # pooled predictions per (target, method)
    pool = {(t, m): {"y": [], "p": []} for t in TARGETS for m in ["independent", "reptile"]}

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"seed {seed}: meta-training Reptile...")
        rep_state = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER, META_LR, EPS, rng).state_dict())
        for t in TARGETS:
            X, y, coords = load_target(t, features)
            if len(X) < K_FOLDS * 4:
                continue
            Xn = standardize_basin(X)
            folds = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
            for f in range(K_FOLDS):
                test_mask = folds == f
                train_mask = ~test_mask
                for meth, state in [("independent", None), ("reptile", rep_state)]:
                    rng_ep = np.random.default_rng(seed * 1000 + f * 100 + zlib.crc32((meth + t).encode()) % 100)
                    for _ in range(N_EP):
                        r = episode_preds(state, Xn, y, train_mask, test_mask, n_in, rng_ep)
                        if r is None:
                            continue
                        pool[(t, meth)]["y"].append(r[0])
                        pool[(t, meth)]["p"].append(r[1])

    rows, curves = [], {}
    for (t, meth), d in pool.items():
        if not d["y"]:
            continue
        y = np.concatenate(d["y"]); p = np.concatenate(d["p"])
        met, rec, prec_pi = prevalence_metrics(y, p, PREVALENCES)
        rows.append({"target": t, "method": meth, "n_query": len(y),
                     "prevalence_pooled": float((y == 1).mean()), **met})
        curves[f"{t}_{meth}_rec"] = rec
        curves[f"{t}_{meth}_prec"] = prec_pi

    df = pd.DataFrame(rows).sort_values(["target", "method"])
    df.to_csv(OUT / "prevalence_robust.csv", index=False)
    np.savez(OUT / "pr_curves_pi.npz", pi=PI_FIG, **curves)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = ["target", "method", "auc", "f1_balanced", "tpr_recall", "fpr",
            "precision_balanced", "precision_pi0.1", "precision_pi0.05", "precision_pi0.01",
            f"ap_pi{PI_FIG:g}"]
    print("\n=== Prevalence-robust metrics (pooled K=1 spatial-CV queries) ===")
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {OUT/'prevalence_robust.csv'} and pr_curves_pi.npz (pi_fig={PI_FIG})")


if __name__ == "__main__":
    main()
