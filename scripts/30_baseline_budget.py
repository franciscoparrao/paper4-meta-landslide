#!/usr/bin/env python3
"""Paper 4 — adaptation-budget fairness control (blind-review M2).

The main benchmark adapts every method for a fixed 30 inner Adam steps, while the
adaptation-curve experiment notes that Independent (random init) needs >100 steps
to plateau. A skeptic can then argue the K=1 lift partly reflects an under-converged
baseline. This control re-runs the spatial-CV K-shot protocol (identical to
16_spatial_benchmark.py) for Independent AND Reptile at inner-step budgets
{30, 100, 300}, K in {1, 5}, all four targets. If Independent does not improve
beyond 30 steps (overfitting on 2K support samples), the fixed budget is vindicated;
if it does, the lift must be restated against Independent-at-its-best.

Output: results/stats/baseline_budget.csv
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta_lib import (
    MLP, evaluate_query, fit_inner, load_basin, load_features, read_h5,
    reptile_train, standardize_basin,
)

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "stats"

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGETS = ["copiapo", "huasco", "elqui", "limari"]
SEEDS = [42, 123, 7]
KS = [1]
STEP_BUDGETS = [30, 100, 300]
K_FOLDS, N_EP = 5, 20
META_OUTER, META_K, META_INNER, META_LR, EPS = 300, 10, 5, 1e-2, 0.1
ADAPT_LR = 1e-2


def load_target(basin, features):
    df = read_h5(ML / f"{basin}.h5")
    return (df[features].to_numpy(dtype=np.float32),
            df["label"].to_numpy(dtype=np.float32),
            df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64))


def main():
    features = load_features()
    n_in = len(features)
    src_std = {b: (standardize_basin(load_basin(b, features)[0]), load_basin(b, features)[1])
               for b in SOURCE}
    rep_states = {}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"meta-training Reptile seed {seed}...", flush=True)
        rep_states[seed] = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER, META_LR, EPS,
            np.random.default_rng(seed)).state_dict())

    from joblib import Parallel, delayed

    def one_chunk(tgt, Xn, y, folds, seed, f, rep_state):
        torch.set_num_threads(1)
        out = []
        te_mask = folds == f
        tr = np.where(~te_mask)[0]
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        te = np.where(te_mask)[0]
        if len(np.unique(y[te])) < 2:
            return out
        for K in KS:
            if len(pos) < K or len(neg) < K:
                continue
            rng_ep = np.random.default_rng(seed * 1000 + f * 100 + K)
            for ep in range(N_EP):
                sup = np.concatenate([rng_ep.choice(pos, K, False),
                                      rng_ep.choice(neg, K, False)])
                for meth, st in [("independent", None), ("reptile", rep_state)]:
                    for steps in STEP_BUDGETS:
                        m = MLP(n_in)
                        if st is not None:
                            m.load_state_dict(st)
                        m = fit_inner(m, Xn[sup], y[sup], lr=ADAPT_LR, steps=steps)
                        r = evaluate_query(m, Xn[te], y[te])
                        out.append({"target": tgt, "K": K, "method": meth,
                                    "steps": steps, "seed": seed, "fold": f,
                                    "episode": ep, "f1": r["f1"]})
        return out

    jobs = []
    for tgt in TARGETS:
        X, y, coords = load_target(tgt, features)
        Xn = standardize_basin(X)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            folds = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
            for f in range(K_FOLDS):
                jobs.append((tgt, Xn, y, folds, seed, f, rep_states[seed]))
    print(f"{len(jobs)} chunks -> parallel", flush=True)
    chunks = Parallel(n_jobs=14, backend="loky", verbose=5)(
        delayed(one_chunk)(*j) for j in jobs)
    rows = [r for c in chunks for r in c]

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_budget_raw.csv", index=False)
    summ = (df.groupby(["target", "K", "method", "steps"])["f1"]
              .mean().reset_index().pivot_table(
                  index=["target", "K"], columns=["method", "steps"], values="f1"))
    summ.to_csv(OUT / "baseline_budget.csv")
    print("\n=== mean F1 by (method, inner steps) — spatial CV ===")
    print(summ.round(3).to_string())
    # headline: does Independent@best close the gap at K=1?
    print("\n=== K=1: Reptile@30 vs Independent@best-of-{30,100,300} ===")
    for tgt in TARGETS:
        sub = df[(df.target == tgt) & (df.K == 1)]
        if sub.empty:
            continue
        ind_best = sub[sub.method == "independent"].groupby("steps")["f1"].mean()
        rep30 = sub[(sub.method == "reptile") & (sub.steps == 30)]["f1"].mean()
        b = ind_best.idxmax()
        print(f"  {tgt:8s}: Indep best {ind_best.max():.3f} @ {b} steps "
              f"(30 steps: {ind_best[30]:.3f}) | Reptile@30 {rep30:.3f} "
              f"| lift vs Indep-best {(rep30 - ind_best.max())*100:+.1f} pp")


if __name__ == "__main__":
    main()
