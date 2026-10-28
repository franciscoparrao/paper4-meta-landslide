#!/usr/bin/env python3
"""Paper 4 (R1) — domain weighting vs episodic training: a 2 x 2 control
(reviewer 3, comment 4; reviewer 2, comment 4).

The submitted benchmark confounds two differences between Fine-tune and the
meta-learners: (a) Fine-tune pre-trains on the size-weighted concatenation of
the source pool (Taltal = 49 % of source positives), whereas Reptile/FOMAML
sample source basins uniformly as tasks; (b) pooled vs episodic optimisation.
This script evaluates, on identical spatial partitions and identical support
draws (deterministic per-episode seeds), five initialisations:

    independent            no source information (reference)
    finetune               concat pre-training, size-weighted         (as submitted)
    finetune_balanced      concat pre-training, basin-balanced loss   (NEW)
    reptile                episodic from scratch, uniform basin tasks (as submitted)
    pretrain_then_reptile  concat pre-training, then episodic Reptile (NEW, R2 c.4)

All share the MLP backbone, 300 pre-training/outer steps, and 30 Adam
adaptation steps on the K-shot support. Spatial-CV protocol identical to
16_spatial_benchmark.py (K-means 5 folds, seeds 42/123/7, 20 episodes/fold,
query = full held-out cluster), except that the per-episode RNG seed uses a
deterministic hash (zlib.crc32) so that every method sees the same supports.

Outputs: results/pooled_vs_episodic/raw_runs.csv, summary.csv,
         cluster_paired.csv (per-cluster paired differences, hierarchical CI,
         Wilcoxon and sign test) for the contrasts that isolate each factor:
           finetune_balanced - finetune        (weighting, pooled)
           reptile - finetune_balanced         (episodic vs pooled, both balanced)
           pretrain_then_reptile - finetune    (adding episodic phase to pooled)
           reptile - independent, finetune - independent, ... (vs reference)
"""
from __future__ import annotations

import sys
import zlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, wilcoxon
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (  # noqa: E402
    DEVICE, MLP, bootstrap_ci, evaluate_query, fit_inner, load_basin,
    load_features, pretrain_concat, read_h5, reptile_train, standardize_basin,
)

ROOT = Path("/mnt/kingston/proyectos/postdoc/papers/paper4_meta_learning_transfer")
import argparse, os
_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="", help="suffix for the output directory (e.g. v1_cfs_all14)")
_ARGS = _ap.parse_args()
ML = Path(os.environ.get("P4_ML_DIR", str(ROOT / "data" / "ml_ready")))
OUT = ROOT / "results" / ("pooled_vs_episodic" + (f"_{_ARGS.tag}" if _ARGS.tag else ""))
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7, 1729, 2718]  # R1: five seeds (as the extended baselines) to average meta-training variance
K_FOLDS = 5
N_EPISODES_PER_FOLD = 20
META_OUTER, META_K, META_INNER_STEPS, META_INNER_LR, REPTILE_EPS = 300, 10, 5, 1e-2, 0.1
PRETRAIN_STEPS, PRETRAIN_LR = 300, 1e-3
ADAPT_LR, ADAPT_STEPS = 1e-2, 30
METHODS = ["independent", "finetune", "finetune_balanced", "reptile", "pretrain_then_reptile"]
N_BOOT = 2000


# ----------------------------------------------------------------------------
def pretrain_balanced(src_data, n_in, steps, lr):
    """Same as pretrain_concat (full-batch Adam, 300 steps, pooled z-scoring)
    but every source basin receives equal total weight in the loss:
    w_i = 1 / (n_basins * n_samples(basin_i))."""
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    w = np.concatenate([np.full(len(y), 1.0 / (len(src_data) * len(y)), dtype=np.float32)
                        for _, y in src_data.values()])
    w = w / w.sum() * len(w)  # mean weight 1, so the loss scale matches concat
    Xs_std = standardize_basin(Xs).astype(np.float32)
    model = MLP(n_in).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt, wt = torch.from_numpy(Xs_std), torch.from_numpy(ys), torch.from_numpy(w)
    for _ in range(steps):
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt), yt, weight=wt)
        loss.backward()
        optim.step()
    return model


def reptile_from(init_state, src_data, n_in, outer_steps, K, inner_steps, inner_lr, eps, rng):
    """Reptile meta-training initialised from a given state (pooled -> episodic)."""
    meta = MLP(n_in).to(DEVICE)
    meta.load_state_dict(init_state)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        sup = np.concatenate([rng.choice(pos, size=min(K, len(pos)), replace=False),
                              rng.choice(neg, size=min(K, len(neg)), replace=False)])
        inner = deepcopy(meta)
        inner = fit_inner(inner, X[sup], y[sup], lr=inner_lr, steps=inner_steps)
        with torch.no_grad():
            for pm, pi in zip(meta.parameters(), inner.parameters()):
                pm.add_(eps * (pi - pm))
    return meta


def load_target_with_coords(basin, features):
    df = read_h5(ML / f"{basin}.h5")
    return (df[features].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=np.float32),
            df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64))


def make_spatial_folds(coords, n_folds, seed):
    return KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(coords)


def episode_seed(seed, fold, K, tgt):
    return seed * 1000 + fold * 100 + K + zlib.crc32(tgt.encode()) % 100


def spatial_kshot_episode(init_state, X_norm, y, train_mask, test_mask, K, n_in, rng):
    train_idx = np.flatnonzero(train_mask)
    pos_train = train_idx[y[train_idx] == 1]; neg_train = train_idx[y[train_idx] == 0]
    test_idx = np.flatnonzero(test_mask)
    Kp, Kn = min(K, len(pos_train)), min(K, len(neg_train))
    if Kp == 0 or Kn == 0 or test_idx.size == 0:
        return None
    sup = np.concatenate([rng.choice(pos_train, size=Kp, replace=False),
                          rng.choice(neg_train, size=Kn, replace=False)])
    if len(np.unique(y[test_idx])) < 2:
        return None
    model = MLP(n_in).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = fit_inner(model, X_norm[sup], y[sup], lr=ADAPT_LR, steps=ADAPT_STEPS)
    return evaluate_query(model, X_norm[test_idx], y[test_idx])


# ----------------------------------------------------------------------------
def cluster_paired(raw, target, K, a, b, rng):
    sub = raw[(raw.target == target) & (raw.K == K)]
    ma = sub[sub.method == a].groupby(["seed", "fold"])["f1"].mean()
    mb = sub[sub.method == b].groupby(["seed", "fold"])["f1"].mean()
    common = ma.index.intersection(mb.index)
    d = (ma.loc[common] - mb.loc[common]).to_numpy()
    nc = len(d)
    if nc == 0:
        return None
    boot = d[rng.integers(0, nc, size=(N_BOOT, nc))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        wp = float(wilcoxon(d, alternative="two-sided").pvalue) if nc >= 2 else np.nan
    except ValueError:
        wp = np.nan
    nz = d[d != 0]
    sp = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else np.nan
    return {"target": target, "K": K, "contrast": f"{a} - {b}", "n_clusters": nc,
            "mean_diff_pp": d.mean() * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
            "ci_excludes_0": bool(lo > 0 or hi < 0), "n_positive": int((d > 0).sum()),
            "wilcoxon_p": wp, "sign_p": sp}


def main():
    features = load_features()
    n_in = len(features)
    print(f"features ({n_in}): {features}")
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    tgt_data = {b: load_target_with_coords(b, features) for b in TARGET}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n===== seed={seed} =====", flush=True)
        print("  pretrain concat (size-weighted)...", flush=True)
        pre_state = deepcopy(pretrain_concat(src_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())
        print("  pretrain balanced (basin-weighted)...", flush=True)
        bal_state = deepcopy(pretrain_balanced(src_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())
        print("  reptile from scratch...", flush=True)
        rep_state = deepcopy(reptile_train(src_std, n_in, META_OUTER, META_K, META_INNER_STEPS,
                                           META_INNER_LR, REPTILE_EPS, rng).state_dict())
        print("  reptile from concat pre-training...", flush=True)
        ptr_state = deepcopy(reptile_from(pre_state, src_std, n_in, META_OUTER, META_K,
                                          META_INNER_STEPS, META_INNER_LR, REPTILE_EPS, rng).state_dict())
        states = {"independent": None, "finetune": pre_state, "finetune_balanced": bal_state,
                  "reptile": rep_state, "pretrain_then_reptile": ptr_state}

        for tgt, (X, y, coords) in tgt_data.items():
            X_std = standardize_basin(X)
            folds = make_spatial_folds(coords, K_FOLDS, seed)
            for fold in range(K_FOLDS):
                test_mask = folds == fold; train_mask = ~test_mask
                for K in KS:
                    if K * 2 > train_mask.sum():
                        continue
                    for method in METHODS:
                        rng_ep = np.random.default_rng(episode_seed(seed, fold, K, tgt))
                        for ep in range(N_EPISODES_PER_FOLD):
                            m = spatial_kshot_episode(states[method], X_std, y, train_mask,
                                                      test_mask, K, n_in, rng_ep)
                            if m is None:
                                continue
                            rows.append({"target": tgt, "fold": fold, "K": K, "method": method,
                                         "seed": seed, "episode": ep, "f1": m["f1"], "auc": m["auc"],
                                         "n_test": int(test_mask.sum())})
            print(f"  {tgt:<10} done ({len(rows)} rows so far)", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_runs.csv", index=False)
    summ = []
    for (tgt, K, method), g in raw.groupby(["target", "K", "method"]):
        f1m, f1lo, f1hi = bootstrap_ci(g["f1"].values)
        aucm, auclo, auchi = bootstrap_ci(g["auc"].dropna().values)
        summ.append({"target": tgt, "K": K, "method": method, "n_runs": len(g),
                     "f1_mean": f1m, "f1_ci_lo": f1lo, "f1_ci_hi": f1hi,
                     "auc_mean": aucm, "auc_ci_lo": auclo, "auc_ci_hi": auchi})
    summ = pd.DataFrame(summ); summ.to_csv(OUT / "summary.csv", index=False)

    rng = np.random.default_rng(42)
    contrasts = [("finetune_balanced", "finetune"), ("reptile", "finetune_balanced"),
                 ("pretrain_then_reptile", "finetune"), ("pretrain_then_reptile", "reptile"),
                 ("reptile", "independent"), ("finetune", "independent"),
                 ("finetune_balanced", "independent"), ("pretrain_then_reptile", "independent")]
    cp = [r for t in TARGET for K in KS for a, b in contrasts
          if (r := cluster_paired(raw, t, K, a, b, rng)) is not None]
    cp = pd.DataFrame(cp); cp.to_csv(OUT / "cluster_paired.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== F1 mean at K=1 (spatial CV) ==")
    print(summ[summ.K == 1].pivot(index="target", columns="method", values="f1_mean")[METHODS].round(3))
    print("\n== Cluster-paired contrasts at K=1 (pp) ==")
    print(cp[cp.K == 1][["target", "contrast", "n_clusters", "mean_diff_pp", "ci_lo_pp", "ci_hi_pp",
                         "n_positive", "wilcoxon_p"]].round(3).to_string(index=False))
    print(f"\nRaw → {OUT / 'raw_runs.csv'} ({len(raw)} rows)")


if __name__ == "__main__":
    main()
