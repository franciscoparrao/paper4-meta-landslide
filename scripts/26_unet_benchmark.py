#!/usr/bin/env python3
"""Paper 4 — Spatial CNN (U-Net-style) meta-learning baseline (ISPRS JPRS R2).

Tests whether spatial deep learning on multi-channel image patches outperforms the
point-wise MLP on landslide susceptibility transfer. Same K-shot spatial-CV protocol,
meta-trained on 10 source basins, evaluated on Huasco (largest target inventory).

Methods compared:
  - Independent CNN: trained from scratch on K target patches
  - Fine-tune CNN: concat-source pretrain + K-shot fine-tune
  - Reptile CNN: episodic meta-train + K-shot adapt
  - FOMAML CNN: query-loss meta-train + K-shot adapt

Output:
  - results/unet_baseline/raw_runs.csv
  - results/unet_baseline/summary.csv
  - results/unet_baseline/mlp_vs_cnn_huasco.csv (head-to-head comparison)
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _cnn_lib import (
    DEVICE, SpatialCNN, evaluate_query_cnn, fit_inner_cnn, fomaml_train_cnn,
    load_patches, pretrain_concat_cnn, reptile_train_cnn,
)
from _meta_lib import bootstrap_ci

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "results" / "unet_baseline"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo"]  # magallanes excluded (raster too big for in-memory stack)
TARGET = ["huasco"]   # primary demonstration; extend if compute permits
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
K_FOLDS = 5
N_EPISODES = 15        # slightly lower than MLP (15 vs 20) to control CNN compute

META_OUTER = 200       # CNN meta-train heavier than MLP; tuned for runtime
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-3
META_OUTER_LR = 5e-4
REPTILE_EPS = 0.1
PRETRAIN_STEPS = 200
PRETRAIN_LR = 5e-4
ADAPT_LR = 1e-3
ADAPT_STEPS = 30
N_QUERY_SOURCE = 10


def load_target_with_coords(basin: str):
    X, y = load_patches(basin)
    with h5py.File(SAMPLES / f"{basin}.h5", "r") as f:
        xu = f["x_utm"][:]; yu = f["y_utm"][:]
    coords = np.stack([xu, yu], axis=1)
    if len(coords) != len(X):
        # trim to common length (defensive: patches list may skip border points)
        m = min(len(coords), len(X)); X = X[:m]; y = y[:m]; coords = coords[:m]
    return X, y, coords


def make_folds(coords, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(coords)


def run_one_seed(seed, src_data, tgt_patches):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_in = next(iter(src_data.values()))[0].shape[1]
    print(f"[seed={seed}] pretrain (concat)...", flush=True)
    pre = pretrain_concat_cnn(src_data, n_in, PRETRAIN_STEPS, PRETRAIN_LR)
    pre_state = deepcopy(pre.state_dict())
    print(f"[seed={seed}] reptile meta-train...", flush=True)
    rep = reptile_train_cnn(src_data, n_in, META_OUTER, META_K,
                            META_INNER_STEPS, META_INNER_LR, REPTILE_EPS, rng)
    rep_state = deepcopy(rep.state_dict())
    print(f"[seed={seed}] fomaml meta-train...", flush=True)
    fom = fomaml_train_cnn(src_data, n_in, META_OUTER, META_K, N_QUERY_SOURCE,
                            META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng)
    fom_state = deepcopy(fom.state_dict())

    rows = []
    for tgt in TARGET:
        X, y, coords = tgt_patches[tgt]
        folds = make_folds(coords, K_FOLDS, seed)
        for fold in range(K_FOLDS):
            test_mask = folds == fold; train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            if len(np.unique(y[test_idx])) < 2:
                continue
            for K in KS:
                train_idx = np.where(train_mask)[0]
                pos = train_idx[y[train_idx] == 1]
                neg = train_idx[y[train_idx] == 0]
                if len(pos) < K or len(neg) < K:
                    continue
                rng_ep = np.random.default_rng(seed * 1000 + fold * 100 + K)
                for ep in range(N_EPISODES):
                    sup = np.concatenate([
                        rng_ep.choice(pos, K, replace=False),
                        rng_ep.choice(neg, K, replace=False),
                    ])
                    Xs = X[sup]; ys = y[sup]
                    Xq = X[test_idx]; yq = y[test_idx]
                    for method, init in [("independent_cnn", None),
                                          ("finetune_cnn", pre_state),
                                          ("reptile_cnn", rep_state),
                                          ("fomaml_cnn", fom_state)]:
                        m = SpatialCNN(n_in).to(DEVICE)
                        if init is not None:
                            m.load_state_dict(init)
                        m = fit_inner_cnn(m, Xs, ys, ADAPT_LR, ADAPT_STEPS)
                        r = evaluate_query_cnn(m, Xq, yq)
                        rows.append({"target": tgt, "fold": fold, "K": K,
                                     "method": method, "seed": seed, "episode": ep,
                                     "f1": r["f1"], "auc": r["auc"],
                                     "n_test": int(test_mask.sum())})
            print(f"[seed={seed}] {tgt} fold={fold} done", flush=True)
    return rows


def main():
    import os
    n_jobs = int(os.environ.get("PAPER4_N_JOBS", "-1"))
    print("Loading source patches...")
    src_data = {b: load_patches(b) for b in SOURCE}
    print(f"  sources: {[(b, X.shape) for b, (X, _) in src_data.items()]}")
    print("Loading target patches + coords...")
    tgt_patches = {b: load_target_with_coords(b) for b in TARGET}
    n_in = next(iter(src_data.values()))[0].shape[1]
    print(f"  n_channels = {n_in}, patch size = "
          f"{next(iter(src_data.values()))[0].shape[-1]}")

    print(f"\nLaunching {len(SEEDS)} seeds in parallel (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=min(n_jobs if n_jobs > 0 else len(SEEDS), len(SEEDS)),
                      backend="loky", verbose=10)(
        delayed(run_one_seed)(s, src_data, tgt_patches) for s in SEEDS
    )
    rows = [r for chunk in results for r in chunk]
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_runs.csv", index=False)
    print(f"\nRaw -> {OUT / 'raw_runs.csv'} ({len(raw)} rows)")

    summary = []
    for (t, K, m), g in raw.groupby(["target", "K", "method"]):
        f1m, f1lo, f1hi = bootstrap_ci(g["f1"].values)
        aum, aulo, auhi = bootstrap_ci(g["auc"].values)
        summary.append({"target": t, "K": K, "method": m, "n_runs": len(g),
                        "f1_mean": f1m, "f1_ci_lo": f1lo, "f1_ci_hi": f1hi,
                        "auc_mean": aum, "auc_ci_lo": aulo, "auc_ci_hi": auhi})
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(f"Summary -> {OUT / 'summary.csv'}")

    # Head-to-head with MLP main benchmark for Huasco
    mlp_path = ROOT / "results" / "spatial_benchmark" / "summary.csv"
    if mlp_path.exists():
        mlp = pd.read_csv(mlp_path)
        cnn = pd.DataFrame(summary)
        h2h = []
        for K in KS:
            mlp_row = mlp[(mlp.target == "huasco") & (mlp.K == K)
                          & (mlp.method.isin(["independent", "finetune",
                                              "reptile", "fomaml"]))]
            cnn_row = cnn[(cnn.target == "huasco") & (cnn.K == K)]
            for _, m in mlp_row.iterrows():
                method_short = m["method"]
                cm = cnn_row[cnn_row.method == f"{method_short}_cnn"]
                if cm.empty:
                    continue
                h2h.append({"K": K, "method": method_short,
                            "mlp_f1": m["f1_mean"], "cnn_f1": cm["f1_mean"].iloc[0],
                            "delta_pp": (cm["f1_mean"].iloc[0] - m["f1_mean"]) * 100})
        pd.DataFrame(h2h).to_csv(OUT / "mlp_vs_cnn_huasco.csv", index=False)
        print(f"Head-to-head -> {OUT / 'mlp_vs_cnn_huasco.csv'}")


if __name__ == "__main__":
    main()
