#!/usr/bin/env python3
"""Paper 4 (R1) — preprocessing sensitivity at K=1 under spatial CV (reviewer 3, comment 3).

Four preprocessing variants, Independent and Reptile, three seeds, same partitions
and support draws (deterministic seeds):
    basin    : z-score per basin over all sampled pixels of that basin (as in the paper)
    source   : z-score with mean/sd estimated on the pooled source basins only
    support  : z-score with mean/sd of the 2K support pixels only (degenerate at K=1)
    onehot   : as `basin`, but the categorical `terrain__landform` is one-hot encoded
Outputs: results/stats/preproc_sensitivity.csv
"""
from __future__ import annotations

import sys
import zlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import DEVICE, MLP, evaluate_query, fit_inner, load_basin, load_features, read_h5, reptile_train  # noqa: E402

ROOT = Path("/mnt/kingston/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "stats"
SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
SEEDS = [42, 123, 7]
K, K_FOLDS, N_EP = 1, 5, 20
META_OUTER, META_K, META_INNER, META_LR, EPS = 300, 10, 5, 1e-2, 0.1
ADAPT_LR, ADAPT_STEPS = 1e-2, 30
CAT = "terrain__landform"
MODES = ["basin", "source", "support", "onehot"]


def zs(X, mu, sd):
    return ((X - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def onehot_transform(X, feats, classes):
    j = feats.index(CAT)
    cont = np.delete(X, j, axis=1)
    oh = (X[:, [j]] == classes[None, :]).astype(np.float32)
    return cont, oh


def main():
    feats = load_features()
    src_raw = {b: load_basin(b, feats) for b in SOURCE}
    tgt = {}
    for b in TARGET:
        df = read_h5(ML / f"{b}.h5")
        tgt[b] = (df[feats].to_numpy(np.float32), df["label"].to_numpy(np.float32), df[["x_utm", "y_utm"]].to_numpy(float))
    Xsrc_all = np.concatenate([X for X, _ in src_raw.values()])
    mu_src, sd_src = Xsrc_all.mean(0), Xsrc_all.std(0)
    classes = np.unique(np.concatenate([X[:, feats.index(CAT)] for X, _ in src_raw.values()] + [X[:, feats.index(CAT)] for X, _, _ in tgt.values()]))
    classes = classes[np.isfinite(classes)]
    print(f"features {len(feats)}; landform classes {len(classes)}")

    rows = []
    for seed in SEEDS:
        for mode in MODES:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed)
            # ---- source representation for meta-training
            if mode == "onehot":
                src_std = {}
                for b, (X, y) in src_raw.items():
                    cont, oh = onehot_transform(X, feats, classes)
                    src_std[b] = (np.concatenate([zs(cont, cont.mean(0), cont.std(0)), oh], 1), y)
                n_in = src_std[SOURCE[0]][0].shape[1]
            else:
                src_std = {b: (zs(X, X.mean(0), X.std(0)), y) for b, (X, y) in src_raw.items()}  # per-basin, as in paper
                n_in = len(feats)
            rep_state = deepcopy(reptile_train(src_std, n_in, META_OUTER, META_K, META_INNER, META_LR, EPS, rng).state_dict())

            for t, (X, y, coords) in tgt.items():
                folds = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
                if mode == "onehot":
                    cont, oh = onehot_transform(X, feats, classes)
                    X_mode = np.concatenate([zs(cont, cont.mean(0), cont.std(0)), oh], 1)
                elif mode == "basin":
                    X_mode = zs(X, X.mean(0), X.std(0))
                elif mode == "source":
                    X_mode = zs(X, mu_src, sd_src)
                else:
                    X_mode = None  # support: computed per episode
                for fold in range(K_FOLDS):
                    test = folds == fold; train = ~test
                    tr = np.flatnonzero(train); te = np.flatnonzero(test)
                    pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
                    if len(pos) < K or len(neg) < K or len(np.unique(y[te])) < 2:
                        continue
                    rng_ep = np.random.default_rng(seed * 1000 + fold * 100 + K + zlib.crc32(t.encode()) % 100)
                    for ep in range(N_EP):
                        sup = np.concatenate([rng_ep.choice(pos, K, replace=False), rng_ep.choice(neg, K, replace=False)])
                        if mode == "support":
                            Xm = zs(X, X[sup].mean(0), X[sup].std(0))
                        else:
                            Xm = X_mode
                        for method, state in [("independent", None), ("reptile", rep_state)]:
                            m = MLP(n_in).to(DEVICE)
                            if state is not None:
                                m.load_state_dict(state)
                            m = fit_inner(m, Xm[sup], y[sup], lr=ADAPT_LR, steps=ADAPT_STEPS)
                            r = evaluate_query(m, Xm[te], y[te])
                            rows.append({"mode": mode, "target": t, "seed": seed, "fold": fold, "episode": ep,
                                         "method": method, "f1": r["f1"], "auc": r["auc"]})
            print(f"seed {seed} mode {mode} done ({len(rows)} rows)", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "preproc_sensitivity_raw.csv", index=False)
    cm = raw.groupby(["mode", "target", "seed", "fold", "method"]).f1.mean().unstack("method")
    cm["lift"] = cm.reptile - cm.independent
    summ = cm.groupby(["mode", "target"]).agg(f1_independent=("independent", "mean"), f1_reptile=("reptile", "mean"),
                                              lift_pp=("lift", lambda d: d.mean() * 100), n_clusters=("lift", "size")).reset_index()
    summ.to_csv(OUT / "preproc_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print("\nmean over targets:"); print(summ.groupby("mode")[["f1_independent", "f1_reptile", "lift_pp"]].mean().round(3))


if __name__ == "__main__":
    main()
