#!/usr/bin/env python3
"""Paper 4 — meta-learning smoke test (Reptile vs baselines).

Validates the end-to-end pipeline on the 8 ML-ready basin datasets:
- 4 source basins for meta-training (chanaral, taltal, maule, choapa)
- 4 target basins for meta-test K-shot evaluation (copiapo, huasco, elqui, limari)

Methods compared:
  1. Independent: train MLP from scratch on K target samples (no source data).
  2. Fine-tune:   pretrain MLP on all source basins, fine-tune on K target samples.
  3. Reptile:     meta-train MLP on source basin episodes, adapt to K target samples.

Reports F1 / AUC on held-out target query set per K (K=5, 10, 50).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10]    # smaller K to fit tiny basins (copiapo N=38, limari N=70)
N_QUERY = 10        # query samples per class per episode (so 2*N_QUERY total)
N_EPISODES_EVAL = 30  # repetitions per K-shot eval (mean ± std)

DEVICE = torch.device("cpu")
SEED = 42


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def load_basin_arrays(basin: str, feature_cols: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y


def standardize(X_train, X_others):
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0) + 1e-6
    return (X_train - mu) / sd, [(X - mu) / sd for X in X_others]


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_inner(model, X, y, lr=1e-2, steps=20):
    """Simple Adam fit on one task. Used for adapt-time inner loop."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    for _ in range(steps):
        optim.zero_grad()
        logits = model(Xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        optim.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()


def sample_kshot(X, y, K, n_query, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    sel_pos = rng.choice(pos_idx, size=min(K, len(pos_idx)), replace=False)
    sel_neg = rng.choice(neg_idx, size=min(K, len(neg_idx)), replace=False)
    support = np.concatenate([sel_pos, sel_neg])
    rest_pos = np.setdiff1d(pos_idx, sel_pos)
    rest_neg = np.setdiff1d(neg_idx, sel_neg)
    nq = min(n_query, len(rest_pos), len(rest_neg))
    qpos = rng.choice(rest_pos, size=nq, replace=False)
    qneg = rng.choice(rest_neg, size=nq, replace=False)
    query = np.concatenate([qpos, qneg])
    return support, query


def evaluate(model, X, y):
    p = predict(model, X)
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
    }


def reptile_meta_train(source_data, n_in, outer_steps=200, K=10,
                       inner_steps=5, inner_lr=1e-2, eps=0.1, rng=None):
    """First-order Reptile (Nichol et al. 2018) on source basins."""
    rng = rng or np.random.default_rng(SEED)
    meta_model = MLP(n_in).to(DEVICE)
    for step in range(outer_steps):
        basin = rng.choice(list(source_data.keys()))
        X, y = source_data[basin]
        sup, _ = sample_kshot(X, y, K, n_query=10, rng=rng)
        inner = deepcopy(meta_model)
        inner = fit_inner(inner, X[sup], y[sup], lr=inner_lr, steps=inner_steps)
        # Reptile update: theta <- theta + eps * (theta_inner - theta)
        with torch.no_grad():
            for p_meta, p_inner in zip(meta_model.parameters(), inner.parameters()):
                p_meta.add_(eps * (p_inner - p_meta))
    return meta_model


def evaluate_kshot(method, target_basin, X, y, K, n_episodes, rng,
                   pretrained_state=None, n_in=14):
    # Standardize using FULL basin statistics (stable with tiny K)
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-6)
    Xn = (X - mu) / sd
    f1s, aucs = [], []
    for ep in range(n_episodes):
        sup, qry = sample_kshot(X, y, K, N_QUERY, rng)
        Xs_s, ys = Xn[sup], y[sup]
        Xq_s, yq = Xn[qry], y[qry]

        if method == "independent":
            model = MLP(n_in).to(DEVICE)
        elif method in ("finetune", "reptile"):
            model = MLP(n_in).to(DEVICE)
            model.load_state_dict(pretrained_state)
        else:
            raise ValueError(method)

        model = fit_inner(model, Xs_s, ys, lr=1e-2, steps=30 if method == "independent" else 20)
        m = evaluate(model, Xq_s, yq)
        f1s.append(m["f1"]); aucs.append(m["auc"])
    return {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Load aligned feature manifest
    feat_path = ML / "../aligned/cfs_selected_features.txt"
    features = [l.strip() for l in feat_path.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    print(f"Features ({len(features)}): {features}")

    # Load all basins
    print("\nLoading basins...")
    source_data = {b: load_basin_arrays(b, features) for b in SOURCE}
    target_data = {b: load_basin_arrays(b, features) for b in TARGET}
    for b, (X, y) in {**source_data, **target_data}.items():
        print(f"  {b:<10} X={X.shape}  y_pos={int(y.sum())}/{len(y)}")

    # ---- Pretrain (concat all source) for fine-tune baseline ----
    print("\n=== Pretraining MLP on all source basins (concat) ===")
    Xs_all = np.concatenate([X for X, _ in source_data.values()], axis=0)
    ys_all = np.concatenate([y for _, y in source_data.values()], axis=0)
    mu = Xs_all.mean(0); sd = Xs_all.std(0) + 1e-6
    Xs_all_s = (Xs_all - mu) / sd
    pre_model = MLP(len(features)).to(DEVICE)
    pre_model = fit_inner(pre_model, Xs_all_s, ys_all, lr=1e-3, steps=300)
    pretrained_state = deepcopy(pre_model.state_dict())

    # ---- Meta-train Reptile ----
    print("\n=== Meta-training Reptile (200 outer steps) ===")
    # Standardize source data per-basin (will re-standardize at adapt time)
    src_for_meta = {b: ((X - X.mean(0)) / np.maximum(X.std(0), 1e-6), y)
                    for b, (X, y) in source_data.items()}
    reptile_model = reptile_meta_train(src_for_meta, n_in=len(features),
                                       outer_steps=200, K=10, inner_steps=5,
                                       inner_lr=1e-2, eps=0.1, rng=rng)
    reptile_state = deepcopy(reptile_model.state_dict())

    # ---- K-shot evaluation on targets ----
    print(f"\n=== K-shot evaluation on target basins ({N_EPISODES_EVAL} episodes/K) ===")
    rows = []
    for tgt, (X, y) in target_data.items():
        for K in KS:
            need = K * 2 + N_QUERY * 2
            if need > len(X):
                print(f"  [skip] {tgt} K={K} (need {need}, have {len(X)})")
                continue
            for method, state in [("independent", None),
                                  ("finetune", pretrained_state),
                                  ("reptile",  reptile_state)]:
                rng_eval = np.random.default_rng(SEED + hash(method + tgt) % 1000)
                m = evaluate_kshot(method, tgt, X, y, K, N_EPISODES_EVAL,
                                  rng_eval, pretrained_state=state,
                                  n_in=len(features))
                rows.append({"target": tgt, "K": K, "method": method, **m})
                print(f"  {tgt:<10} K={K:<3} {method:<12} "
                      f"F1={m['f1_mean']:.3f}±{m['f1_std']:.3f}  "
                      f"AUC={m['auc_mean']:.3f}±{m['auc_std']:.3f}")

    # Save results
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "smoke_results.csv", index=False)
    print(f"\nResults → {OUT / 'smoke_results.csv'}")

    # Summary pivot
    pivot = res_df.pivot_table(index=["target", "K"], columns="method", values="f1_mean")
    print("\n=== F1 (mean) by method × target × K ===")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
