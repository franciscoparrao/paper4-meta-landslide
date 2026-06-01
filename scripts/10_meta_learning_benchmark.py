#!/usr/bin/env python3
"""Paper 4 — formal meta-learning benchmark (Reptile / FOMAML / fine-tune / independent).

Design:
- 3 random seeds for meta-training (42, 123, 7); each method trained 3 times.
- For each (target_basin, K, method, seed): run 50 K-shot evaluation episodes.
- Bootstrap 95% CI on F1 / AUC across all (seed, episode) pooled samples.

Output:
- results/benchmark/raw_runs.csv      (target, K, method, seed, episode, f1, auc)
- results/benchmark/summary.csv       (target, K, method, f1_mean, f1_ci_lo, f1_ci_hi, auc_*)
- results/benchmark/f1_table.csv      (pivot: rows=(target, K), cols=method, vals=f1_mean)
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
OUT = ROOT / "results" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
N_QUERY = 10
N_EPISODES = 50

# Meta-training config
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
REPTILE_EPS = 0.1

# Pretrain (for fine-tune baseline)
PRETRAIN_STEPS = 300
PRETRAIN_LR = 1e-3

# Adapt config (test-time)
ADAPT_STEPS = 30
ADAPT_LR = 1e-2

DEVICE = torch.device("cpu")


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def load_basin(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y


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


def fit_inner(model, X, y, lr, steps):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    for _ in range(steps):
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt), yt)
        loss.backward()
        optim.step()
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).cpu().numpy()


def sample_kshot(X, y, K, n_query, rng):
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    Kp = min(K, len(pos_idx)); Kn = min(K, len(neg_idx))
    sel_pos = rng.choice(pos_idx, size=Kp, replace=False)
    sel_neg = rng.choice(neg_idx, size=Kn, replace=False)
    sup = np.concatenate([sel_pos, sel_neg])
    rest_pos = np.setdiff1d(pos_idx, sel_pos); rest_neg = np.setdiff1d(neg_idx, sel_neg)
    nq = min(n_query, len(rest_pos), len(rest_neg))
    qpos = rng.choice(rest_pos, size=nq, replace=False)
    qneg = rng.choice(rest_neg, size=nq, replace=False)
    qry = np.concatenate([qpos, qneg])
    return sup, qry


def standardize_basin(X):
    mu = X.mean(0); sd = np.maximum(X.std(0), 1e-6)
    return (X - mu) / sd


def reptile_train(src_data, n_in, outer_steps, K, inner_steps, inner_lr, eps, rng):
    meta = MLP(n_in).to(DEVICE)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        sup, _ = sample_kshot(X, y, K, n_query=10, rng=rng)
        inner = deepcopy(meta)
        inner = fit_inner(inner, X[sup], y[sup], lr=inner_lr, steps=inner_steps)
        with torch.no_grad():
            for pm, pi in zip(meta.parameters(), inner.parameters()):
                pm.add_(eps * (pi - pm))
    return meta


def fomaml_train(src_data, n_in, outer_steps, K, n_query, inner_steps,
                 inner_lr, outer_lr, rng):
    meta = MLP(n_in).to(DEVICE)
    meta_optim = torch.optim.Adam(meta.parameters(), lr=outer_lr)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        sup, qry = sample_kshot(X, y, K, n_query, rng)
        inner = deepcopy(meta)
        inner_optim = torch.optim.SGD(inner.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            inner_optim.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                inner(torch.from_numpy(X[sup])),
                torch.from_numpy(y[sup]),
            )
            loss.backward()
            inner_optim.step()
        # Compute query gradient through inner; copy to meta (FOMAML approximation)
        for p in inner.parameters():
            p.grad = None
        loss_q = F.binary_cross_entropy_with_logits(
            inner(torch.from_numpy(X[qry])),
            torch.from_numpy(y[qry]),
        )
        loss_q.backward()
        meta_optim.zero_grad()
        for pm, pi in zip(meta.parameters(), inner.parameters()):
            if pi.grad is not None:
                pm.grad = pi.grad.detach().clone()
        meta_optim.step()
    return meta


def pretrain(src_data, n_in, steps, lr):
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    Xs = standardize_basin(Xs).astype(np.float32)
    model = MLP(n_in).to(DEVICE)
    return fit_inner(model, Xs, ys, lr=lr, steps=steps)


def evaluate_episode(method, init_state, X_norm, y, K, rng, n_in):
    sup, qry = sample_kshot(X_norm, y, K, N_QUERY, rng)
    Xs, ys = X_norm[sup], y[sup]
    Xq, yq = X_norm[qry], y[qry]
    model = MLP(n_in).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = fit_inner(model, Xs, ys, lr=ADAPT_LR, steps=ADAPT_STEPS)
    p = predict_proba(model, Xq)
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(yq, pred, zero_division=0),
        "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan"),
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95, rng=None):
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def main():
    feat_path = ML / "../aligned/cfs_selected_features.txt"
    features = [l.strip() for l in feat_path.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    n_in = len(features)
    print(f"Features ({n_in}): {features}")

    print("\nLoading basins...")
    src_data_raw = {b: load_basin(b, features) for b in SOURCE}
    tgt_data_raw = {b: load_basin(b, features) for b in TARGET}
    # Per-basin standardization for source (used for both meta + pretrain)
    src_data_std = {b: (standardize_basin(X), y) for b, (X, y) in src_data_raw.items()}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n========== seed={seed} ==========")

        # Train meta-models
        print("  pretrain (concat all source)...")
        pre_state = deepcopy(pretrain(src_data_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())

        print("  reptile meta-train...")
        rep_state = deepcopy(reptile_train(
            src_data_std, n_in, META_OUTER, META_K, META_INNER_STEPS,
            META_INNER_LR, REPTILE_EPS, rng,
        ).state_dict())

        print("  fomaml meta-train...")
        fom_state = deepcopy(fomaml_train(
            src_data_std, n_in, META_OUTER, META_K, N_QUERY,
            META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng,
        ).state_dict())

        # Evaluate on each target × K × method
        for tgt in TARGET:
            X, y = tgt_data_raw[tgt]
            X_std = standardize_basin(X)
            for K in KS:
                if K * 2 + N_QUERY * 2 > len(X):
                    continue
                for method, init_state in [
                    ("independent", None),
                    ("finetune",    pre_state),
                    ("reptile",     rep_state),
                    ("fomaml",      fom_state),
                ]:
                    rng_eval = np.random.default_rng(seed * 1000 + hash(tgt + method) % 1000)
                    for ep in range(N_EPISODES):
                        m = evaluate_episode(method, init_state, X_std, y, K,
                                             rng_eval, n_in)
                        rows.append({
                            "target": tgt, "K": K, "method": method,
                            "seed": seed, "episode": ep,
                            "f1": m["f1"], "auc": m["auc"],
                        })
                print(f"  {tgt:<10} K={K:<3} done")

    # Save raw + summary
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_runs.csv", index=False)
    print(f"\nRaw runs → {OUT / 'raw_runs.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (tgt, K, method), grp in raw.groupby(["target", "K", "method"]):
        f1_m, f1_lo, f1_hi = bootstrap_ci(grp["f1"].values)
        auc_m, auc_lo, auc_hi = bootstrap_ci(grp["auc"].values)
        summary_rows.append({
            "target": tgt, "K": K, "method": method, "n_runs": len(grp),
            "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    pivot = summary.pivot_table(index=["target", "K"], columns="method", values="f1_mean")
    pivot.to_csv(OUT / "f1_table.csv")
    print(f"Summary → {OUT / 'summary.csv'}")
    print("\n=== F1 mean by method × target × K ===")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
