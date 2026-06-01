#!/usr/bin/env python3
"""Paper 4 — adaptation curves: F1 / AUC as a function of inner gradient steps.

Validates H_p4_2: meta-learned models converge in 3-5 steps vs >100 for independent.

For each (target_basin, K, method, seed): run N_EPISODES K-shot episodes; at each
predefined adaptation step, evaluate F1/AUC on the query set. This reveals the
*adaptation efficiency* per method, not just final performance.

Output:
- results/adaptation/curves_raw.csv  (target, K, method, seed, episode, step, f1, auc)
- results/adaptation/curves_summary.csv  (mean ± 95% CI bootstrap per (target, K, method, step))
- figures/adaptation_curves.png      (4-panel per target, F1 vs step)
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "adaptation"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [5, 10]   # representative; not full grid (this is about adaptation speed not best K)
SEEDS = [42, 123, 7]
N_EPISODES = 30
N_QUERY = 10
ADAPT_STEPS_TRACKED = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200]
ADAPT_LR = 1e-2

# Meta-training config (matches benchmark)
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
REPTILE_EPS = 0.1
PRETRAIN_STEPS = 300
PRETRAIN_LR = 1e-3

DEVICE = torch.device("cpu")


def read_h5(path: Path):
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


def fit_one_step(model, optim, X, y):
    optim.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(X), y)
    loss.backward()
    optim.step()
    return loss.item()


def fit_inner(model, X, y, lr, steps):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    for _ in range(steps):
        fit_one_step(model, optim, Xt, yt)
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).cpu().numpy()


def evaluate_query(model, Xq, yq):
    p = predict_proba(model, Xq)
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(yq, pred, zero_division=0),
        "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan"),
    }


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


def pretrain_concat(src_data, n_in, steps, lr):
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    Xs = standardize_basin(Xs).astype(np.float32)
    model = MLP(n_in).to(DEVICE)
    return fit_inner(model, Xs, ys, lr=lr, steps=steps)


def adapt_with_tracking(model, Xs, ys, Xq, yq, lr, max_steps, tracked):
    """Adapt model on support; record metrics on query at each tracked step."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xst = torch.from_numpy(Xs); yst = torch.from_numpy(ys)
    rows = []
    if 0 in tracked:
        m = evaluate_query(model, Xq, yq)
        rows.append((0, m["f1"], m["auc"]))
    for step in range(1, max_steps + 1):
        fit_one_step(model, optim, Xst, yst)
        if step in tracked:
            m = evaluate_query(model, Xq, yq)
            rows.append((step, m["f1"], m["auc"]))
    return rows


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
    print(f"Features: {n_in}")

    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    tgt_raw = {b: load_basin(b, features) for b in TARGET}

    rows = []
    max_step = max(ADAPT_STEPS_TRACKED)

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n========== seed={seed} ==========")

        print("  pretrain...")
        pre_state = deepcopy(pretrain_concat(src_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())
        print("  reptile...")
        rep_state = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER_STEPS,
            META_INNER_LR, REPTILE_EPS, rng,
        ).state_dict())
        print("  fomaml...")
        fom_state = deepcopy(fomaml_train(
            src_std, n_in, META_OUTER, META_K, N_QUERY,
            META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng,
        ).state_dict())

        for tgt in TARGET:
            X, y = tgt_raw[tgt]
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
                    rng_ep = np.random.default_rng(seed * 1000 + hash(tgt + method) % 1000)
                    for ep in range(N_EPISODES):
                        sup, qry = sample_kshot(X_std, y, K, N_QUERY, rng_ep)
                        Xs, ys_ = X_std[sup], y[sup]
                        Xq, yq = X_std[qry], y[qry]
                        model = MLP(n_in).to(DEVICE)
                        if init_state is not None:
                            model.load_state_dict(init_state)
                        for step, f1, auc in adapt_with_tracking(
                            model, Xs, ys_, Xq, yq, ADAPT_LR, max_step, ADAPT_STEPS_TRACKED,
                        ):
                            rows.append({
                                "target": tgt, "K": K, "method": method,
                                "seed": seed, "episode": ep, "step": step,
                                "f1": f1, "auc": auc,
                            })
                print(f"  {tgt:<10} K={K:<3} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "curves_raw.csv", index=False)
    print(f"\nRaw → {OUT / 'curves_raw.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (tgt, K, method, step), grp in raw.groupby(["target", "K", "method", "step"]):
        f1_m, f1_lo, f1_hi = bootstrap_ci(grp["f1"].values)
        auc_m, auc_lo, auc_hi = bootstrap_ci(grp["auc"].values)
        summary_rows.append({
            "target": tgt, "K": K, "method": method, "step": step,
            "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "curves_summary.csv", index=False)

    # Plot — F1 vs step, K=5, 4-panel
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    methods = ["independent", "finetune", "reptile", "fomaml"]
    colors = {"independent": "#888888", "finetune": "#2c7fb8",
              "reptile": "#d95f0e", "fomaml": "#c51b8a"}
    target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                     "elqui": "Elqui", "limari": "Limarí"}

    for K_show in KS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        for ax, tgt in zip(axes.flat, TARGET):
            sub = summary[(summary.target == tgt) & (summary.K == K_show)]
            if sub.empty:
                ax.set_title(f"{target_labels[tgt]} (skip K={K_show})")
                continue
            for m in methods:
                d = sub[sub.method == m].sort_values("step")
                if d.empty:
                    continue
                ax.plot(d["step"], d["f1_mean"], marker="o",
                        color=colors[m], label=m.capitalize(),
                        linewidth=1.8, markersize=5)
                ax.fill_between(d["step"], d["f1_ci_lo"], d["f1_ci_hi"],
                                color=colors[m], alpha=0.18, linewidth=0)
            ax.set_title(f"{target_labels[tgt]} (K={K_show})")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0, 1, 3, 5, 10, 50, 200])
            ax.set_xticklabels(["0", "1", "3", "5", "10", "50", "200"])
            ax.grid(True, alpha=0.25, linewidth=0.5)
        axes[0, 0].set_ylabel("F1")
        axes[1, 0].set_ylabel("F1")
        axes[1, 0].set_xlabel("Adaptation steps")
        axes[1, 1].set_xlabel("Adaptation steps")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
        fig.suptitle(f"Adaptation efficiency (K={K_show})", y=1.04)
        fig.tight_layout()
        out = FIGS / f"adaptation_curves_K{K_show}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out.name}")


if __name__ == "__main__":
    main()
