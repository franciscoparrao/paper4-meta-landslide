#!/usr/bin/env python3
"""Paper 4 — DANN baseline (Domain-Adversarial Neural Network, Ganin & Lempitsky 2015).

Architecture:
- Feature extractor F: 14 → 64 → 32 (shared)
- Class classifier C: 32 → 1 (binary landslide)
- Domain classifier D: 32 → n_source_basins (which source basin)
- Gradient Reversal Layer between F and D

Training: source samples drive class loss (L_class); domain loss (L_domain) flows
through GRL so F learns domain-INVARIANT features. Lambda annealed via 2/(1+e^-10p)-1.

K-shot evaluation: load F+C state into MLP and adapt with K target samples (same
protocol as other methods to ensure apples-to-apples comparison).

Output:
- results/dann/dann_raw.csv  (target × K × seed × episode)
- results/dann/dann_summary.csv
- figures/dann_comparison.png  (DANN vs FOMAML/Reptile/finetune)
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, MLP, bootstrap_ci, evaluate_kshot_episode, load_basin,
    load_features, sample_kshot, standardize_basin,
)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
OUT = ROOT / "results" / "dann"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
N_EPISODES = 50
N_QUERY = 10

# DANN training config
DANN_EPOCHS = 100
DANN_BATCH = 64
DANN_LR = 1e-3


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DANN(nn.Module):
    """Feature extractor + class head + domain head (with GRL)."""
    def __init__(self, n_in, n_domains, hidden=(64, 32)):
        super().__init__()
        # Feature extractor (matches MLP feature path)
        self.feat = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
        )
        self.class_head = nn.Linear(hidden[1], 1)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden[1], 32), nn.ReLU(),
            nn.Linear(32, n_domains),
        )

    def forward(self, x, lambd=1.0):
        z = self.feat(x)
        return self.class_head(z).squeeze(-1), self.domain_head(grad_reverse(z, lambd))


def dann_to_mlp_state(dann: DANN) -> dict:
    """Pack DANN feat + class_head into the MLP state-dict layout used by adapt-time."""
    mlp = MLP(dann.feat[0].in_features)
    # MLP layout: net[0]=Linear, net[3]=Linear, net[6]=Linear (head)
    mlp.net[0].load_state_dict(dann.feat[0].state_dict())
    mlp.net[3].load_state_dict(dann.feat[3].state_dict())
    mlp.net[6].load_state_dict(dann.class_head.state_dict())
    return deepcopy(mlp.state_dict())


def train_dann(src_data, n_in, epochs, batch, lr, rng):
    """Standard DANN training: alternating class + domain loss with annealed lambda."""
    domain_to_id = {b: i for i, b in enumerate(src_data)}
    Xs_list, ys_list, ds_list = [], [], []
    for b, (X, y) in src_data.items():
        Xs_list.append(X)
        ys_list.append(y)
        ds_list.append(np.full(len(y), domain_to_id[b], dtype=np.int64))
    X_all = np.concatenate(Xs_list, axis=0)
    y_all = np.concatenate(ys_list, axis=0)
    d_all = np.concatenate(ds_list, axis=0)

    # Standardize globally (sources concatenated)
    mu = X_all.mean(0); sd = np.maximum(X_all.std(0), 1e-6)
    X_all = ((X_all - mu) / sd).astype(np.float32)

    model = DANN(n_in, n_domains=len(domain_to_id)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(X_all)
    Xt = torch.from_numpy(X_all)
    yt = torch.from_numpy(y_all.astype(np.float32))
    dt = torch.from_numpy(d_all)

    total_iters = epochs * (n // batch)
    iter_count = 0

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            batch_idx = idx[start:start + batch]
            x = Xt[batch_idx]; y_ = yt[batch_idx]; d_ = dt[batch_idx]
            p = iter_count / max(1, total_iters)
            lambd = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            optim.zero_grad()
            class_logits, domain_logits = model(x, lambd)
            loss_c = F.binary_cross_entropy_with_logits(class_logits, y_)
            loss_d = F.cross_entropy(domain_logits, d_)
            loss = loss_c + loss_d
            loss.backward()
            optim.step()
            iter_count += 1
    return model


def main():
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    tgt_raw = {b: load_basin(b, features) for b in TARGET}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n========== seed={seed} ==========")
        print("  training DANN...")
        dann = train_dann(src_raw, n_in, DANN_EPOCHS, DANN_BATCH, DANN_LR, rng)
        init_state = dann_to_mlp_state(dann)

        for tgt in TARGET:
            X, y = tgt_raw[tgt]
            X_std = standardize_basin(X)
            for K in KS:
                if K * 2 + N_QUERY * 2 > len(X):
                    continue
                rng_ep = np.random.default_rng(seed * 1000 + hash(tgt + "dann") % 1000)
                for ep in range(N_EPISODES):
                    m = evaluate_kshot_episode(init_state, X_std, y, K, N_QUERY,
                                               n_in, rng_ep)
                    rows.append({"target": tgt, "K": K, "method": "dann",
                                 "seed": seed, "episode": ep,
                                 "f1": m["f1"], "auc": m["auc"]})
                print(f"  {tgt:<10} K={K:<3} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "dann_raw.csv", index=False)
    print(f"\nRaw → {OUT / 'dann_raw.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (tgt, K, method), grp in raw.groupby(["target", "K", "method"]):
        f1_m, f1_lo, f1_hi = bootstrap_ci(grp["f1"].values)
        auc_m, auc_lo, auc_hi = bootstrap_ci(grp["auc"].values)
        summary_rows.append({"target": tgt, "K": K, "method": method, "n_runs": len(grp),
                             "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
                             "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "dann_summary.csv", index=False)
    print(f"Summary → {OUT / 'dann_summary.csv'}")

    # Combined plot: DANN vs other methods (read from benchmark/summary.csv)
    bench = pd.read_csv(ROOT / "results/benchmark/summary.csv")
    combined = pd.concat([bench, summary], ignore_index=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    colors = {"independent": "#888888", "finetune": "#2c7fb8",
              "reptile": "#d95f0e", "fomaml": "#c51b8a", "dann": "#1a9850"}
    target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                     "elqui": "Elqui", "limari": "Limarí"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, tgt in zip(axes.flat, TARGET):
        sub = combined[combined.target == tgt]
        for m in ["independent", "finetune", "dann", "reptile", "fomaml"]:
            d = sub[sub.method == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d["f1_mean"], marker="o", color=colors[m],
                    label=m.capitalize(), linewidth=1.8, markersize=6)
            ax.fill_between(d["K"], d["f1_ci_lo"], d["f1_ci_hi"],
                            color=colors[m], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20]); ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K (shots per class)")
        ax.set_title(target_labels[tgt])
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("F1"); axes[1, 0].set_ylabel("F1")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    out = FIGS / "dann_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    main()
