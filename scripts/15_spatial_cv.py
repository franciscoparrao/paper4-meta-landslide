#!/usr/bin/env python3
"""Paper 4 — spatial cross-validation to verify no spatial autocorrelation leakage.

For each target basin:
  1. Cluster pixel coordinates (x_utm, y_utm) into K_FOLDS spatial blocks via KMeans.
  2. For each fold: hold-out one block as test; sample K_ADAPT support points from
     the OTHER N-1 blocks; adapt meta-model and evaluate on held-out block.
  3. Compare with random-CV baseline (no spatial structure).

If meta-learning F1 stays stable across spatial folds → no leakage. If it drops
sharply vs random CV → there IS spatial leakage (warning sign).

Output:
- results/spatial_cv/spatial_cv.csv  (target × fold × method × eval F1/AUC)
- figures/spatial_cv.png              (bar plot per target, spatial vs random)
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, MLP, fit_inner, fomaml_train, load_basin, load_features,
    pretrain_concat, read_h5, reptile_train, sample_kshot,
    standardize_basin, evaluate_query,
)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "spatial_cv"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
K_FOLDS = 5
K_ADAPT = 20  # samples per class for adaptation
SEEDS = [42, 123, 7]
N_EPISODES_PER_FOLD = 10  # repetitions per fold to average over support sampling

# Meta-train config (matches benchmark)
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
REPTILE_EPS = 0.1
PRETRAIN_STEPS = 300
PRETRAIN_LR = 1e-3
ADAPT_LR = 1e-2
ADAPT_STEPS = 30


def make_spatial_folds(coords, n_folds, seed):
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def make_random_folds(n, n_folds, rng):
    order = rng.permutation(n)
    folds = np.zeros(n, dtype=np.int64)
    for i, idx in enumerate(np.array_split(order, n_folds)):
        folds[idx] = i
    return folds


def load_target_with_coords(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def evaluate_fold(model_state, X_norm, y, train_mask, test_mask, K, n_in, rng):
    """Sample K-shot from train_mask; evaluate on test_mask."""
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if test_idx.size == 0:
        return None
    Xtr, ytr = X_norm[train_idx], y[train_idx]
    Xte, yte = X_norm[test_idx], y[test_idx]
    pos_train = train_idx[y[train_idx] == 1]
    neg_train = train_idx[y[train_idx] == 0]
    Kp = min(K, len(pos_train)); Kn = min(K, len(neg_train))
    if Kp == 0 or Kn == 0:
        return None
    sup_pos = rng.choice(pos_train, size=Kp, replace=False)
    sup_neg = rng.choice(neg_train, size=Kn, replace=False)
    sup = np.concatenate([sup_pos, sup_neg])
    Xs, ys = X_norm[sup], y[sup]
    model = MLP(n_in).to(DEVICE)
    if model_state is not None:
        model.load_state_dict(model_state)
    model = fit_inner(model, Xs, ys, lr=ADAPT_LR, steps=ADAPT_STEPS)
    return evaluate_query(model, Xte, yte)


def main():
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    target_data = {b: load_target_with_coords(b, features) for b in TARGET}

    rows = []
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
            src_std, n_in, META_OUTER, META_K, K_ADAPT,
            META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng,
        ).state_dict())

        for tgt in TARGET:
            X, y, coords = target_data[tgt]
            X_std = standardize_basin(X)
            if len(X) < K_FOLDS * 4:  # not enough samples
                print(f"  [skip] {tgt} (N={len(X)} too small)")
                continue
            spatial_folds = make_spatial_folds(coords, K_FOLDS, seed)

            for cv_type in ["spatial", "random"]:
                rng_cv = np.random.default_rng(seed * 7 + hash(tgt + cv_type) % 1000)
                if cv_type == "spatial":
                    folds = spatial_folds
                else:
                    folds = make_random_folds(len(X), K_FOLDS, rng_cv)

                for fold in range(K_FOLDS):
                    test_mask = folds == fold
                    train_mask = ~test_mask
                    for method, state in [("independent", None),
                                          ("finetune", pre_state),
                                          ("reptile", rep_state),
                                          ("fomaml", fom_state)]:
                        for ep in range(N_EPISODES_PER_FOLD):
                            rng_e = np.random.default_rng(
                                seed * 100 + fold * 10 + ep
                                + hash(method + cv_type + tgt) % 100,
                            )
                            m = evaluate_fold(state, X_std, y, train_mask, test_mask,
                                              K_ADAPT, n_in, rng_e)
                            if m is None:
                                continue
                            rows.append({"target": tgt, "cv_type": cv_type,
                                         "fold": fold, "method": method,
                                         "seed": seed, "episode": ep,
                                         "f1": m["f1"], "auc": m["auc"]})
                print(f"  {tgt:<10} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "spatial_cv.csv", index=False)
    print(f"\nRaw → {OUT / 'spatial_cv.csv'} ({len(raw)} rows)")

    # Summary: F1 mean per (target, method, cv_type)
    summary = (raw.groupby(["target", "method", "cv_type"])
                  .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                       auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                       n=("f1", "count"))
                  .reset_index())
    summary.to_csv(OUT / "spatial_cv_summary.csv", index=False)
    print(f"Summary → {OUT / 'spatial_cv_summary.csv'}")

    # Plot: per target, group of bars (random vs spatial) × method
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    methods = ["independent", "finetune", "reptile", "fomaml"]
    colors = {"independent": "#888888", "finetune": "#2c7fb8",
              "reptile": "#d95f0e", "fomaml": "#c51b8a"}
    target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                     "elqui": "Elqui", "limari": "Limarí"}

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for ax, tgt in zip(axes, TARGET):
        sub = summary[summary.target == tgt]
        if sub.empty:
            ax.set_title(f"{target_labels[tgt]} (skipped)")
            continue
        x_positions = np.arange(len(methods))
        width = 0.35
        for offset, cv in [(-width / 2, "random"), (width / 2, "spatial")]:
            vals = [sub[(sub.method == m) & (sub.cv_type == cv)]["f1_mean"].values
                    for m in methods]
            vals = [v[0] if len(v) else np.nan for v in vals]
            errs = [sub[(sub.method == m) & (sub.cv_type == cv)]["f1_std"].values
                    for m in methods]
            errs = [e[0] if len(e) else 0 for e in errs]
            ax.bar(x_positions + offset, vals, width=width,
                   yerr=errs, capsize=3,
                   color=[colors[m] for m in methods],
                   alpha=1.0 if cv == "spatial" else 0.5,
                   edgecolor="black", linewidth=0.5,
                   label=cv)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([m[:4] for m in methods], rotation=0)
        ax.set_title(target_labels[tgt])
        ax.grid(True, alpha=0.25, axis="y")
    axes[0].set_ylabel("F1 (mean ± std across folds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.5),
               plt.Rectangle((0, 0), 1, 1, color="gray", alpha=1.0)]
    fig.legend(handles, ["Random CV", "Spatial CV"], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.suptitle("Spatial vs random CV (K-shot K=20)", y=1.05)
    fig.tight_layout()
    out = FIGS / "spatial_cv.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    main()
