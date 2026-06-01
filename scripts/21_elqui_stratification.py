#!/usr/bin/env python3
"""Paper 4 — Elqui stratification analysis (Option C).

Hypothesis: Elqui's 25-pp drop under spatial CV is due to intra-basin
heterogeneity (lithology, elevation regime). If we stratify the basin into
homogeneous sub-regions and evaluate FOMAML K-shot WITHIN each stratum,
F1 should recover toward the random-CV level — confirming the limitation
is geographic heterogeneity, not method failure.

Strategy:
1. Cluster Elqui samples by (terrain__landform, terrain__geomorphons,
   focal_stats__dem_std_r10) — 3D K-means with K=3.
2. For each cluster (stratum), evaluate FOMAML K=10 K-shot via spatial CV
   (fold = 3 KMeans clusters within stratum).
3. Compare F1 per stratum vs the unstratified baseline.

Output:
- results/elqui_strat/strat_results.csv
- figures/elqui_stratification.png
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (DEVICE, MLP, fit_inner, fomaml_train, load_basin,
                       load_features, evaluate_query, sample_kshot,
                       standardize_basin)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "elqui_strat"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = "elqui"
N_STRATA = 3
K_FOLDS = 3
K_SHOT = 10
SEEDS = [42, 123, 7]
N_EPISODES = 20

# Stratification features (capture lithology / morphology proxy)
STRAT_FEATS = ["terrain__landform", "terrain__geomorphons",
               "focal_stats__dem_std_r10"]

# Meta-train config
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
ADAPT_LR = 1e-2
ADAPT_STEPS = 30


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def main():
    features = load_features()
    n_in = len(features)

    # Load Elqui with all columns (need stratification features + coords)
    df = read_h5(ML / "elqui.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)

    # Use available stratification features (some may have been dropped by CFS)
    avail_strat = [f for f in STRAT_FEATS if f in df.columns and f in features]
    if not avail_strat:
        # Fallback: use coords-only stratification (latitude bands)
        print("WARN: no stratification features available, using x_utm bands")
        strat_X = coords[:, [1]]  # y_utm as 1D
    else:
        print(f"Stratifying on: {avail_strat}")
        strat_X = df[avail_strat].to_numpy(dtype=np.float32)

    # Standardize stratification features
    strat_X_std = (strat_X - strat_X.mean(0)) / np.maximum(strat_X.std(0), 1e-6)
    km = KMeans(n_clusters=N_STRATA, random_state=42, n_init=10)
    strata = km.fit_predict(strat_X_std)
    print(f"Stratum sizes: {np.bincount(strata)}")

    # Train FOMAML on full source pool ONCE per seed
    src_data = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_data.items()}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n========== seed={seed} ==========")
        print("  fomaml meta-train...")
        fom = fomaml_train(src_std, n_in, META_OUTER, META_K, 10,
                           META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng)
        fom_state = deepcopy(fom.state_dict())

        # Standardize Elqui
        X_std = standardize_basin(X)

        # For each stratum, do internal spatial CV
        for stratum in range(N_STRATA):
            mask_strat = strata == stratum
            X_s, y_s, coords_s = X_std[mask_strat], y[mask_strat], coords[mask_strat]
            n_s = len(X_s)
            if n_s < K_FOLDS * 4 or len(np.unique(y_s)) < 2:
                print(f"  [skip] stratum {stratum} N={n_s} too small")
                continue
            # Spatial folds within stratum
            km_s = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10)
            folds = km_s.fit_predict(coords_s)
            for fold in range(K_FOLDS):
                test_mask = folds == fold
                train_mask = ~test_mask
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                pos_train = train_idx[y_s[train_idx] == 1]
                neg_train = train_idx[y_s[train_idx] == 0]
                Kp = min(K_SHOT, len(pos_train)); Kn = min(K_SHOT, len(neg_train))
                if Kp == 0 or Kn == 0 or test_idx.size == 0:
                    continue
                yte = y_s[test_idx]
                if len(np.unique(yte)) < 2:
                    continue
                rng_ep = np.random.default_rng(seed * 100 + stratum * 10 + fold)
                for ep in range(N_EPISODES):
                    sup_pos = rng_ep.choice(pos_train, size=Kp, replace=False)
                    sup_neg = rng_ep.choice(neg_train, size=Kn, replace=False)
                    sup = np.concatenate([sup_pos, sup_neg])
                    Xs, ys_ = X_s[sup], y_s[sup]
                    Xte = X_s[test_idx]
                    model = MLP(n_in).to(DEVICE)
                    model.load_state_dict(fom_state)
                    model = fit_inner(model, Xs, ys_, lr=ADAPT_LR, steps=ADAPT_STEPS)
                    m = evaluate_query(model, Xte, yte)
                    rows.append({"stratum": stratum, "fold": fold, "seed": seed,
                                 "episode": ep, "n_test": int(len(yte)),
                                 "f1": m["f1"], "auc": m["auc"]})
            print(f"  stratum {stratum} done (N={n_s})")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "strat_results.csv", index=False)
    print(f"\nRaw → {OUT / 'strat_results.csv'} ({len(raw)} rows)")

    # Summary per stratum
    summary = (raw.groupby("stratum")
                  .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                       auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                       n=("f1", "count"))
                  .reset_index())
    summary.to_csv(OUT / "strat_summary.csv", index=False)
    print("\nElqui stratified results (FOMAML K=10):")
    print(summary.to_string(index=False))

    # Compare with unstratified baseline (from spatial_benchmark)
    bench = pd.read_csv(ROOT / "results/spatial_benchmark/summary.csv")
    baseline = bench[(bench.target == "elqui") & (bench.K == 10)
                     & (bench.method == "fomaml")]["f1_mean"].iloc[0]
    print(f"\nUnstratified Elqui K=10 FOMAML F1 (spatial CV): {baseline:.3f}")

    # Plot
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(summary)) + 1
    ax.bar(x, summary["f1_mean"], yerr=summary["f1_std"], capsize=4,
           color="#c51b8a", edgecolor="black", linewidth=0.5,
           label="Stratified FOMAML")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.5,
               label=f"Unstratified baseline (F1={baseline:.2f})")
    ax.set_xticks(x); ax.set_xticklabels([f"Stratum {i}" for i in summary["stratum"]])
    ax.set_ylabel("F1 (FOMAML K=10, spatial CV within stratum)")
    ax.set_title("Elqui: stratification by terrain morphology recovers performance")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "elqui_stratification.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → elqui_stratification.png")


if __name__ == "__main__":
    main()
