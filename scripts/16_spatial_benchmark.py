#!/usr/bin/env python3
"""Paper 4 — spatial-aware K-shot benchmark (replaces random sampling).

Spatial protocol: for each target basin, partition pixel coordinates into K_FOLDS
spatial blocks via KMeans. For each fold:
  - Support: K positives + K negatives sampled from N-1 *training* blocks.
  - Query: full held-out block (all positives + all negatives there).
  - This guarantees support and query are spatially disjoint.

Total runs: 5 methods × 4 targets × 5 folds × 4 K × 3 seeds × 20 episodes ≈ 24,000
Compared with the original random-CV benchmark (8,400 runs), this is the rigorous
version that addresses spatial autocorrelation.

Output:
- results/spatial_benchmark/raw_runs.csv
- results/spatial_benchmark/summary.csv  (mean + bootstrap 95% CI per target × K × method)
- figures/spatial_f1_vs_k_panel.png
- figures/spatial_lift_vs_k.png
- figures/spatial_vs_random_comparison.png  (combined view: random vs spatial benchmark)
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
    DEVICE, MLP, bootstrap_ci, dann_to_mlp_state, evaluate_query,
    fit_inner, fomaml_train, load_basin, load_features, pretrain_concat,
    read_h5, reptile_train, sample_kshot, standardize_basin, train_dann,
)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "spatial_benchmark"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
K_FOLDS = 5
N_EPISODES_PER_FOLD = 20

# Meta-train + adapt configs (matches benchmark)
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
REPTILE_EPS = 0.1
PRETRAIN_STEPS = 300
PRETRAIN_LR = 1e-3
DANN_EPOCHS = 100
DANN_BATCH = 64
DANN_LR = 1e-3
ADAPT_LR = 1e-2
ADAPT_STEPS = 30
N_QUERY_SOURCE = 10  # query size used during meta-training


def load_target_with_coords(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def make_spatial_folds(coords, n_folds, seed):
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def spatial_kshot_episode(init_state, X_norm, y, train_mask, test_mask, K,
                           n_in, rng):
    """Sample K from train_mask; evaluate on full test_mask."""
    train_idx = np.where(train_mask)[0]
    pos_train = train_idx[y[train_idx] == 1]
    neg_train = train_idx[y[train_idx] == 0]
    test_idx = np.where(test_mask)[0]
    Kp = min(K, len(pos_train)); Kn = min(K, len(neg_train))
    if Kp == 0 or Kn == 0 or test_idx.size == 0:
        return None
    sup_pos = rng.choice(pos_train, size=Kp, replace=False)
    sup_neg = rng.choice(neg_train, size=Kn, replace=False)
    sup = np.concatenate([sup_pos, sup_neg])
    Xs, ys = X_norm[sup], y[sup]
    Xte, yte = X_norm[test_idx], y[test_idx]
    if len(np.unique(yte)) < 2:
        return None
    model = MLP(n_in).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = fit_inner(model, Xs, ys, lr=ADAPT_LR, steps=ADAPT_STEPS)
    return evaluate_query(model, Xte, yte)


def main():
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    tgt_data = {b: load_target_with_coords(b, features) for b in TARGET}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        print(f"\n========== seed={seed} ==========")

        print("  pretrain (concat source)...")
        pre_state = deepcopy(pretrain_concat(src_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())
        print("  reptile meta-train...")
        rep_state = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER_STEPS,
            META_INNER_LR, REPTILE_EPS, rng,
        ).state_dict())
        print("  fomaml meta-train...")
        fom_state = deepcopy(fomaml_train(
            src_std, n_in, META_OUTER, META_K, N_QUERY_SOURCE,
            META_INNER_STEPS, META_INNER_LR, META_OUTER_LR, rng,
        ).state_dict())
        print("  dann train...")
        dann = train_dann(src_raw, n_in, DANN_EPOCHS, DANN_BATCH, DANN_LR, rng)
        dann_state = dann_to_mlp_state(dann)

        for tgt in TARGET:
            X, y, coords = tgt_data[tgt]
            X_std = standardize_basin(X)
            if len(X) < K_FOLDS * 4:
                print(f"  [skip] {tgt} N={len(X)} too small")
                continue
            folds = make_spatial_folds(coords, K_FOLDS, seed)

            for fold in range(K_FOLDS):
                test_mask = folds == fold
                train_mask = ~test_mask
                for K in KS:
                    if K * 2 > train_mask.sum():
                        continue
                    for method, state in [
                        ("independent", None),
                        ("finetune",    pre_state),
                        ("dann",        dann_state),
                        ("reptile",     rep_state),
                        ("fomaml",      fom_state),
                    ]:
                        rng_ep = np.random.default_rng(
                            seed * 1000 + fold * 100 + K
                            + hash(method + tgt) % 100,
                        )
                        for ep in range(N_EPISODES_PER_FOLD):
                            m = spatial_kshot_episode(state, X_std, y, train_mask,
                                                      test_mask, K, n_in, rng_ep)
                            if m is None:
                                continue
                            rows.append({
                                "target": tgt, "fold": fold, "K": K,
                                "method": method, "seed": seed, "episode": ep,
                                "f1": m["f1"], "auc": m["auc"],
                                "n_test": int(test_mask.sum()),
                            })
                print(f"  {tgt:<10} fold={fold} done")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_runs.csv", index=False)
    print(f"\nRaw → {OUT / 'raw_runs.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (tgt, K, method), grp in raw.groupby(["target", "K", "method"]):
        f1_m, f1_lo, f1_hi = bootstrap_ci(grp["f1"].values)
        auc_m, auc_lo, auc_hi = bootstrap_ci(grp["auc"].values)
        summary_rows.append({"target": tgt, "K": K, "method": method,
                             "n_runs": len(grp),
                             "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
                             "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(f"Summary → {OUT / 'summary.csv'}")

    # ---- Plots ----
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    methods_order = ["independent", "finetune", "dann", "reptile", "fomaml"]
    colors = {"independent": "#888888", "finetune": "#2c7fb8", "dann": "#1a9850",
              "reptile": "#d95f0e", "fomaml": "#c51b8a"}
    target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                     "elqui": "Elqui", "limari": "Limarí"}

    # Spatial F1 vs K
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, tgt in zip(axes.flat, TARGET):
        sub = summary[summary.target == tgt]
        for m in methods_order:
            d = sub[sub.method == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d["f1_mean"], marker="o", color=colors[m],
                    label=m.capitalize(), linewidth=1.8, markersize=6)
            ax.fill_between(d["K"], d["f1_ci_lo"], d["f1_ci_hi"],
                            color=colors[m], alpha=0.15, linewidth=0)
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K (shots per class)")
        ax.set_title(target_labels[tgt])
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("F1 (spatial CV)"); axes[1, 0].set_ylabel("F1 (spatial CV)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    out = FIGS / "spatial_f1_vs_k_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # Spatial lift vs K
    pivot = summary.pivot_table(index=["target", "K"], columns="method",
                                values="f1_mean").reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ["finetune", "dann", "reptile", "fomaml"]:
        if m not in pivot.columns:
            continue
        mean_lift = (pivot[m] - pivot["independent"]).groupby(pivot["K"]).mean()
        ax.plot(mean_lift.index, mean_lift.values, marker="s",
                color=colors[m], linewidth=2.5, label=f"{m.capitalize()}")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(["1", "5", "10", "20"])
    ax.set_xlabel("K (shots per class)")
    ax.set_ylabel("F1 advantage over Independent (spatial CV, mean across targets)")
    ax.set_title("Meta-learning advantage under spatial CV")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = FIGS / "spatial_lift_vs_k.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # Random vs spatial benchmark side-by-side
    bench = pd.read_csv(ROOT / "results/benchmark/summary.csv")
    bench["protocol"] = "random"
    sp_summary = summary.copy(); sp_summary["protocol"] = "spatial"
    combined = pd.concat([bench, sp_summary], ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for ax, tgt in zip(axes, TARGET):
        sub = combined[combined.target == tgt]
        for m in methods_order:
            for proto, ls, alpha in [("random", "--", 0.5), ("spatial", "-", 1.0)]:
                d = sub[(sub.method == m) & (sub.protocol == proto)].sort_values("K")
                if d.empty:
                    continue
                ax.plot(d["K"], d["f1_mean"], marker="o", color=colors[m],
                        linestyle=ls, alpha=alpha, linewidth=1.5, markersize=5,
                        label=f"{m} ({proto})" if tgt == TARGET[0] else None)
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K"); ax.set_title(target_labels[tgt])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.18, 0.5),
               frameon=False, fontsize=8)
    fig.suptitle("Random CV (dashed) vs Spatial CV (solid)", y=1.02)
    fig.tight_layout()
    out = FIGS / "spatial_vs_random_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    main()
