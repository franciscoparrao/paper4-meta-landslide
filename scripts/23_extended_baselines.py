#!/usr/bin/env python3
"""Paper 4 — extended baselines for ISPRS JPRS R1 readiness.

Adds modern meta-learning and DA baselines under spatial CV:
  - ProtoNet (Snell et al. 2017) — episodic prototypical networks
  - Meta-Baseline (Chen et al. 2021) — concat-pretrain + cosine prototypes
  - CDAN (Long et al. 2018) — conditional adversarial DA

Runs with 5 seeds (vs 3 in main benchmark) to address reviewer concern about stat power.

Total runs (spatial CV): 3 methods × 4 targets × 5 folds × 4 K × 5 seeds × 20 episodes
                       = 24,000 K-shot adaptations

Output:
- results/extended_baselines/raw_runs.csv
- results/extended_baselines/summary.csv  (mean + bootstrap 95% CI)
- figures/extended_baselines_f1_vs_k.pdf
- Combined comparison table merging with results/spatial_benchmark/summary.csv
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, bootstrap_ci, cdan_to_mlp_state, evaluate_meta_baseline_episode,
    evaluate_protonet_episode, load_basin, load_features,
    meta_baseline_train, protonet_train, read_h5, standardize_basin, train_cdan,
)
from _meta_lib import MLP, fit_inner, evaluate_query, sample_kshot  # noqa: F401

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "extended_baselines"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7, 1729, 2718]  # 5 seeds (was 3)
K_FOLDS = 5
N_EPISODES_PER_FOLD = 20

# Training configs (match main benchmark)
META_OUTER = 300
META_K = 10
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
PRETRAIN_STEPS = 300
PRETRAIN_LR = 1e-3
CDAN_EPOCHS = 100
CDAN_BATCH = 64
CDAN_LR = 1e-3
ADAPT_LR = 1e-2
ADAPT_STEPS = 30
N_QUERY_SOURCE = 10


def load_target_with_coords(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def make_spatial_folds(coords, n_folds, seed):
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def spatial_episode_indices(y, train_mask, test_mask, K, rng):
    """Return (support_idx, query_idx) for spatial K-shot, or (None, None)."""
    train_idx = np.where(train_mask)[0]
    pos_train = train_idx[y[train_idx] == 1]
    neg_train = train_idx[y[train_idx] == 0]
    test_idx = np.where(test_mask)[0]
    Kp = min(K, len(pos_train)); Kn = min(K, len(neg_train))
    if Kp == 0 or Kn == 0 or test_idx.size == 0:
        return None, None
    if len(np.unique(y[test_idx])) < 2:
        return None, None
    sup_pos = rng.choice(pos_train, size=Kp, replace=False)
    sup_neg = rng.choice(neg_train, size=Kn, replace=False)
    return np.concatenate([sup_pos, sup_neg]), test_idx


def spatial_kshot_finetune(init_state, X_norm, y, sup, qry, n_in):
    """Standard fine-tune evaluation used for CDAN (mirrors DANN protocol)."""
    Xs, ys = X_norm[sup], y[sup]
    Xq, yq = X_norm[qry], y[qry]
    model = MLP(n_in).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = fit_inner(model, Xs, ys, lr=ADAPT_LR, steps=ADAPT_STEPS)
    return evaluate_query(model, Xq, yq)


def run_one_seed(seed, src_raw, src_std, tgt_data, n_in):
    """Run pretraining + per-target evaluation for one seed. Returns list of row dicts."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    print(f"[seed={seed}] starting...", flush=True)

    proto_enc = protonet_train(
        src_std, n_in, META_OUTER, META_K, N_QUERY_SOURCE, META_OUTER_LR, rng,
    )
    mb_enc = meta_baseline_train(src_raw, n_in, PRETRAIN_STEPS, PRETRAIN_LR)
    cdan = train_cdan(src_raw, n_in, CDAN_EPOCHS, CDAN_BATCH, CDAN_LR, rng)
    cdan_state = cdan_to_mlp_state(cdan)
    print(f"[seed={seed}] pretraining done", flush=True)

    rows = []
    for tgt in TARGET:
        X, y, coords = tgt_data[tgt]
        X_std = standardize_basin(X)
        if len(X) < K_FOLDS * 4:
            continue
        folds = make_spatial_folds(coords, K_FOLDS, seed)

        for fold in range(K_FOLDS):
            test_mask = folds == fold
            train_mask = ~test_mask
            for K in KS:
                if K * 2 > train_mask.sum():
                    continue
                rng_ep = np.random.default_rng(
                    seed * 1000 + fold * 100 + K + hash(tgt) % 100,
                )
                for ep in range(N_EPISODES_PER_FOLD):
                    sup, qry = spatial_episode_indices(y, train_mask, test_mask, K, rng_ep)
                    if sup is None:
                        continue

                    m_pn = evaluate_protonet_episode(
                        proto_enc, X_std, y, K, 0, n_in, rng_ep,
                        support_idx=sup, query_idx=qry,
                    )
                    m_mb = evaluate_meta_baseline_episode(
                        mb_enc, X_std, y, K, 0, n_in, rng_ep,
                        support_idx=sup, query_idx=qry,
                    )
                    m_cd = spatial_kshot_finetune(cdan_state, X_std, y, sup, qry, n_in)

                    for method, mres in [("protonet", m_pn),
                                          ("meta_baseline", m_mb),
                                          ("cdan", m_cd)]:
                        rows.append({
                            "target": tgt, "fold": fold, "K": K,
                            "method": method, "seed": seed, "episode": ep,
                            "f1": mres["f1"], "auc": mres["auc"],
                            "n_test": int(test_mask.sum()),
                        })
    print(f"[seed={seed}] done ({len(rows)} rows)", flush=True)
    return rows


def main():
    import os
    n_jobs = int(os.environ.get("PAPER4_N_JOBS", "-1"))
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    tgt_data = {b: load_target_with_coords(b, features) for b in TARGET}

    print(f"Launching {len(SEEDS)} seeds in parallel (n_jobs={n_jobs})...", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one_seed)(seed, src_raw, src_std, tgt_data, n_in)
        for seed in SEEDS
    )
    rows = [r for chunk in results for r in chunk]

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

    # Combined plot vs existing main benchmark
    main_summary_path = ROOT / "results" / "spatial_benchmark" / "summary.csv"
    if main_summary_path.exists():
        main_sum = pd.read_csv(main_summary_path)
        main_sum["family"] = "main_benchmark"
        summary["family"] = "extended"
        combined = pd.concat([main_sum, summary], ignore_index=True)
        combined.to_csv(OUT / "combined_with_main.csv", index=False)
        print(f"Combined → {OUT / 'combined_with_main.csv'}")

    # Quick figure: F1 vs K per target × method
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    colors = {"protonet": "#984ea3", "meta_baseline": "#ff7f00", "cdan": "#377eb8"}
    for ax, tgt in zip(axes, TARGET):
        for method in ["protonet", "meta_baseline", "cdan"]:
            sub = summary[(summary.target == tgt) & (summary.method == method)].sort_values("K")
            if sub.empty:
                continue
            ax.plot(sub["K"], sub["f1_mean"], marker="o", color=colors[method], label=method)
            ax.fill_between(sub["K"], sub["f1_ci_lo"], sub["f1_ci_hi"],
                            color=colors[method], alpha=0.2)
        ax.set_title(tgt.capitalize()); ax.set_xlabel("K shots/class")
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$F_1$")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Extended baselines under spatial CV (5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "extended_baselines_f1_vs_k.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "extended_baselines_f1_vs_k.png", bbox_inches="tight", dpi=150)
    print(f"Figure → {FIGS / 'extended_baselines_f1_vs_k.pdf'}")


if __name__ == "__main__":
    main()
