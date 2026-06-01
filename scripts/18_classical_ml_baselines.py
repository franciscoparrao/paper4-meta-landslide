#!/usr/bin/env python3
"""Paper 4 — classical ML baselines (XGBoost, CatBoost, Random Forest, Logistic Regression).

K-shot evaluation: train each classical model on K target labels (independent),
on concat-source pretrain + K-target fine-tune (where applicable), and as
zero-shot transfer (train source-only, predict target).

Output:
- results/classical/raw_runs.csv
- results/classical/summary.csv
- figures/classical_vs_meta.png
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (bootstrap_ci, load_basin, load_features, sample_kshot,
                       standardize_basin)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "classical"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa"]  # will be updated by expansion
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
N_QUERY = 10
N_EPISODES = 50

# Classical algorithms (xgboost/catboost are conditionally imported)
def get_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             verbosity=0)
    except ImportError:
        return None


def get_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                  verbose=0, allow_writing_files=False)
    except ImportError:
        return None


def make_models():
    models = {
        "logreg": lambda: LogisticRegression(max_iter=500, C=1.0),
        "rf":     lambda: RandomForestClassifier(n_estimators=100, max_depth=10,
                                                 random_state=42, n_jobs=1),
    }
    if get_xgb() is not None:
        models["xgb"] = get_xgb
    if get_catboost() is not None:
        models["catboost"] = get_catboost
    return models


def evaluate_classical(model_factory, X_norm, y, K, n_query, rng,
                       pretrain_X=None, pretrain_y=None):
    sup, qry = sample_kshot(X_norm, y, K, n_query, rng)
    Xs, ys = X_norm[sup], y[sup]
    Xq, yq = X_norm[qry], y[qry]

    if pretrain_X is not None:
        # concat-source: combine pretrain + K target
        Xtrain = np.concatenate([pretrain_X, Xs])
        ytrain = np.concatenate([pretrain_y, ys])
    else:
        Xtrain, ytrain = Xs, ys

    if len(np.unique(ytrain)) < 2:
        return None
    model = model_factory()
    try:
        model.fit(Xtrain, ytrain)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(Xq)[:, 1]
        else:
            p = model.predict(Xq).astype(float)
    except Exception:
        return None
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(yq, pred, zero_division=0),
        "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan"),
    }


def main():
    features = load_features()
    n_in = len(features)

    # Detect dynamically all available source basins
    src_h5 = list(ML.glob("*.h5"))
    available_basins = sorted([p.stem for p in src_h5 if p.stem != "all_basins"])
    target_set = set(TARGET)
    sources = [b for b in available_basins if b not in target_set]
    print(f"Sources detected: {sources}")
    print(f"Targets: {TARGET}")

    src_data = {b: load_basin(b, features) for b in sources}
    tgt_data = {b: load_basin(b, features) for b in TARGET}

    # Concat source for "concat-source pretrain" mode
    Xs_concat = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys_concat = np.concatenate([y for _, y in src_data.values()], axis=0)
    src_mu = Xs_concat.mean(0); src_sd = np.maximum(Xs_concat.std(0), 1e-6)
    Xs_concat_n = (Xs_concat - src_mu) / src_sd

    models = make_models()
    print(f"Classical models: {list(models.keys())}")

    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        for tgt in TARGET:
            X, y = tgt_data[tgt]
            X_std = standardize_basin(X)
            for K in KS:
                if K * 2 + N_QUERY * 2 > len(X):
                    continue
                for model_name, factory in models.items():
                    for mode in ["independent", "concat_source"]:
                        rng = np.random.default_rng(
                            seed * 1000 + hash(tgt + model_name + mode) % 1000)
                        for ep in range(N_EPISODES):
                            if mode == "concat_source":
                                m = evaluate_classical(factory, X_std, y, K, N_QUERY,
                                                       rng,
                                                       pretrain_X=Xs_concat_n.astype(np.float32),
                                                       pretrain_y=ys_concat.astype(np.float32))
                            else:
                                m = evaluate_classical(factory, X_std, y, K, N_QUERY, rng)
                            if m is None:
                                continue
                            rows.append({"target": tgt, "K": K, "model": model_name,
                                         "mode": mode, "seed": seed, "episode": ep,
                                         "f1": m["f1"], "auc": m["auc"]})
                print(f"  seed={seed} {tgt:<10} K={K:<3} done ({len(models)} models × 2 modes)")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_runs.csv", index=False)
    print(f"\nRaw → {OUT / 'raw_runs.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (tgt, K, model, mode), grp in raw.groupby(["target", "K", "model", "mode"]):
        f1_m, f1_lo, f1_hi = bootstrap_ci(grp["f1"].values)
        auc_m, auc_lo, auc_hi = bootstrap_ci(grp["auc"].values)
        summary_rows.append({"target": tgt, "K": K, "model": model, "mode": mode,
                             "n_runs": len(grp),
                             "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
                             "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(f"Summary → {OUT / 'summary.csv'}")

    # Compare with meta-learning
    bench = pd.read_csv(ROOT / "results/spatial_benchmark/summary.csv")
    bench_meta = bench[bench.method.isin(["independent", "fomaml", "reptile"])]

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                     "elqui": "Elqui", "limari": "Limarí"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, tgt in zip(axes.flat, TARGET):
        # Classical: best mode per model
        sub_c = summary[summary.target == tgt]
        for model, color in [("logreg", "#999999"), ("rf", "#984ea3"),
                             ("xgb", "#a6611a"), ("catboost", "#dfc27d")]:
            for mode, ls in [("independent", "--"), ("concat_source", "-")]:
                d = sub_c[(sub_c.model == model) & (sub_c.mode == mode)].sort_values("K")
                if d.empty:
                    continue
                ax.plot(d["K"], d["f1_mean"], marker="^", color=color,
                        linestyle=ls, alpha=0.7, linewidth=1.4, markersize=5,
                        label=f"{model} ({mode[:4]})" if tgt == "copiapo" else None)
        # Meta-learning baseline (FOMAML, Independent NN)
        sub_m = bench_meta[bench_meta.target == tgt]
        for m, color in [("independent", "#888888"), ("fomaml", "#c51b8a")]:
            d = sub_m[sub_m.method == m].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["K"], d["f1_mean"], marker="o", color=color,
                    linewidth=2.5, markersize=7,
                    label=f"NN {m}" if tgt == "copiapo" else None)
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K"); ax.set_title(target_labels[tgt])
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("F1 (spatial CV)"); axes[1, 0].set_ylabel("F1 (spatial CV)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.20, 0.5),
               frameon=False, fontsize=8, ncol=1)
    fig.suptitle("Classical ML vs meta-learning (K-shot, spatial CV)", y=1.02)
    fig.tight_layout()
    out = FIGS / "classical_vs_meta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    main()
