#!/usr/bin/env python3
"""Paper 4 — classical ML under SPATIAL CV (apples-to-apples vs meta-learning).

Same K-shot protocol as scripts/16_spatial_benchmark.py but with
classical ML algorithms (XGBoost, CatBoost, Random Forest, Logistic Regression).

Output:
- results/classical_spatial/raw_runs.csv
- results/classical_spatial/summary.csv
- figures/classical_spatial_vs_meta.png  (4-panel: classical vs meta-learning under spatial CV)
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import bootstrap_ci, load_basin, load_features, standardize_basin

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "classical_spatial"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
K_FOLDS = 5
N_EPISODES_PER_FOLD = 10


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


def read_h5(path):
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def load_target_with_coords(basin, features):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def evaluate_fold(model_factory, X_norm, y, train_mask, test_mask, K, rng,
                   pretrain_X=None, pretrain_y=None):
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
    if pretrain_X is not None:
        Xtr = np.concatenate([pretrain_X, Xs])
        ytr = np.concatenate([pretrain_y, ys])
    else:
        Xtr, ytr = Xs, ys
    if len(np.unique(ytr)) < 2:
        return None
    try:
        model = model_factory()
        model.fit(Xtr, ytr)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(Xte)[:, 1]
        else:
            p = model.predict(Xte).astype(float)
    except Exception:
        return None
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(yte, pred, zero_division=0),
        "auc": roc_auc_score(yte, p) if len(np.unique(yte)) == 2 else float("nan"),
    }


def main():
    features = load_features()

    # Build concat-source pretrain data once
    src_data = {b: load_basin(b, features) for b in SOURCE}
    Xs_concat = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys_concat = np.concatenate([y for _, y in src_data.values()], axis=0)
    src_mu = Xs_concat.mean(0); src_sd = np.maximum(Xs_concat.std(0), 1e-6)
    Xs_concat_n = ((Xs_concat - src_mu) / src_sd).astype(np.float32)

    target_data = {b: load_target_with_coords(b, features) for b in TARGET}
    models = make_models()
    print(f"Classical models: {list(models.keys())}")

    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n========== seed={seed} ==========")
        for tgt in TARGET:
            X, y, coords = target_data[tgt]
            X_std = standardize_basin(X)
            if len(X) < K_FOLDS * 4:
                continue
            km = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10)
            folds = km.fit_predict(coords)
            for fold in range(K_FOLDS):
                test_mask = folds == fold
                train_mask = ~test_mask
                for K in KS:
                    if K * 2 > train_mask.sum():
                        continue
                    for model_name, factory in models.items():
                        for mode in ["independent", "concat_source"]:
                            rng = np.random.default_rng(
                                seed * 1000 + fold * 100 + K
                                + hash(model_name + mode + tgt) % 100)
                            for ep in range(N_EPISODES_PER_FOLD):
                                if mode == "concat_source":
                                    m = evaluate_fold(factory, X_std, y, train_mask,
                                                      test_mask, K, rng,
                                                      pretrain_X=Xs_concat_n,
                                                      pretrain_y=ys_concat)
                                else:
                                    m = evaluate_fold(factory, X_std, y, train_mask,
                                                      test_mask, K, rng)
                                if m is None:
                                    continue
                                rows.append({"target": tgt, "fold": fold, "K": K,
                                             "model": model_name, "mode": mode,
                                             "seed": seed, "episode": ep,
                                             "f1": m["f1"], "auc": m["auc"]})
                print(f"  {tgt:<10} fold={fold} done")

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

    # Plot best classical (concat_source) vs meta-learning under spatial CV
    bench_path = ROOT / "results/spatial_benchmark/summary.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
        target_labels = {"copiapo": "Copiapó", "huasco": "Huasco",
                         "elqui": "Elqui", "limari": "Limarí"}
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for ax, tgt in zip(axes.flat, TARGET):
            sub_c = summary[summary.target == tgt]
            for model, color in [("logreg", "#999999"), ("rf", "#984ea3"),
                                 ("xgb", "#a6611a"), ("catboost", "#dfc27d")]:
                # best mode per model
                d_best = (sub_c[sub_c.model == model]
                          .groupby("K", as_index=False)
                          .apply(lambda x: x.loc[x["f1_mean"].idxmax()])
                          .sort_values("K"))
                if d_best.empty:
                    continue
                ax.plot(d_best["K"], d_best["f1_mean"], marker="^",
                        color=color, linewidth=1.4, markersize=5,
                        label=f"{model} (best)" if tgt == "copiapo" else None)
            sub_m = bench[bench.target == tgt]
            for m, color in [("independent", "#888888"),
                             ("fomaml", "#c51b8a"), ("reptile", "#d95f0e")]:
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
        fig.legend(h, l, loc="center right", bbox_to_anchor=(1.18, 0.5),
                   frameon=False, fontsize=8)
        fig.suptitle("Classical ML vs meta-learning (spatial CV, apples-to-apples)", y=1.02)
        fig.tight_layout()
        out = FIGS / "classical_spatial_vs_meta.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out.name}")


if __name__ == "__main__":
    main()
