#!/usr/bin/env python3
"""Paper 4 (R1) — diagnostics of the spatial-CV partitions (reviewer 3, comments 5, 7, 8).

Replicates the K-means partitions of 16_spatial_benchmark.py (same seeds, same
K_FOLDS, same random_state) and reports, per target basin:

  1. Partition overlap across seeds: adjusted Rand index (ARI) and the mean
     best-match Jaccard overlap of held-out sets  -> results/stats/fold_overlap.csv
  2. Excluded (seed, fold) partitions and why, for every K in KS, with the
     held-out cluster's class counts, area proxy and centroid
                                                  -> results/stats/fold_validity.csv
  3. Query-to-training nearest-neighbour distances per (seed, fold): the
     distance from each held-out pixel to the nearest pixel of the training
     clusters (median, p05, min)                   -> results/stats/fold_nn_distance.csv
  4. Retained vs excluded regions: two-sample KS test per feature on the
     pixels of retained vs excluded held-out clusters (K=1 rule)
                                                  -> results/stats/fold_excluded_ks.csv
  5. Map of partitions per target x seed, excluded clusters hatched
                                                  -> figures_pub/out/figS_fold_partitions.pdf

Only needs data/ml_ready/*.h5 (coordinates, labels, features). No model runs.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import load_features, read_h5  # noqa: E402

ROOT = Path("/mnt/kingston/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "stats"
FIG = ROOT / "figures_pub" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
SEEDS = [42, 123, 7, 1729, 2718]  # identical to 16_spatial_benchmark.py (R1: five seeds)
K_FOLDS = 5
KS = [1, 5, 10, 20]
LABELS = {"copiapo": "Copiapó", "huasco": "Huasco", "elqui": "Elqui", "limari": "Limarí"}


def make_spatial_folds(coords, n_folds, seed):
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def best_match_jaccard(f_a, f_b, n_folds):
    """Mean over folds of A of the max Jaccard with any fold of B."""
    vals = []
    for a in range(n_folds):
        sa = set(np.flatnonzero(f_a == a))
        best = 0.0
        for b in range(n_folds):
            sb = set(np.flatnonzero(f_b == b))
            j = len(sa & sb) / max(len(sa | sb), 1)
            best = max(best, j)
        vals.append(best)
    return float(np.mean(vals))


def main():
    features = load_features()
    overlap_rows, validity_rows, nn_rows, ks_rows = [], [], [], []
    partitions = {}  # (target, seed) -> folds

    for tgt in TARGETS:
        df = read_h5(ML / f"{tgt}.h5")
        coords = df[["x_utm", "y_utm"]].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        X = df[features].to_numpy(dtype=float)
        n = len(df)

        folds_by_seed = {s: make_spatial_folds(coords, K_FOLDS, s) for s in SEEDS}
        partitions.update({(tgt, s): f for s, f in folds_by_seed.items()})

        # 1. overlap across seeds
        for s1, s2 in combinations(SEEDS, 2):
            f1, f2 = folds_by_seed[s1], folds_by_seed[s2]
            overlap_rows.append({
                "target": tgt, "seed_a": s1, "seed_b": s2, "n_pixels": n,
                "ari": adjusted_rand_score(f1, f2),
                "mean_best_jaccard": best_match_jaccard(f1, f2, K_FOLDS),
            })

        # 2./3. validity and NN distance per (seed, fold)
        excluded_mask_k1 = np.zeros(n, dtype=bool)
        retained_mask_k1 = np.zeros(n, dtype=bool)
        for s in SEEDS:
            folds = folds_by_seed[s]
            for fold in range(K_FOLDS):
                test = folds == fold
                train = ~test
                n_pos_te, n_neg_te = int((y[test] == 1).sum()), int((y[test] == 0).sum())
                n_pos_tr, n_neg_tr = int((y[train] == 1).sum()), int((y[train] == 0).sum())
                cx, cy = coords[test].mean(0)
                # area proxy: bounding-box area of the held-out pixels (km^2)
                span = coords[test].max(0) - coords[test].min(0)
                bbox_km2 = float(span[0] * span[1]) / 1e6
                # NN distance from each held-out pixel to the nearest training pixel
                tree = cKDTree(coords[train])
                d, _ = tree.query(coords[test], k=1)
                d_km = d / 1000.0
                nn_rows.append({
                    "target": tgt, "seed": s, "fold": fold, "n_test": int(test.sum()),
                    "n_train": int(train.sum()),
                    "nn_min_km": float(d_km.min()), "nn_p05_km": float(np.percentile(d_km, 5)),
                    "nn_median_km": float(np.median(d_km)), "nn_mean_km": float(d_km.mean()),
                    "frac_test_within_1km": float((d_km < 1).mean()),
                    "frac_test_within_5km": float((d_km < 5).mean()),
                })
                for K in KS:
                    reason = []
                    if n_pos_te == 0 or n_neg_te == 0:
                        reason.append("held-out cluster single-class")
                    if n_pos_tr < K or n_neg_tr < K:
                        reason.append(f"training clusters < {K} per class")
                    # 16_spatial_benchmark also skips K if 2K > n_train
                    if 2 * K > train.sum():
                        reason.append("2K > n_train")
                    valid = len(reason) == 0
                    validity_rows.append({
                        "target": tgt, "seed": s, "fold": fold, "K": K, "valid": valid,
                        "reason": "; ".join(reason) if reason else "",
                        "n_test": int(test.sum()), "n_pos_test": n_pos_te, "n_neg_test": n_neg_te,
                        "n_pos_train": n_pos_tr, "n_neg_train": n_neg_tr,
                        "centroid_x_utm": float(cx), "centroid_y_utm": float(cy),
                        "bbox_km2": bbox_km2,
                    })
                    if K == 1:
                        (retained_mask_k1 if valid else excluded_mask_k1)[test] = True

        # 4. KS retained vs excluded (K=1 rule), pooled over seeds
        if excluded_mask_k1.any() and retained_mask_k1.any():
            for j, f in enumerate(features):
                stat, p = ks_2samp(X[retained_mask_k1, j], X[excluded_mask_k1, j])
                ks_rows.append({
                    "target": tgt, "feature": f, "ks_stat": float(stat), "p_value": float(p),
                    "n_retained_pixels": int(retained_mask_k1.sum()),
                    "n_excluded_pixels": int(excluded_mask_k1.sum()),
                    "mean_retained": float(X[retained_mask_k1, j].mean()),
                    "mean_excluded": float(X[excluded_mask_k1, j].mean()),
                })
        else:
            ks_rows.append({"target": tgt, "feature": "(none excluded at K=1)", "ks_stat": np.nan,
                            "p_value": np.nan, "n_retained_pixels": int(retained_mask_k1.sum()),
                            "n_excluded_pixels": int(excluded_mask_k1.sum()),
                            "mean_retained": np.nan, "mean_excluded": np.nan})

    ov = pd.DataFrame(overlap_rows); ov.to_csv(OUT / "fold_overlap.csv", index=False)
    va = pd.DataFrame(validity_rows); va.to_csv(OUT / "fold_validity.csv", index=False)
    nn = pd.DataFrame(nn_rows); nn.to_csv(OUT / "fold_nn_distance.csv", index=False)
    ks = pd.DataFrame(ks_rows); ks.to_csv(OUT / "fold_excluded_ks.csv", index=False)

    # ---- console summary ----
    print("\n== Partition overlap across seeds (ARI / mean best-match Jaccard) ==")
    print(ov.groupby("target")[["ari", "mean_best_jaccard"]].agg(["mean", "min", "max"]).round(3))

    print("\n== Valid (seed, fold) partitions per target and K ==")
    print(va.groupby(["target", "K"])["valid"].sum().unstack("K"))
    print("\n== Exclusion reasons (K=1) ==")
    print(va[(va.K == 1) & (~va.valid)].groupby(["target", "reason"]).size())

    print("\n== Query->training NN distance per target (km), over valid K=1 partitions ==")
    v1 = va[(va.K == 1) & va.valid][["target", "seed", "fold"]]
    nnv = nn.merge(v1, on=["target", "seed", "fold"])
    print(nnv.groupby("target")[["nn_min_km", "nn_p05_km", "nn_median_km",
                                 "frac_test_within_1km", "frac_test_within_5km"]]
          .agg(["median", "min", "max"]).round(3).T)

    print("\n== Retained vs excluded held-out regions (K=1): features with KS p<0.05 ==")
    sig = ks[ks.p_value < 0.05]
    print(sig[["target", "feature", "ks_stat", "p_value", "mean_retained", "mean_excluded"]].round(4)
          .to_string(index=False) if len(sig) else "  none")

    # ---- 5. map ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(TARGETS), len(SEEDS), figsize=(3.2 * len(SEEDS), 3.0 * len(TARGETS)),
                                 squeeze=False)
        cmap = plt.get_cmap("tab10")
        for i, tgt in enumerate(TARGETS):
            df = read_h5(ML / f"{tgt}.h5")
            coords = df[["x_utm", "y_utm"]].to_numpy(dtype=float) / 1000.0
            y = df["label"].to_numpy(dtype=int)
            for j, s in enumerate(SEEDS):
                ax = axes[i, j]
                folds = partitions[(tgt, s)]
                inval = set(va[(va.target == tgt) & (va.seed == s) & (va.K == 1) & (~va.valid)].fold)
                for fold in range(K_FOLDS):
                    m = folds == fold
                    col = cmap(fold)
                    ax.scatter(coords[m & (y == 0), 0], coords[m & (y == 0), 1], s=6, marker="o",
                               color=col, alpha=0.35 if fold in inval else 0.8, linewidths=0)
                    ax.scatter(coords[m & (y == 1), 0], coords[m & (y == 1), 1], s=10, marker="^",
                               color=col, alpha=0.35 if fold in inval else 1.0, edgecolors="k",
                               linewidths=0.3)
                    cx, cy = coords[m].mean(0)
                    ax.text(cx, cy, f"{fold}" + ("×" if fold in inval else ""), fontsize=8,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
                ax.set_aspect("equal")
                ax.set_title(f"{LABELS[tgt]} — seed {s}", fontsize=9)
                ax.tick_params(labelsize=6)
                if i == len(TARGETS) - 1:
                    ax.set_xlabel("UTM x (km)", fontsize=7)
                if j == 0:
                    ax.set_ylabel("UTM y (km)", fontsize=7)
        fig.suptitle("Spatial-CV partitions (K-means, 5 folds). Triangles: landslides; circles: "
                     "non-landslides.\nFaded clusters marked × are excluded at K=1 "
                     "(single-class held-out cluster or insufficient training support).", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(FIG / "figS_fold_partitions.pdf")
        fig.savefig(FIG / "figS_fold_partitions.png", dpi=150)
        print(f"\nMap → {FIG / 'figS_fold_partitions.pdf'}")
    except Exception as e:  # pragma: no cover
        print("Map skipped:", e)


if __name__ == "__main__":
    main()
