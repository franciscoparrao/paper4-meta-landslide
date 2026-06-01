#!/usr/bin/env python3
"""Paper 4 — empirical variogram validation of spatial CV folds (ISPRS JPRS R1 fix).

For each target basin:
  1. Train a baseline classifier (Independent MLP) on full basin data.
  2. Compute residuals (y - pred) at all pixels.
  3. Estimate empirical variogram of residuals.
  4. Fit a spherical model and extract the autocorrelation range (effective range).
  5. Report K-means fold centroids distance vs range — flag if folds are smaller than range.

Then re-run spatial CV with K ∈ {5, 10, 20} folds and compare F1 metrics.
This addresses reviewer concern about K=5 being arbitrarily chosen.

Output:
- results/variogram/variogram_per_basin.csv   (range_km, sill, nugget per basin)
- results/variogram/fold_size_comparison.csv  (median nearest-neighbor distance between folds)
- results/variogram/k_sensitivity.csv         (F1 mean at K_folds ∈ {5,10,20} for Independent + FOMAML)
- figures/variogram_per_basin.pdf
- figures/k_folds_sensitivity.pdf
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
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, MLP, bootstrap_ci, evaluate_query, fit_inner, fomaml_train,
    load_basin, load_features, read_h5, standardize_basin,
)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "variogram"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 7]
K_FOLDS_LIST = [5, 10, 20]
ADAPT_LR = 1e-2
ADAPT_STEPS = 30
META_OUTER = 300
META_K = 10
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
N_QUERY = 10


def load_with_coords(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def train_independent(X, y, n_in, steps=100):
    """Train an MLP on full data to compute residuals."""
    model = MLP(n_in).to(DEVICE)
    return fit_inner(model, X.astype(np.float32), y.astype(np.float32),
                     lr=1e-3, steps=steps)


def empirical_variogram(coords, residuals, n_bins=20, max_dist_pct=0.4):
    """Compute classical isotropic empirical variogram γ(h) = 0.5 E[(z_i - z_j)^2].

    Returns (bin_centers_km, semivariance, n_pairs_per_bin).
    """
    if len(coords) > 3000:
        # Subsample to keep pdist tractable; spatial coverage maintained
        rng = np.random.default_rng(0)
        sel = rng.choice(len(coords), size=3000, replace=False)
        coords = coords[sel]; residuals = residuals[sel]
    dists = pdist(coords)  # meters
    vals = pdist(residuals.reshape(-1, 1), metric="sqeuclidean") * 0.5
    max_d = np.quantile(dists, max_dist_pct)
    bin_edges = np.linspace(0, max_d, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000.0  # km
    semivar = np.full(n_bins, np.nan)
    n_pairs = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if m.sum() > 10:
            semivar[i] = np.nanmean(vals[m])
            n_pairs[i] = int(m.sum())
    return bin_centers, semivar, n_pairs


def fit_spherical(h_km, gamma):
    """Fit a spherical variogram model: γ(h) = c0 + c*(1.5*h/a - 0.5*(h/a)^3) for h<=a, else c0+c."""
    valid = ~np.isnan(gamma)
    if valid.sum() < 4:
        return {"nugget": np.nan, "sill": np.nan, "range_km": np.nan}
    from scipy.optimize import curve_fit

    def model(h, nugget, sill_partial, rng):
        rng = max(rng, 1e-3)
        out = np.empty_like(h)
        within = h <= rng
        out[within] = nugget + sill_partial * (1.5 * h[within] / rng - 0.5 * (h[within] / rng) ** 3)
        out[~within] = nugget + sill_partial
        return out

    try:
        p0 = [gamma[valid].min(), gamma[valid].max() - gamma[valid].min(), h_km[valid].max() / 2]
        popt, _ = curve_fit(model, h_km[valid], gamma[valid], p0=p0,
                            bounds=([0, 0, 0.1], [np.inf, np.inf, h_km[valid].max() * 2]),
                            maxfev=5000)
        return {"nugget": float(popt[0]), "sill": float(popt[0] + popt[1]),
                "range_km": float(popt[2])}
    except Exception:
        return {"nugget": np.nan, "sill": np.nan, "range_km": np.nan}


def fold_size_stats(coords, folds, n_folds):
    """Return median nearest-distance between fold centroids (km)."""
    centroids = np.array([coords[folds == k].mean(axis=0) for k in range(n_folds)])
    d = squareform(pdist(centroids))
    d[d == 0] = np.inf
    nn = d.min(axis=1)
    return float(np.median(nn) / 1000.0)


def k_folds_sensitivity_one_basin(tgt, X, y, coords, src_std, n_in, k_folds, seed):
    """Re-run spatial CV with given k_folds; return F1 for Independent + FOMAML."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=k_folds, random_state=seed, n_init=10)
    folds = km.fit_predict(coords)
    X_std = standardize_basin(X)
    fom = fomaml_train(src_std, n_in, META_OUTER, META_K, N_QUERY,
                       5, META_INNER_LR, META_OUTER_LR, rng)
    fom_state = deepcopy(fom.state_dict())

    results = {"independent": [], "fomaml": []}
    for fold in range(k_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        if len(test_idx) < 4 or len(np.unique(y[test_idx])) < 2:
            continue
        for K in [5, 10]:  # focus on most informative K values
            train_idx = np.where(train_mask)[0]
            pos = train_idx[y[train_idx] == 1]; neg = train_idx[y[train_idx] == 0]
            if len(pos) < K or len(neg) < K:
                continue
            for _ in range(10):  # 10 episodes (lighter than main)
                sup = np.concatenate([
                    rng.choice(pos, K, replace=False),
                    rng.choice(neg, K, replace=False),
                ])
                for method, init_state in [("independent", None), ("fomaml", fom_state)]:
                    model = MLP(n_in).to(DEVICE)
                    if init_state is not None:
                        model.load_state_dict(init_state)
                    model = fit_inner(model, X_std[sup], y[sup],
                                      lr=ADAPT_LR, steps=ADAPT_STEPS)
                    m = evaluate_query(model, X_std[test_idx], y[test_idx])
                    results[method].append(m["f1"])
    return {m: float(np.mean(v)) if v else float("nan") for m, v in results.items()}


def main():
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}

    # 1+2+3+4: Variogram per target basin
    print("=== Empirical variogram analysis ===")
    vario_rows = []
    fig_vario, axes = plt.subplots(1, len(TARGET), figsize=(13, 3.2), sharey=True)
    for ax, tgt in zip(axes, TARGET):
        X, y, coords = load_with_coords(tgt, features)
        model = train_independent(standardize_basin(X), y, n_in)
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(standardize_basin(X)))).numpy()
        residuals = (y - pred).astype(np.float64)
        h_km, gamma, npairs = empirical_variogram(coords, residuals)
        fit = fit_spherical(h_km, gamma)
        vario_rows.append({"target": tgt, "n_pixels": len(X),
                           **fit, "max_lag_km": float(h_km.max())})
        ax.plot(h_km, gamma, "o", ms=4, color="#377eb8")
        if not np.isnan(fit["range_km"]):
            ax.axvline(fit["range_km"], color="red", ls="--",
                       label=f"range ≈ {fit['range_km']:.1f} km")
        ax.set_title(tgt.capitalize()); ax.set_xlabel("Lag distance (km)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Semivariance γ(h)")
    fig_vario.suptitle("Empirical variogram of Independent-model residuals", y=1.02)
    fig_vario.tight_layout()
    fig_vario.savefig(FIGS / "variogram_per_basin.pdf", bbox_inches="tight")
    fig_vario.savefig(FIGS / "variogram_per_basin.png", bbox_inches="tight", dpi=150)
    vario_df = pd.DataFrame(vario_rows)
    vario_df.to_csv(OUT / "variogram_per_basin.csv", index=False)
    print(vario_df)

    # 5: Fold-size comparison vs variogram range
    print("\n=== Fold-size comparison ===")
    fold_rows = []
    for tgt in TARGET:
        X, y, coords = load_with_coords(tgt, features)
        for k in K_FOLDS_LIST:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
            nn_km = fold_size_stats(coords, km, k)
            fold_rows.append({"target": tgt, "k_folds": k,
                              "median_fold_centroid_dist_km": nn_km})
    fold_df = pd.DataFrame(fold_rows)
    fold_df = fold_df.merge(vario_df[["target", "range_km"]], on="target", how="left")
    fold_df["block_exceeds_range"] = fold_df["median_fold_centroid_dist_km"] > fold_df["range_km"]
    fold_df.to_csv(OUT / "fold_size_comparison.csv", index=False)
    print(fold_df)

    # 6: K-folds sensitivity for Independent + FOMAML (parallelized over tuples)
    print("\n=== K-folds sensitivity (parallelized) ===")
    import os
    n_jobs = int(os.environ.get("PAPER4_N_JOBS", "-1"))

    jobs = []
    for tgt in TARGET:
        X, y, coords = load_with_coords(tgt, features)
        if len(X) < 20:
            continue
        for k in K_FOLDS_LIST:
            for seed in SEEDS:
                jobs.append((tgt, X, y, coords, k, seed))

    def _one(tgt, X, y, coords, k, seed):
        res = k_folds_sensitivity_one_basin(tgt, X, y, coords, src_std, n_in, k, seed)
        print(f"  {tgt:<10} k_folds={k} seed={seed} done", flush=True)
        return {"target": tgt, "k_folds": k, "seed": seed,
                "f1_independent": res["independent"],
                "f1_fomaml": res["fomaml"]}

    sens_rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_one)(*j) for j in jobs
    )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT / "k_sensitivity.csv", index=False)
    print(sens_df.groupby(["target", "k_folds"])[["f1_independent", "f1_fomaml"]].mean())

    # Sensitivity figure
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = sens_df.groupby(["target", "k_folds"])[["f1_independent", "f1_fomaml"]].mean().reset_index()
    width = 0.2
    offsets = {5: -1.5, 10: -0.5, 20: 0.5}
    for k in K_FOLDS_LIST:
        sub = agg[agg.k_folds == k]
        x = np.arange(len(sub)) + offsets[k] * width
        ax.bar(x, sub["f1_fomaml"] - sub["f1_independent"], width=width,
               label=f"K_folds={k}")
    ax.set_xticks(np.arange(len(TARGET)))
    ax.set_xticklabels([t.capitalize() for t in TARGET])
    ax.set_ylabel("FOMAML $-$ Independent (mean $F_1$ lift)")
    ax.set_title("Meta-learning lift sensitivity to spatial CV K_folds")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "k_folds_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "k_folds_sensitivity.png", bbox_inches="tight", dpi=150)
    print(f"Figure → {FIGS / 'k_folds_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
