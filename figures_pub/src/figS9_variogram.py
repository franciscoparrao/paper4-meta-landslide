"""Figure S9 (supp): Empirical variogram of Independent-model residuals per target
basin, with fitted spherical model and autocorrelation range. Validates that the
K-means spatial CV folds achieve genuine spatial decorrelation.

Recomputes the empirical variogram locally (fast; data in data/ml_ready) and reads
the fitted parameters from results/variogram/variogram_per_basin.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "figures_pub" / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
from style import COLOR_WONG, JOURNAL_WIDTH, add_panel_label, save_pub, setup_style, style_ax
from _meta_lib import MLP, fit_inner, load_basin, load_features, standardize_basin, read_h5

ML = ROOT / "data" / "ml_ready"
VARIO_CSV = ROOT / "results" / "variogram" / "variogram_per_basin.csv"
OUT = ROOT / "figures_pub" / "out" / "figS9_variogram"

TARGETS = ["copiapo", "huasco", "elqui", "limari"]
TARGET_LABELS = {"copiapo": "Copiapó", "huasco": "Huasco",
                 "elqui": "Elqui", "limari": "Limarí"}


def load_with_coords(basin, features):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    return X, y, coords


def empirical_variogram(coords, residuals, n_bins=18, max_dist_pct=0.4, seed=0):
    if len(coords) > 3000:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(coords), size=3000, replace=False)
        coords, residuals = coords[sel], residuals[sel]
    dists = pdist(coords)
    vals = pdist(residuals.reshape(-1, 1), metric="sqeuclidean") * 0.5
    max_d = np.quantile(dists, max_dist_pct)
    edges = np.linspace(0, max_d, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]) / 1000.0
    sv = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (dists >= edges[i]) & (dists < edges[i + 1])
        if m.sum() > 10:
            sv[i] = np.nanmean(vals[m])
    return centers, sv


def spherical(h, nugget, partial, rng):
    rng = max(rng, 1e-3)
    out = np.empty_like(h)
    within = h <= rng
    out[within] = nugget + partial * (1.5 * h[within] / rng - 0.5 * (h[within] / rng) ** 3)
    out[~within] = nugget + partial
    return out


def main():
    setup_style(journal="rse", base_fontsize=9)
    features = load_features()
    n_in = len(features)
    vario_params = pd.read_csv(VARIO_CSV).set_index("target")

    w = JOURNAL_WIDTH["rse"]["double"]
    fig, axes = plt.subplots(1, 4, figsize=(w, w * 0.28), sharey=False)

    torch.manual_seed(0); np.random.seed(0)
    for i, (ax, tgt) in enumerate(zip(axes, TARGETS)):
        X, y, coords = load_with_coords(tgt, features)
        Xs = standardize_basin(X)
        model = fit_inner(MLP(n_in), Xs, y, lr=1e-3, steps=100)
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(Xs))).numpy()
        resid = (y - pred).astype(np.float64)
        h, sv = empirical_variogram(coords, resid)

        ax.plot(h, sv, "o", ms=4, color=COLOR_WONG["blue"],
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)

        rng_km = vario_params.loc[tgt, "range_km"]
        if np.isfinite(rng_km):
            valid = ~np.isnan(sv)
            try:
                popt, _ = curve_fit(
                    spherical, h[valid], sv[valid],
                    p0=[sv[valid].min(), sv[valid].max() - sv[valid].min(), rng_km],
                    bounds=([0, 0, 0.1], [np.inf, np.inf, h.max() * 2]), maxfev=5000)
                hh = np.linspace(0, h.max(), 100)
                ax.plot(hh, spherical(hh, *popt), "-", color=COLOR_WONG["red"],
                        lw=1.4, zorder=2)
            except Exception:
                pass
            ax.axvline(rng_km, color="#444444", ls="--", lw=0.9, zorder=1)
            ax.text(rng_km, ax.get_ylim()[1] * 0.96,
                    f" range\n {rng_km:.0f} km", fontsize=7, ha="left", va="top",
                    color="#444444")

        add_panel_label(ax, "abcd"[i], x=-0.18, y=1.08, fontsize=10)
        ax.text(0.96, 0.06, TARGET_LABELS[tgt], transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="right", va="bottom")
        ax.set_xlabel("Lag distance (km)")
        if i == 0:
            ax.set_ylabel("Semivariance $\\gamma(h)$")
        style_ax(ax, x_grid=False, y_grid=True)

    fig.tight_layout()
    save_pub(fig, OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
