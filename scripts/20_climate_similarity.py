#!/usr/bin/env python3
"""Paper 4 — climate-similarity analysis: does meta-learning advantage scale
with source-target climate similarity?

For each (source, target) pair:
- Compute climate distance: euclidean norm in (mean precip, mean elevation,
  mean temp annual range) standardized space.
- Cross-reference with the benchmark FOMAML F1 advantage at K=5 in target.

If climate similarity correlates with transfer advantage → confirms hypothesis
that climate-matched sources are more useful, supports the discussion.

Outputs:
- results/climate/distance_matrix.csv (source × target distance)
- figures/climate_distance_heatmap.png
- figures/transfer_vs_climate_distance.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import load_basin, load_features

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "climate"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]

CLIMATE_FEAT = "climate__bio_12"  # annual precipitation


def basin_climate_summary(basin):
    """Return mean precipitation for the basin (from sample-level data)."""
    with h5py.File(ML / f"{basin}.h5", "r") as f:
        if CLIMATE_FEAT in f:
            arr = f[CLIMATE_FEAT][:]
        else:
            return float("nan")
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def main():
    basins = SOURCE + TARGET
    precip = {b: basin_climate_summary(b) for b in basins}
    print("Mean annual precipitation per basin:")
    for b, p in precip.items():
        print(f"  {b:<12} {p:8.1f} mm/yr")

    # Distance matrix in 1D (precip)
    rows = []
    for s in SOURCE:
        for t in TARGET:
            d = abs(precip[s] - precip[t])
            rows.append({"source": s, "target": t, "precip_distance": d})
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "distance_matrix.csv", index=False)

    # Pivot for heatmap
    pivot = dist.pivot(index="source", columns="target", values="precip_distance")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(TARGET))); ax.set_xticklabels(TARGET, rotation=20)
    ax.set_yticks(range(len(SOURCE))); ax.set_yticklabels(SOURCE)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > pivot.values.mean() else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Δ annual precip (mm/yr)")
    ax.set_title("Source–target climate distance (precipitation)")
    fig.tight_layout()
    fig.savefig(FIGS / "climate_distance_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → climate_distance_heatmap.png")

    # Try to correlate with FOMAML transfer advantage from spatial benchmark
    bench_path = ROOT / "results/spatial_benchmark/summary.csv"
    if not bench_path.exists():
        print(f"\n[skip] {bench_path} not found — run benchmark first")
        return

    bench = pd.read_csv(bench_path)
    # Per target: FOMAML K=5 lift over independent (averaged across all source pretrains)
    target_advantage = {}
    for t in TARGET:
        sub = bench[(bench.target == t) & (bench.K == 5)]
        if sub.empty:
            continue
        f_fom = sub[sub.method == "fomaml"]["f1_mean"].mean()
        f_ind = sub[sub.method == "independent"]["f1_mean"].mean()
        target_advantage[t] = f_fom - f_ind

    # Per target: mean climate distance to source pool
    target_mean_dist = {t: dist[dist.target == t]["precip_distance"].mean()
                       for t in TARGET}

    rows = []
    for t in TARGET:
        if t not in target_advantage:
            continue
        rows.append({"target": t, "mean_dist": target_mean_dist[t],
                     "fomaml_lift_K5": target_advantage[t]})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT / "advantage_vs_distance.csv", index=False)
    print(f"\n  Per-target advantage vs mean climate distance:")
    print(corr_df.to_string(index=False))
    if len(corr_df) >= 3:
        r_p, p_p = pearsonr(corr_df["mean_dist"], corr_df["fomaml_lift_K5"])
        r_s, p_s = spearmanr(corr_df["mean_dist"], corr_df["fomaml_lift_K5"])
        print(f"  Pearson r={r_p:+.3f} (p={p_p:.3f})")
        print(f"  Spearman ρ={r_s:+.3f} (p={p_s:.3f})")

    # Scatter
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(corr_df["mean_dist"], corr_df["fomaml_lift_K5"],
               s=140, color="#c51b8a", edgecolor="black")
    for _, r in corr_df.iterrows():
        ax.annotate(r["target"], (r["mean_dist"], r["fomaml_lift_K5"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean climate distance to source pool (mm/yr)")
    ax.set_ylabel("FOMAML lift over Independent at K=5")
    ax.set_title("Transfer advantage vs source-target climate distance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "transfer_vs_climate_distance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → transfer_vs_climate_distance.png")


if __name__ == "__main__":
    main()
