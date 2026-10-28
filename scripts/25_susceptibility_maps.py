#!/usr/bin/env python3
"""Paper 4 — full-basin susceptibility maps from K=1 adapted models.

Produces the map-level evidence behind the K-shot benchmark (checklist A1):
for each target basin (Copiapo, Elqui), predict per-pixel susceptibility with
  - Independent K=1: MLP trained from random init on 1 pos + 1 neg (30 Adam steps)
  - Reptile K=1:     meta-trained init (10 sources) adapted identically
Each map is the mean sigmoid output over SEEDS x N_EPISODES adaptations, so it
reflects the expected operational product rather than a single lucky support draw.

Protocol mirrors 16_spatial_benchmark.py exactly (same _meta_lib functions,
same hyperparameters, same per-basin sample standardization).

Output:
- results/susceptibility_maps/{basin}_{method}_k1_mean.tif  (float32, EPSG:32719)
- results/susceptibility_maps/map_run_metadata.csv
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, MLP, fit_inner, load_basin, load_features, read_h5, reptile_train,
)

ROOT = Path("/mnt/kingston/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
PAPER1 = Path("/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad")
FACTORS = PAPER1 / "factors"
WORLDCLIM = FACTORS / "_worldclim" / "wc2.1_2.5m_bio_12.tif"
BNA = PAPER1 / "Cuencas_BNA" / "Cuencas_BNA.shp"
OUT = ROOT / "results" / "susceptibility_maps"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
# BNA codes verified by max-overlap of the clipped DEM with Cuencas_BNA.shp
TARGETS = {
    "copiapo": ("05_rio_copiapo", "034"),
    "elqui":   ("07_rio_elqui", "043"),
}
SEEDS = [42, 123, 7]
N_EPISODES = 20
K = 1

# Meta-train + adapt configs (identical to 16_spatial_benchmark.py)
META_OUTER = 300
META_K = 10
META_INNER_STEPS = 5
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3
REPTILE_EPS = 0.1
ADAPT_LR = 1e-2
ADAPT_STEPS = 30
CHUNK_ROWS = 512


def raster_path(slug: str, feat: str) -> Path:
    """Resolve a CFS feature name to its GeoTIFF (paper1 factors or factors_extra)."""
    cat, name = feat.split("__", 1)
    sub = {"terrain": "terrain", "focal_stats": "focal_stats", "texture": "texture"}[cat]
    fname = f"{name}.tif"
    for root in (FACTORS / slug, ROOT / "factors_extra" / slug):
        p = root / sub / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"{slug}: {feat}")


def load_feature_stack(basin: str, slug: str, features: list[str]):
    """Read the 11 CFS rasters aligned to the basin DEM grid + basin mask."""
    dem = rasterio.open(FACTORS / slug / "dem_30m.tif")
    shape, transform, crs = dem.shape, dem.transform, dem.crs
    stack = np.full((len(features), *shape), np.nan, dtype=np.float32)

    for i, feat in enumerate(features):
        if feat == "climate__bio_12":
            with rasterio.open(WORLDCLIM) as src:
                reproject(
                    source=rasterio.band(src, 1), destination=stack[i],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=crs,
                    resampling=Resampling.bilinear, dst_nodata=np.nan,
                )
        else:
            with rasterio.open(raster_path(slug, feat)) as src:
                assert src.shape == shape, f"{feat} grid mismatch {src.shape} vs {shape}"
                band = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    band[band == src.nodata] = np.nan
                stack[i] = band

    bna = gpd.read_file(BNA).to_crs(crs)
    cod = TARGETS[basin][1]
    geom = bna[bna["COD_CUEN"].astype(str).str.zfill(3) == cod].geometry
    inside = ~geometry_mask(geom, out_shape=shape, transform=transform, invert=False)
    valid = inside & np.isfinite(stack).all(axis=0)
    return stack, valid, transform, crs


def adapted_models(basin: str, features: list[str], src_std: dict):
    """Yield (method, seed, episode, model) for K=1 adaptations, benchmark protocol."""
    X, y = load_basin(basin, features)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    X_std = ((X - mu) / sd).astype(np.float32)
    n_in = X.shape[1]
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]

    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        rep_state = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER_STEPS,
            META_INNER_LR, REPTILE_EPS, rng,
        ).state_dict())
        for ep in range(N_EPISODES):
            sup = np.concatenate([rng.choice(pos_idx, K, replace=False),
                                  rng.choice(neg_idx, K, replace=False)])
            Xs, ys = X_std[sup], y[sup].astype(np.float32)
            for method, state in [("independent", None), ("reptile", rep_state)]:
                model = MLP(n_in).to(DEVICE)
                if state is not None:
                    model.load_state_dict(state)
                model = fit_inner(model, Xs, ys, lr=ADAPT_LR, steps=ADAPT_STEPS)
                model.eval()
                models.append((method, seed, ep, model))
    return models, mu, sd


def predict_mean_maps(stack, valid, mu, sd, models):
    """Mean sigmoid over all adapted models, chunked by rows."""
    n_feat, H, W = stack.shape
    sums = {"independent": np.zeros((H, W), dtype=np.float64),
            "reptile": np.zeros((H, W), dtype=np.float64)}
    counts = {"independent": 0, "reptile": 0}
    for m, *_ in models:
        counts[m] += 1

    with torch.no_grad():
        for r0 in range(0, H, CHUNK_ROWS):
            r1 = min(r0 + CHUNK_ROWS, H)
            vmask = valid[r0:r1]
            if not vmask.any():
                continue
            feats = stack[:, r0:r1, :][:, vmask].T  # (n_valid, n_feat)
            feats = ((feats - mu) / sd).astype(np.float32)
            ft = torch.from_numpy(feats)
            for method, _, _, model in models:
                p = torch.sigmoid(model(ft)).numpy()
                block = sums[method][r0:r1]
                block[vmask] += p
    maps = {}
    for method in sums:
        m = (sums[method] / counts[method]).astype(np.float32)
        m[~valid] = np.nan
        maps[method] = m
    return maps


def main():
    only = set(sys.argv[1:])  # optional basin filter, e.g. `25_susceptibility_maps.py copiapo`
    features = load_features()
    print(f"features ({len(features)}): {features}")
    src_std = {}
    for b in SOURCE:
        X, y = load_basin(b, features)
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        src_std[b] = (((X - mu) / sd).astype(np.float32), y)

    meta_rows = []
    for basin, (slug, cod) in TARGETS.items():
        if only and basin not in only:
            continue
        print(f"\n=== {basin} ===")
        stack, valid, transform, crs = load_feature_stack(basin, slug, features)
        print(f"  grid {stack.shape[1:]} | valid pixels: {valid.sum()/1e6:.1f} M")
        models, mu, sd = adapted_models(basin, features, src_std)
        print(f"  adapted models: {len(models)} ({len(SEEDS)} seeds x {N_EPISODES} episodes x 2 methods)")
        maps = predict_mean_maps(stack, valid, mu, sd, models)
        for method, arr in maps.items():
            out = OUT / f"{basin}_{method}_k1_mean.tif"
            with rasterio.open(
                out, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                count=1, dtype="float32", crs=crs, transform=transform,
                nodata=np.nan, compress="deflate", tiled=True,
            ) as dst:
                dst.write(arr, 1)
            q = np.nanpercentile(arr, [5, 50, 95])
            print(f"  {method:12s} -> {out.name}  p5/p50/p95 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")
            meta_rows.append({"basin": basin, "method": method, "K": K,
                              "seeds": len(SEEDS), "episodes": N_EPISODES,
                              "n_models_averaged": len(models) // 2,
                              "valid_pixels": int(valid.sum()),
                              "p5": q[0], "p50": q[1], "p95": q[2]})
    pd.DataFrame(meta_rows).to_csv(OUT / "map_run_metadata.csv", index=False)
    print(f"\nOK -> {OUT}")


if __name__ == "__main__":
    main()
