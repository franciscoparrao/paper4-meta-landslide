#!/usr/bin/env python3
"""Paper 4 — sensitivity of the K=1 advantage to negative-sample selection (B1/M3).

Eight of fourteen basins (including all four targets) lack explicit non-landslide
points, so negatives are drawn randomly within the basin outside a 500 m buffer of
positives. A known concern in susceptibility modelling is that results depend on this
choice. We test robustness by re-generating the target negatives under:
  - exclusion buffer in {500 m (baseline), 1 km, 2 km};
  - a terrain-aware alternative: slope-matched negatives whose slope distribution
    equals that of the positives (controls the "steep vs flat" confound).
For each condition we re-run the K=1 spatial-CV protocol (Independent vs Reptile),
reusing the source-trained meta-model (sources are unchanged), and report the
meta-learning lift. Features are re-extracted from rasters, so absolute F1 may differ
slightly from Table 2; the comparison across conditions is internally consistent.

Output: results/stats/negatives_sensitivity.csv
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta_lib import (
    MLP, evaluate_query, fit_inner, load_basin, load_features, read_h5,
    reptile_train, standardize_basin,
)

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "ml_ready"
FACT = Path("/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad/factors")
WORLDCLIM = FACT / "_worldclim" / "wc2.1_2.5m_bio_12.tif"
OUT = ROOT / "results" / "stats"

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles",
          "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGETS = {"huasco": "06_rio_huasco", "copiapo": "05_rio_copiapo"}
SLOPE_MATCHED_TARGETS = {"huasco"}
BUFFERS_M = [500, 1000, 2000]
SEEDS = [42, 123, 7]
K_FOLDS, N_EP, K = 5, 20, 1
META_OUTER, META_K, META_INNER, META_LR, EPS = 300, 10, 5, 1e-2, 0.1
ADAPT_LR, ADAPT_STEPS = 1e-2, 30
RES_M = 30


def raster_path(slug, feat):
    cat, name = feat.split("__", 1)
    for root in (FACT / slug, ROOT / "factors_extra" / slug):
        p = root / cat / f"{name}.tif"
        if p.exists():
            return p
    raise FileNotFoundError(f"{slug}:{feat}")


def load_stack(slug, features):
    """Return CFS feature stack [F,H,W], valid mask, transform, slope [H,W]."""
    dem = rasterio.open(FACT / slug / "dem_30m.tif")
    shape, transform = dem.shape, dem.transform
    stack = np.full((len(features), *shape), np.nan, np.float32)
    for i, feat in enumerate(features):
        if feat == "climate__bio_12":
            with rasterio.open(WORLDCLIM) as src:
                reproject(rasterio.band(src, 1), stack[i], src_transform=src.transform,
                          src_crs=src.crs, dst_transform=transform, dst_crs=dem.crs,
                          resampling=Resampling.bilinear, dst_nodata=np.nan)
        else:
            with rasterio.open(raster_path(slug, feat)) as src:
                b = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    b[b == src.nodata] = np.nan
                stack[i] = b
    valid = np.isfinite(stack).all(0)
    with rasterio.open(raster_path(slug, "terrain__slope")) as src:
        slope = src.read(1).astype(np.float32)
        if src.nodata is not None:
            slope[slope == src.nodata] = np.nan
    return stack, valid, transform, slope


def gen_negatives_buffer(pos_rc, valid, n, buffer_m, rng):
    pos_mask = np.zeros(valid.shape, bool)
    for r, c in pos_rc:
        pos_mask[r, c] = True
    excl = binary_dilation(pos_mask, iterations=max(1, round(buffer_m / RES_M)))
    cand = np.argwhere(valid & ~excl)
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return cand[idx]


def gen_negatives_slopematched(pos_rc, valid, slope, n, rng, buffer_m=500):
    """Negatives outside the buffer whose slope matches the positives' distribution."""
    pos_mask = np.zeros(valid.shape, bool)
    for r, c in pos_rc:
        pos_mask[r, c] = True
    excl = binary_dilation(pos_mask, iterations=max(1, round(buffer_m / RES_M)))
    ok = valid & ~excl & np.isfinite(slope)
    cand = np.argwhere(ok)
    cand_slope = slope[ok]
    pos_slope = np.array([slope[r, c] for r, c in pos_rc if np.isfinite(slope[r, c])])
    edges = np.quantile(pos_slope, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    pos_hist, _ = np.histogram(pos_slope, edges)
    want = np.round(pos_hist / pos_hist.sum() * n).astype(int)
    picks = []
    cbin = np.digitize(cand_slope, edges[1:-1])
    for b in range(5):
        pool = np.where(cbin == b)[0]
        if len(pool) == 0 or want[b] == 0:
            continue
        picks.append(rng.choice(pool, size=min(want[b], len(pool)), replace=False))
    picks = np.concatenate(picks) if picks else np.array([], int)
    return cand[picks]


def feats_at(stack, transform, pos_rc, neg_rc):
    rc = np.vstack([pos_rc, neg_rc])
    X = stack[:, rc[:, 0], rc[:, 1]].T.astype(np.float32)
    y = np.concatenate([np.ones(len(pos_rc)), np.zeros(len(neg_rc))]).astype(np.float32)
    xy = np.array([transform * (c + 0.5, r + 0.5) for r, c in rc])
    keep = np.isfinite(X).all(1)
    return X[keep], y[keep], xy[keep]


def kshot_lift(Xn, y, coords, rep_state, n_in, seed):
    rng = np.random.default_rng(seed)
    folds = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
    res = {"independent": [], "reptile": []}
    for f in range(K_FOLDS):
        te = folds == f
        tr = ~te
        tri = np.where(tr)[0]
        pos, neg = tri[y[tri] == 1], tri[y[tri] == 0]
        tei = np.where(te)[0]
        if len(pos) < K or len(neg) < K or len(np.unique(y[tei])) < 2:
            continue
        for _ in range(N_EP):
            sup = np.concatenate([rng.choice(pos, K, False), rng.choice(neg, K, False)])
            for meth, st in [("independent", None), ("reptile", rep_state)]:
                m = MLP(n_in)
                if st is not None:
                    m.load_state_dict(st)
                m = fit_inner(m, Xn[sup], y[sup], lr=ADAPT_LR, steps=ADAPT_STEPS)
                res[meth].append(evaluate_query(m, Xn[tei], y[tei])["f1"])
    return {k: float(np.mean(v)) if v else np.nan for k, v in res.items()}


def main():
    features = load_features()
    n_in = len(features)
    src_std = {b: (standardize_basin(load_basin(b, features)[0]), load_basin(b, features)[1])
               for b in SOURCE}
    # meta-train once per seed (sources fixed across all conditions)
    rep_states = {}
    for seed in SEEDS:
        import torch
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"meta-training Reptile seed {seed}...")
        rep_states[seed] = deepcopy(reptile_train(
            src_std, n_in, META_OUTER, META_K, META_INNER, META_LR, EPS,
            np.random.default_rng(seed)).state_dict())

    rows = []
    for tgt, slug in TARGETS.items():
        with h5py.File(ML / f"{tgt}.h5") as f:
            yl = f["label"][:]
            pos_rc = np.column_stack([f["pixel_row"][:], f["pixel_col"][:]])[yl == 1]
        stack, valid, transform, slope = load_stack(slug, features)
        n_pos = len(pos_rc)
        conds = [("buffer_500m", lambda r: gen_negatives_buffer(pos_rc, valid, n_pos, 500, r)),
                 ("buffer_1km", lambda r: gen_negatives_buffer(pos_rc, valid, n_pos, 1000, r)),
                 ("buffer_2km", lambda r: gen_negatives_buffer(pos_rc, valid, n_pos, 2000, r))]
        if tgt in SLOPE_MATCHED_TARGETS:
            conds.append(("slope_matched",
                          lambda r: gen_negatives_slopematched(pos_rc, valid, slope, n_pos, r)))
        for cname, gen in conds:
            per_seed = {"independent": [], "reptile": []}
            for seed in SEEDS:
                neg_rc = gen(np.random.default_rng(seed * 7 + 1))
                X, y, coords = feats_at(stack, transform, pos_rc, neg_rc)
                Xn = standardize_basin(X)
                r = kshot_lift(Xn, y, coords, rep_states[seed], n_in, seed)
                per_seed["independent"].append(r["independent"])
                per_seed["reptile"].append(r["reptile"])
            ind = np.nanmean(per_seed["independent"])
            rep = np.nanmean(per_seed["reptile"])
            rows.append({"target": tgt, "condition": cname, "n_pos": n_pos,
                         "f1_independent": ind, "f1_reptile": rep,
                         "lift_pp": (rep - ind) * 100})
            print(f"  {tgt:8s} {cname:14s}: Indep {ind:.3f}  Reptile {rep:.3f}  lift {(rep-ind)*100:+.1f} pp")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "negatives_sensitivity.csv", index=False)
    print("\n=== Negative-sampling sensitivity (K=1 spatial CV) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nlift range: {df.lift_pp.min():+.1f} to {df.lift_pp.max():+.1f} pp "
          f"(SD {df.lift_pp.std():.1f})")


if __name__ == "__main__":
    main()
