#!/usr/bin/env python3
"""Paper 4 — extract image patches for the spatial (U-Net/CNN) baseline.

For each basin, stack the 25 gridded feature rasters (terrain + hydrology + texture
+ focal stats) into a multi-channel array, then extract PATCH_SIZE x PATCH_SIZE
patches centred on each inventory point (positives + negatives) from samples/<basin>.h5.

This provides the spatial-context inputs that the point-wise MLP discards, enabling a
fair U-Net/CNN meta-learning baseline (ISPRS JPRS reviewer Issue #2).

Output: data/patches/<basin>.npz with X[N, C, P, P] float32, y[N], (rows, cols).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
FX = ROOT / "factors_extra"
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "data" / "patches"
OUT.mkdir(parents=True, exist_ok=True)

PATCH = 32  # 32 x 32 at 30 m = ~960 m window

# basin slug -> factors_extra subdir
BASIN_DIR = {
    "chanaral": "04_rio_salado", "taltal": "taltal",
    "tilviche": "02_costeras_tilviche_loa", "caracoles": "03_costeras_loa_caracoles",
    "copiapo": "05_rio_copiapo", "huasco": "06_rio_huasco",
    "elqui": "07_rio_elqui", "limari": "08_rio_limari",
    "choapa": "choapa", "maipo": "09_rio_maipo", "rapel": "10_rio_rapel",
    "maule": "11_rio_maule", "bueno_puelo": "14_costeras_bueno_puelo",
    "magallanes": "15_costeras_magallanes",
}

# Canonical channel order (25), consistent across all basins
CHANNELS = [
    ("terrain", "curvedness"), ("terrain", "landform"), ("terrain", "ls_factor"),
    ("terrain", "relative_slope_position"), ("terrain", "shape_index"),
    ("terrain", "slope_rad"), ("terrain", "surface_area_ratio"),
    ("terrain", "tpi_r21"), ("terrain", "tpi_r3"), ("terrain", "tpi_r9"),
    ("terrain", "valley_depth"),
    ("hydrology", "drainage_density"), ("hydrology", "sediment_connectivity"),
    ("texture", "glcm_contrast_dem"), ("texture", "glcm_correlation_dem"),
    ("texture", "glcm_dissimilarity_dem"), ("texture", "glcm_energy_dem"),
    ("texture", "glcm_entropy_dem"), ("texture", "glcm_homogeneity_dem"),
    ("focal_stats", "dem_range_r4"), ("focal_stats", "dem_std_r10"),
    ("focal_stats", "dem_std_r1"), ("focal_stats", "dem_std_r4"),
    ("focal_stats", "slope_mean_r4"), ("focal_stats", "slope_std_r4"),
]


def load_stack(basin_dir: Path):
    """Load all 25 channels into a [C, H, W] array on a common grid."""
    arrs, ref_transform, ref_shape, ref_crs, nodata0 = [], None, None, None, None
    for sub, name in CHANNELS:
        p = basin_dir / sub / f"{name}.tif"
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
            nd = src.nodata
            if ref_shape is None:
                ref_shape, ref_transform, ref_crs = a.shape, src.transform, src.crs
            elif a.shape != ref_shape:
                raise ValueError(f"shape mismatch {p}: {a.shape} vs {ref_shape}")
            if nd is not None:
                a = np.where(a == nd, np.nan, a)
        arrs.append(a)
    return np.stack(arrs, axis=0), ref_transform, ref_crs


def standardize(stack):
    """Per-channel z-score using finite values; NaN -> 0 after standardization."""
    out = np.empty_like(stack)
    for c in range(stack.shape[0]):
        ch = stack[c]
        m = np.nanmean(ch); s = np.nanstd(ch)
        s = s if s > 1e-6 else 1.0
        z = (ch - m) / s
        out[c] = np.nan_to_num(z, nan=0.0)
    return out


def extract(basin: str):
    bdir = FX / BASIN_DIR[basin]
    stack, transform, crs = load_stack(bdir)
    stack = standardize(stack)
    C, H, W = stack.shape

    with h5py.File(SAMPLES / f"{basin}.h5", "r") as f:
        x = f["x_utm"][:]; y = f["y_utm"][:]; lab = f["label"][:].astype(np.int64)

    inv = ~transform  # world -> pixel
    half = PATCH // 2
    patches, labels, rows, cols = [], [], [], []
    skipped = 0
    for xi, yi, li in zip(x, y, lab):
        col, row = inv * (xi, yi)
        row, col = int(round(row)), int(round(col))
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            # pad with edge reflection for border points
            pad = max(-r0, -c0, r1 - H, c1 - W, 0) + 1
            padded = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
            rr, cc = row + pad, col + pad
            patch = padded[:, rr - half:rr + half, cc - half:cc + half]
        else:
            patch = stack[:, r0:r1, c0:c1]
        if patch.shape != (C, PATCH, PATCH):
            skipped += 1
            continue
        patches.append(patch.astype(np.float32)); labels.append(li)
        rows.append(row); cols.append(col)

    X = np.stack(patches); yv = np.array(labels, dtype=np.int64)
    np.savez_compressed(OUT / f"{basin}.npz", X=X, y=yv,
                        rows=np.array(rows), cols=np.array(cols))
    print(f"  {basin:<12} patches={X.shape} pos={int((yv==1).sum())} "
          f"neg={int((yv==0).sum())} skipped={skipped}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(BASIN_DIR.keys())
    print(f"Extracting {PATCH}x{PATCH} x {len(CHANNELS)}-channel patches...")
    for b in targets:
        try:
            extract(b)
        except Exception as e:
            print(f"  [ERROR] {b}: {e}")


if __name__ == "__main__":
    main()
