#!/usr/bin/env python3
"""Paper 4 — sample inventory locations to DEM grid and extract feature values.

For each basin:
  1. Load unified inventory GPKG (data/inventarios/{basin}.gpkg).
  2. Reproject to DEM CRS (EPSG:32719 UTM 19S).
  3. Snap positives to DEM pixel centers (1 point = 1 unique pixel).
     - Polygons (Taltal): rasterize all_touched=True, take unique pixel centers.
  4. Get negatives:
     - If inventory has neg points: snap them.
     - Else: random sample within DEM valid mask, excluding 500m buffer around positives.
       Balanced: N_neg = N_pos.
  5. Extract feature values at each pixel center from all rasters in
     factors/{basin}/* and factors_extra/{basin}/*.
  6. Save to data/samples/{basin}.h5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation


def write_h5(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to HDF5 via h5py (pandas requires tables>=3.10 which we lack)."""
    with h5py.File(path, "w") as f:
        for col in df.columns:
            arr = df[col].to_numpy()
            if arr.dtype.kind in "fiub":  # numeric/bool
                f.create_dataset(col, data=arr, compression="gzip", compression_opts=4)
            else:
                f.create_dataset(col, data=arr.astype("S"), compression="gzip", compression_opts=4)


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
PAPER1 = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/factors")
INV_DIR = ROOT / "data" / "inventarios"
OUT_DIR = ROOT / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-basin metadata: (factors_root, paper1_slug_for_extras, dem_relative_path)
# factors_root: where the base factors (terrain/hydrology/spectral/...) live
# extras_slug:  used to find factors_extra/{slug}/
BASINS = {
    # Source basins (10 total: 4 original + 6 expanded)
    "chanaral":     (PAPER1 / "04_rio_salado", "04_rio_salado"),
    "taltal":       (ROOT / "factors" / "taltal", "taltal"),
    "maule":        (PAPER1 / "11_rio_maule", "11_rio_maule"),
    "choapa":       (ROOT / "factors" / "choapa", "choapa"),
    "tilviche":     (PAPER1 / "02_costeras_tilviche_loa", "02_costeras_tilviche_loa"),
    "caracoles":    (PAPER1 / "03_costeras_loa_caracoles", "03_costeras_loa_caracoles"),
    "maipo":        (PAPER1 / "09_rio_maipo", "09_rio_maipo"),
    "rapel":        (PAPER1 / "10_rio_rapel", "10_rio_rapel"),
    "bueno_puelo":  (PAPER1 / "14_costeras_bueno_puelo", "14_costeras_bueno_puelo"),
    "magallanes":   (PAPER1 / "15_costeras_magallanes", "15_costeras_magallanes"),
    # Target basins (4)
    "copiapo":      (PAPER1 / "05_rio_copiapo", "05_rio_copiapo"),
    "huasco":       (PAPER1 / "06_rio_huasco", "06_rio_huasco"),
    "elqui":        (PAPER1 / "07_rio_elqui", "07_rio_elqui"),
    "limari":       (PAPER1 / "08_rio_limari", "08_rio_limari"),
}

NEG_BUFFER_M = 500     # exclusion buffer around positives when generating negatives
NEG_RATIO = 1.0         # negatives = NEG_RATIO * positives
NEG_EXPAND_M = 60       # buffer around inventory neg points to add neighboring pixels
MAX_PIXELS_PER_POLYGON = 30  # cap pixels per positive polygon to avoid imbalance + spatial leak
SEED = 42


def iter_feature_rasters(factors_root: Path, extras_slug: str) -> Iterator[tuple[str, Path]]:
    """Yield (feature_name, raster_path) for all .tif under base + extras."""
    extras_root = ROOT / "factors_extra" / extras_slug
    seen = set()
    for root in (factors_root, extras_root):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.tif")):
            # Skip the DEM at top level (we already use it as reference grid)
            if p.name in ("dem.tif", "dem_30m.tif", "dem_wgs84.tif"):
                continue
            # Feature name: parent_subdir + filename stem (e.g. "terrain/slope")
            try:
                rel = p.relative_to(root)
                if len(rel.parts) >= 2:
                    name = f"{rel.parts[-2]}__{p.stem}"
                else:
                    name = p.stem
            except ValueError:
                name = p.stem
            if name in seen:
                # Disambiguate base vs extras by prefix
                name = f"extra__{name}" if root == extras_root else f"base__{name}"
            seen.add(name)
            yield name, p


def snap_points_to_pixels(geom_iter, dem_src) -> set[tuple[int, int]]:
    """For each point geometry, return (row, col) of containing pixel."""
    pixels = set()
    h, w = dem_src.height, dem_src.width
    for g in geom_iter:
        if g is None or g.is_empty:
            continue
        r, c = dem_src.index(g.x, g.y)
        if 0 <= r < h and 0 <= c < w:
            pixels.add((int(r), int(c)))
    return pixels


def rasterize_polygon_pixels(polygon_iter, dem_src, max_per_polygon=None,
                             rng=None) -> set[tuple[int, int]]:
    """Rasterize each polygon individually; optionally cap pixels per polygon."""
    polys = [g for g in polygon_iter if g is not None and not g.is_empty]
    if not polys:
        return set()
    pixels: set[tuple[int, int]] = set()
    for g in polys:
        mask = rasterize(
            [(g, 1)],
            out_shape=(dem_src.height, dem_src.width),
            transform=dem_src.transform,
            all_touched=True,
            dtype="uint8",
        )
        rc = np.argwhere(mask > 0)
        if max_per_polygon is not None and len(rc) > max_per_polygon:
            idx = rng.choice(len(rc), size=max_per_polygon, replace=False)
            rc = rc[idx]
        for r, c in rc:
            pixels.add((int(r), int(c)))
    return pixels


def expand_negatives_with_buffer(neg_pixels, buffer_pixels, dem_data, dem_nodata,
                                 exclusion_pixels=None):
    """For each negative pixel, add neighboring pixels within `buffer_pixels` radius.

    Excludes pixels in `exclusion_pixels` (positives) and invalid DEM cells.
    """
    if not neg_pixels or buffer_pixels <= 0:
        return set(neg_pixels)
    valid_mask = np.isfinite(dem_data)
    if dem_nodata is not None:
        valid_mask &= (dem_data != dem_nodata)
    neg_mask = np.zeros_like(valid_mask, dtype=bool)
    for r, c in neg_pixels:
        neg_mask[r, c] = True
    expanded = binary_dilation(neg_mask, iterations=buffer_pixels)
    expanded &= valid_mask
    if exclusion_pixels:
        for r, c in exclusion_pixels:
            expanded[r, c] = False
    rows, cols = np.where(expanded)
    return {(int(r), int(c)) for r, c in zip(rows, cols)}


def generate_negatives(pos_pixels, dem_data, dem_nodata, n, buffer_pixels, rng):
    """Random sample n pixels from valid DEM, excluding buffer around positives."""
    valid_mask = dem_data != dem_nodata if dem_nodata is not None else np.ones_like(dem_data, dtype=bool)
    valid_mask &= np.isfinite(dem_data)

    pos_mask = np.zeros_like(valid_mask, dtype=bool)
    for r, c in pos_pixels:
        pos_mask[r, c] = True

    excl_mask = binary_dilation(pos_mask, iterations=buffer_pixels)
    sample_mask = valid_mask & ~excl_mask
    candidates = np.argwhere(sample_mask)
    n_take = min(n, len(candidates))
    if n_take < n:
        print(f"      WARN: only {len(candidates)} candidate pixels for {n} requested negatives")
    if n_take == 0:
        return set()
    idx = rng.choice(len(candidates), size=n_take, replace=False)
    return {(int(candidates[i, 0]), int(candidates[i, 1])) for i in idx}


def sample_basin(basin: str) -> None:
    factors_root, extras_slug = BASINS[basin]
    dem_path = factors_root / "dem_30m.tif"
    inv_path = INV_DIR / f"{basin}.gpkg"
    out_path = OUT_DIR / f"{basin}.h5"

    print(f"\n=== {basin} ===")
    if not dem_path.exists():
        print(f"  [skip] DEM missing: {dem_path}")
        return
    if not inv_path.exists():
        print(f"  [skip] Inventory missing: {inv_path}")
        return

    inv = gpd.read_file(inv_path)
    print(f"  Inventory: {len(inv)} records, labels={inv['label'].value_counts().to_dict()}")

    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs
        inv = inv.to_crs(dem_crs)
        rng = np.random.default_rng(SEED)
        cell_m = abs(dem.transform.a)

        # Positives
        pos_pts = inv[(inv["label"] == 1) & (inv["geom_type"] == "point")]
        pos_polys = inv[(inv["label"] == 1) & (inv["geom_type"] == "polygon")]
        pos_pixels = snap_points_to_pixels(pos_pts.geometry, dem)
        pos_pixels |= rasterize_polygon_pixels(pos_polys.geometry, dem,
                                              max_per_polygon=MAX_PIXELS_PER_POLYGON,
                                              rng=rng)
        print(f"  Positives: {len(pos_pts)} pts + {len(pos_polys)} polys "
              f"(cap {MAX_PIXELS_PER_POLYGON}/poly) → {len(pos_pixels)} unique pixels")

        # Negatives: hybrid strategy
        # 1) snap inventory neg points; 2) buffer-expand them; 3) random fill to NEG_RATIO * pos
        n_target = int(NEG_RATIO * len(pos_pixels))
        dem_data = dem.read(1)
        excl_buffer_px = max(1, int(NEG_BUFFER_M / cell_m))
        expand_buffer_px = max(0, int(NEG_EXPAND_M / cell_m))

        neg_pts = inv[(inv["label"] == 0)]
        if len(neg_pts) > 0:
            inv_neg = snap_points_to_pixels(neg_pts.geometry, dem)
            inv_neg -= pos_pixels
            expanded = expand_negatives_with_buffer(
                inv_neg, expand_buffer_px, dem_data, dem.nodata,
                exclusion_pixels=pos_pixels,
            )
            n_inv = len(expanded)
            if n_inv > n_target:
                expanded_arr = np.array(sorted(expanded))
                idx = rng.choice(len(expanded_arr), size=n_target, replace=False)
                neg_pixels = {(int(r), int(c)) for r, c in expanded_arr[idx]}
                action = f"subsample→{len(neg_pixels)}"
            else:
                need = n_target - n_inv
                random_neg = generate_negatives(pos_pixels, dem_data, dem.nodata,
                                               need, excl_buffer_px, rng)
                random_neg -= expanded
                neg_pixels = expanded | random_neg
                action = f"+random({need})→{len(neg_pixels)}"
            print(f"  Negatives: inv={len(inv_neg)} +buffer({expand_buffer_px}px)→{n_inv} "
                  f"{action} (target={n_target})")
        else:
            neg_pixels = generate_negatives(pos_pixels, dem_data, dem.nodata,
                                           n_target, excl_buffer_px, rng)
            print(f"  Negatives (generated): target={n_target}, buffer={excl_buffer_px}px "
                  f"→ {len(neg_pixels)} pixels")

        # Build samples table
        records = []
        for label, pixels in [(1, pos_pixels), (0, neg_pixels)]:
            for r, c in pixels:
                x, y = dem.xy(r, c)
                records.append({
                    "x_utm": x, "y_utm": y,
                    "pixel_row": r, "pixel_col": c,
                    "label": label,
                })
        df = pd.DataFrame(records)
        print(f"  Total samples: {len(df)} ({df['label'].sum()} pos + {(df['label']==0).sum()} neg)")

    # Feature extraction
    rasters = list(iter_feature_rasters(factors_root, extras_slug))
    print(f"  Extracting {len(rasters)} features...")
    coords = list(zip(df["x_utm"].tolist(), df["y_utm"].tolist()))
    for name, path in rasters:
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
            if src.nodata is not None:
                vals = np.where(vals == src.nodata, np.nan, vals)
            # Defense: float32 min/max are usually uninitialized memory or sentinel values
            extreme = (np.abs(vals) > 1e30) | ~np.isfinite(vals)
            vals = np.where(extreme, np.nan, vals)
        df[name] = vals

    write_h5(df, out_path)
    print(f"  → {out_path.name} ({len(df)} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    for basin in BASINS:
        sample_basin(basin)
    print("\n=== summary ===")
    for basin in BASINS:
        p = OUT_DIR / f"{basin}.h5"
        if p.exists():
            df = read_h5(p)
            n_pos = int((df["label"] == 1).sum())
            n_neg = int((df["label"] == 0).sum())
            n_feat = len(df.columns) - 5  # x_utm, y_utm, pixel_row, pixel_col, label
            print(f"  {basin:<10} pos={n_pos:<5} neg={n_neg:<5} features={n_feat}")
