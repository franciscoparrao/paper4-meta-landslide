#!/usr/bin/env python3
"""Paper 4 — align per-basin sample DataFrames to a common feature schema.

Input:  data/samples/{basin}.h5  (heterogeneous column names: composite__B02 vs imagery__s2_blue)
Output: data/aligned/{basin}.h5  (unified naming: spectral__blue, etc.)

Strategy:
- Rename per-basin spectral columns to wavelength-named keys.
- Keep only features common to all 8 basins (intersection).
- Drop Paper 1 red-edge bands (B05, B06, B07, B8A) that Paper 4 lacks.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
SAMPLES = ROOT / "data" / "samples"
ALIGNED = ROOT / "data" / "aligned"
ALIGNED.mkdir(parents=True, exist_ok=True)

BASINS = ["chanaral", "taltal", "maule", "choapa",
          "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes",
          "copiapo", "huasco", "elqui", "limari"]

# Rename map (per-basin spectral naming → unified). Keys include all variants we may encounter.
RENAME = {
    # Paper 1 (composite__BXX)
    "composite__B02": "spectral__blue",
    "composite__B03": "spectral__green",
    "composite__B04": "spectral__red",
    "composite__B08": "spectral__nir",
    "composite__B11": "spectral__swir16",
    "composite__B12": "spectral__swir22",
    # Paper 4 (imagery__s2_*)
    "imagery__s2_blue":   "spectral__blue",
    "imagery__s2_green":  "spectral__green",
    "imagery__s2_red":    "spectral__red",
    "imagery__s2_nir":    "spectral__nir",
    "imagery__s2_swir16": "spectral__swir16",
    "imagery__s2_swir22": "spectral__swir22",
}

# Features dropped (only in Paper 1 basins — red-edge bands)
DROP = {"composite__B05", "composite__B06", "composite__B07", "composite__B8A"}

META_COLS = ["label", "x_utm", "y_utm", "pixel_row", "pixel_col"]


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def write_h5(df: pd.DataFrame, path: Path) -> None:
    with h5py.File(path, "w") as f:
        for col in df.columns:
            arr = df[col].to_numpy()
            if arr.dtype.kind in "fiub":
                f.create_dataset(col, data=arr, compression="gzip", compression_opts=4)
            else:
                f.create_dataset(col, data=arr.astype("S"),
                                 compression="gzip", compression_opts=4)


# Step 1: read all + build column inventory after rename
all_dfs = {}
for b in BASINS:
    df = read_h5(SAMPLES / f"{b}.h5")
    df = df.drop(columns=[c for c in df.columns if c in DROP], errors="ignore")
    df = df.rename(columns=RENAME)
    all_dfs[b] = df
    print(f"  {b}: {len(df.columns)} cols, {len(df)} rows")

# Step 2: find intersection
common = set(all_dfs[BASINS[0]].columns)
for b in BASINS[1:]:
    common &= set(all_dfs[b].columns)
print(f"\nCommon columns across all 8 basins: {len(common)}")

# Order: meta first, then sorted features
features = sorted(c for c in common if c not in META_COLS)
ordered = [c for c in META_COLS if c in common] + features
print(f"Meta: {len([c for c in META_COLS if c in common])}, features: {len(features)}")

# Step 3: write aligned
print("\nWriting aligned per-basin H5 files:")
for b in BASINS:
    df = all_dfs[b][ordered].copy()
    df["basin"] = b
    out = ALIGNED / f"{b}.h5"
    write_h5(df, out)
    print(f"  {b}: {len(df)} rows × {len(df.columns)} cols → {out.name}")

# Save feature manifest
manifest_path = ALIGNED / "feature_manifest.txt"
manifest_path.write_text(
    "Aligned features (common across 8 basins)\n"
    f"Total features: {len(features)}\n\n"
    + "\n".join(features) + "\n"
)
print(f"\nFeature manifest: {manifest_path.name}")
