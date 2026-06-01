#!/usr/bin/env python3
"""Paper 4 — build per-basin ML-ready datasets with CFS-selected features.

Input:
  data/aligned/{basin}.h5
  data/aligned/cfs_selected_features.txt
Output:
  data/ml_ready/{basin}.h5  (X: features, y: label, plus coordinates)
  data/ml_ready/dataset_summary.csv
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ALIGNED = ROOT / "data" / "aligned"
ML_READY = ROOT / "data" / "ml_ready"
ML_READY.mkdir(parents=True, exist_ok=True)

BASINS = ["chanaral", "taltal", "maule", "choapa",
          "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes",
          "copiapo", "huasco", "elqui", "limari"]
SOURCE_BASINS = {"chanaral", "taltal", "maule", "choapa",
                 "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"}
TARGET_BASINS = {"copiapo", "huasco", "elqui", "limari"}


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


# Read CFS selection
sel_path = ALIGNED / "cfs_selected_features.txt"
sel = [ln.strip() for ln in sel_path.read_text().splitlines()
       if ln.strip() and not ln.startswith("#")]
print(f"CFS features ({len(sel)}):")
for f in sel:
    print(f"  - {f}")

KEEP_COLS = ["label", "x_utm", "y_utm", "pixel_row", "pixel_col"] + sel

print("\nBuilding ML-ready per-basin datasets:")
summary = []
for b in BASINS:
    df = read_h5(ALIGNED / f"{b}.h5")
    df = df[KEEP_COLS].copy()

    # Impute NaN per feature with column median (per basin to preserve regional signal)
    for c in sel:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    role = "source" if b in SOURCE_BASINS else "target"
    df["role"] = role
    df["basin"] = b

    out = ML_READY / f"{b}.h5"
    write_h5(df, out)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    summary.append({
        "basin": b, "role": role, "n_total": len(df),
        "n_pos": n_pos, "n_neg": n_neg,
        "n_features": len(sel),
        "any_nan_after_impute": bool(df[sel].isna().any().any()),
    })
    print(f"  {b:<10} ({role}) {len(df):>5} samples | pos={n_pos} neg={n_neg}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(ML_READY / "dataset_summary.csv", index=False)
print(f"\nSummary written → {ML_READY / 'dataset_summary.csv'}")
print(summary_df.to_string(index=False))

# Sanity: also write a single combined file for cross-basin training
print("\nWriting combined dataset (all basins concatenated)...")
combined = pd.concat([read_h5(ML_READY / f"{b}.h5") for b in BASINS], ignore_index=True)
write_h5(combined, ML_READY / "all_basins.h5")
print(f"  {len(combined)} rows × {len(combined.columns)} cols → all_basins.h5")
