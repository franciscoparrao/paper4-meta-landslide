#!/usr/bin/env python3
"""Paper 4 — Correlation-based Feature Selection (Hall 1999) on aligned dataset.

CFS subset score:
    M_S = (k * r_cf_avg) / sqrt(k + k(k-1) * r_ff_avg)
    where r_cf_avg = mean correlation between selected features and class
          r_ff_avg = mean inter-feature correlation among selected
          k        = number of features in subset

Search: forward greedy. At each step, add the feature that maximizes M_S.
Stop when no addition improves the score.

Correlation: symmetric uncertainty (information-theoretic, robust to mixed types)
            U(X, Y) = 2 * I(X;Y) / (H(X) + H(Y))
            All features discretized into 10 bins for SU computation.

Input:  data/aligned/{basin}.h5  (8 basins)
Output: data/aligned/cfs_selected_features.txt + data/aligned/cfs_score_history.csv
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ALIGNED = ROOT / "data" / "aligned"
N_BINS = 10
SEED = 42

BASINS = ["chanaral", "taltal", "maule", "choapa",
          "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes",
          "copiapo", "huasco", "elqui", "limari"]
META = {"label", "x_utm", "y_utm", "pixel_row", "pixel_col", "basin"}


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def discretize(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1D array. NaNs → bin -1."""
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, -1, dtype=np.int32)
    if mask.sum() < 2:
        return out
    valid = arr[mask]
    if np.unique(valid).size <= n_bins:
        # Already few unique values — treat as ordinal
        _, inv = np.unique(valid, return_inverse=True)
        out[mask] = inv
        return out
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, n_bins + 1)))
    out[mask] = np.clip(np.digitize(valid, edges[1:-1]), 0, len(edges) - 2)
    return out


def entropy(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    pairs = x.astype(np.int64) * 10000 + y.astype(np.int64)
    return entropy(pairs)


def symmetric_uncertainty(x: np.ndarray, y: np.ndarray) -> float:
    hx = entropy(x); hy = entropy(y)
    if hx == 0 or hy == 0:
        return 0.0
    hxy = joint_entropy(x, y)
    mi = hx + hy - hxy
    return float(2.0 * mi / (hx + hy))


def cfs_subset_score(features: list[str], su_cf: dict, su_ff: dict) -> float:
    k = len(features)
    if k == 0:
        return -1
    rcf = np.mean([su_cf[f] for f in features])
    if k == 1:
        return rcf
    pairs = [su_ff[(features[i], features[j])]
             for i in range(k) for j in range(i + 1, k)]
    rff = np.mean(pairs) if pairs else 0.0
    return (k * rcf) / np.sqrt(k + k * (k - 1) * rff)


def main():
    # Load + concat all basins
    print("Loading aligned per-basin samples...")
    dfs = []
    for b in BASINS:
        df = read_h5(ALIGNED / f"{b}.h5")
        df["basin"] = b
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in full.columns if c not in META]
    print(f"  Total rows: {len(full)}, features: {len(feature_cols)}")
    print(f"  Class balance: {full['label'].value_counts().to_dict()}")

    # Discretize
    print("\nDiscretizing features (quantile bins)...")
    disc = {}
    for c in feature_cols:
        disc[c] = discretize(full[c].to_numpy(), N_BINS)
    label = full["label"].astype(np.int32).to_numpy()

    # SU(feature, class) for each feature
    print("Computing SU(feature, class)...")
    su_cf = {f: symmetric_uncertainty(disc[f], label) for f in feature_cols}
    cf_rank = sorted(su_cf.items(), key=lambda kv: -kv[1])
    print("  Top 10 by SU(f, class):")
    for f, v in cf_rank[:10]:
        print(f"    {f:<40} {v:.4f}")

    # Pairwise SU(feature, feature) — only computed lazily as needed below
    print("\nForward greedy CFS search...")
    su_ff = {}
    selected: list[str] = []
    remaining = set(feature_cols)
    history = []

    while remaining:
        best_f = None
        best_score = cfs_subset_score(selected, su_cf, su_ff) if selected else -1
        for f in remaining:
            cand = selected + [f]
            # Fill any missing pair SU
            for s in selected:
                key = (s, f) if s < f else (f, s)
                if key not in su_ff:
                    su_ff[key] = symmetric_uncertainty(disc[s], disc[f])
                    su_ff[(key[1], key[0])] = su_ff[key]
            score = cfs_subset_score(cand, su_cf, su_ff)
            if score > best_score:
                best_score = score
                best_f = f
        if best_f is None:
            break
        selected.append(best_f)
        remaining.remove(best_f)
        history.append({"step": len(selected), "added": best_f, "score": best_score,
                        "su_cf": su_cf[best_f]})
        print(f"  step {len(selected):2d}: +{best_f:<40} score={best_score:.4f}")

    # Save outputs
    out_features = ALIGNED / "cfs_selected_features.txt"
    out_features.write_text(f"# CFS-selected features ({len(selected)})\n"
                             f"# Final score: {history[-1]['score']:.4f}\n\n"
                             + "\n".join(selected) + "\n")
    print(f"\nSelected {len(selected)} features → {out_features.name}")

    out_history = ALIGNED / "cfs_score_history.csv"
    pd.DataFrame(history).to_csv(out_history, index=False)
    print(f"History → {out_history.name}")


if __name__ == "__main__":
    main()
