#!/usr/bin/env python3
"""Paper 4 (R1) — source-pool ablations (reviewer 3, comments 11 and 12).

Three source-pool conditions, evaluated with the spatial-CV protocol of
16_spatial_benchmark.py (5 seeds, K-means 5 folds, 20 episodes, query = full
held-out cluster, deterministic episode seeds) on the four unchanged targets:

    baseline   : the ten source basins as in the paper
    no_maule   : Maule 2010 (the only purely earthquake-triggered inventory) removed
                 from the source pool (c.11, mixed triggers)
    homog_neg  : negatives of Chañaral and Taltal (the only two basins with mapped
                 non-landslide points) re-generated with the rule used in the other
                 twelve basins: random pixels of the DEM valid area outside a 500 m
                 buffer around positives, 1:1 (c.12, rule-induced domain signal)

Methods: Independent (reference), Reptile, Fine-tune (basin-balanced). K in {1, 5}.
Outputs: results/source_ablations/{raw_runs,summary,paired}.csv
"""
from __future__ import annotations

import importlib.util
import sys
import zlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import binary_dilation
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta_lib import (  # noqa: E402
    load_basin, load_features, read_h5, reptile_train, standardize_basin,
)

_spec = importlib.util.spec_from_file_location("pve", Path(__file__).parent / "34_pooled_vs_episodic.py")
_pve = importlib.util.module_from_spec(_spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]; _spec.loader.exec_module(_pve); sys.argv = _argv
pretrain_balanced, spatial_kshot_episode = _pve.pretrain_balanced, _pve.spatial_kshot_episode

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "ml_ready"
OUT = ROOT / "results" / "source_ablations"
OUT.mkdir(parents=True, exist_ok=True)
PAPER1 = Path("/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad/factors")
RASTERS = {"chanaral": (PAPER1 / "04_rio_salado", ROOT / "factors_extra" / "04_rio_salado"),
           "taltal": (ROOT / "factors" / "taltal", ROOT / "factors_extra" / "taltal")}

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
KS = [1, 5]
SEEDS = [42, 123, 7, 1729, 2718]
K_FOLDS, N_EP = 5, 20
META_OUTER, META_K, META_INNER, META_LR, EPS = 300, 10, 5, 1e-2, 0.1
PRETRAIN_STEPS, PRETRAIN_LR = 300, 1e-3
NEG_BUFFER_M, RES_M = 500, 30
METHODS = ["independent", "reptile", "finetune_balanced"]
CONDITIONS = ["baseline", "no_maule", "homog_neg"]


def raster(basin, feat):
    cat, name = feat.split("__", 1)
    for root in RASTERS[basin]:
        p = root / cat / f"{name}.tif"
        if p.exists():
            return p
    raise FileNotFoundError(f"{basin}:{feat}")


def homogeneous_negatives(basin, features, rng):
    """Positives as in the paper; negatives re-drawn at random outside a 500 m buffer."""
    df = read_h5(ML / f"{basin}.h5")
    pos = df[df.label == 1][["pixel_row", "pixel_col"]].to_numpy(int)
    stack = []
    for f in features:
        with rasterio.open(raster(basin, f)) as src:
            b = src.read(1).astype(np.float32)
            if src.nodata is not None:
                b[b == src.nodata] = np.nan
            stack.append(b)
    stack = np.stack(stack)
    valid = np.isfinite(stack).all(0)
    pos_mask = np.zeros(valid.shape, bool); pos_mask[pos[:, 0], pos[:, 1]] = True
    excl = binary_dilation(pos_mask, iterations=round(NEG_BUFFER_M / RES_M))
    cand = np.argwhere(valid & ~excl)
    neg = cand[rng.choice(len(cand), size=len(pos), replace=False)]
    rc = np.vstack([pos, neg])
    X = stack[:, rc[:, 0], rc[:, 1]].T
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.float32)
    keep = np.isfinite(X).all(1)
    # impute the few NaN positives with the basin median, as in 08_build_ml_datasets
    if not keep.all():
        med = np.nanmedian(X, axis=0); X = np.where(np.isfinite(X), X, med)
    return X.astype(np.float32), y, int(len(pos)), int(len(cand))


def episode_seed(seed, fold, K, tgt):
    return seed * 1000 + fold * 100 + K + zlib.crc32(tgt.encode()) % 100


def paired(raw, cond, t, K, a, b="independent"):
    s = raw[(raw.condition == cond) & (raw.target == t) & (raw.K == K)]
    ma = s[s.method == a].groupby(["seed", "fold"]).f1.mean()
    mb = s[s.method == b].groupby(["seed", "fold"]).f1.mean()
    c = ma.index.intersection(mb.index)
    return (ma.loc[c] - mb.loc[c]).to_numpy()


def main():
    feats = load_features(); n_in = len(feats)
    src_base = {b: load_basin(b, feats) for b in SOURCE}
    rng0 = np.random.default_rng(42)
    src_homog = dict(src_base)
    for b in ["chanaral", "taltal"]:
        X, y, npos, ncand = homogeneous_negatives(b, feats, rng0)
        src_homog[b] = (X, y)
        print(f"{b}: {npos} positives kept, {npos} random negatives from {ncand:,} candidate pixels", flush=True)
    pools = {"baseline": src_base,
             "no_maule": {b: v for b, v in src_base.items() if b != "maule"},
             "homog_neg": src_homog}
    tgt = {}
    for b in TARGET:
        df = read_h5(ML / f"{b}.h5")
        tgt[b] = (df[feats].to_numpy(np.float32), df.label.to_numpy(np.float32), df[["x_utm", "y_utm"]].to_numpy(float))

    rows = []
    for cond in CONDITIONS:
        pool = pools[cond]
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed)
            src_std = {b: (standardize_basin(X), y) for b, (X, y) in pool.items()}
            states = {"independent": None,
                      "reptile": deepcopy(reptile_train(src_std, n_in, META_OUTER, META_K, META_INNER, META_LR, EPS, rng).state_dict()),
                      "finetune_balanced": deepcopy(pretrain_balanced(pool, n_in, PRETRAIN_STEPS, PRETRAIN_LR).state_dict())}
            for t, (X, y, coords) in tgt.items():
                Xs = standardize_basin(X)
                folds = KMeans(n_clusters=K_FOLDS, random_state=seed, n_init=10).fit_predict(coords)
                for fold in range(K_FOLDS):
                    test = folds == fold; train = ~test
                    for K in KS:
                        for m in METHODS:
                            rng_ep = np.random.default_rng(episode_seed(seed, fold, K, t))
                            for ep in range(N_EP):
                                r = spatial_kshot_episode(states[m], Xs, y, train, test, K, n_in, rng_ep)
                                if r is None:
                                    continue
                                rows.append({"condition": cond, "target": t, "K": K, "method": m, "seed": seed,
                                             "fold": fold, "episode": ep, "f1": r["f1"], "auc": r["auc"]})
            print(f"{cond} seed {seed} done ({len(rows)} rows)", flush=True)

    raw = pd.DataFrame(rows); raw.to_csv(OUT / "raw_runs.csv", index=False)
    summ = raw.groupby(["condition", "target", "K", "method"]).f1.mean().reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)

    rng = np.random.default_rng(42)
    prow = []
    for cond in CONDITIONS:
        for K in KS:
            for m in ["reptile", "finetune_balanced"]:
                diffs = {t: paired(raw, cond, t, K, m) for t in TARGET}
                for t, d in diffs.items():
                    boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
                    try:
                        wp = float(wilcoxon(d).pvalue)
                    except ValueError:
                        wp = np.nan
                    prow.append({"condition": cond, "K": K, "method": m, "target": t, "n_clusters": len(d),
                                 "diff_pp": d.mean() * 100, "ci_lo_pp": np.percentile(boot, 2.5) * 100,
                                 "ci_hi_pp": np.percentile(boot, 97.5) * 100, "wilcoxon_p": wp})
                arrs = list(diffs.values()); obs = np.mean([a.mean() for a in arrs])
                st = np.array([np.mean([(a * rng.choice([-1.0, 1.0], len(a))).mean() for a in arrs]) for _ in range(20000)])
                prow.append({"condition": cond, "K": K, "method": m, "target": "JOINT",
                             "n_clusters": sum(len(a) for a in arrs), "diff_pp": obs * 100,
                             "ci_lo_pp": np.nan, "ci_hi_pp": np.nan,
                             "wilcoxon_p": float((np.sum(np.abs(st) >= abs(obs)) + 1) / 20001)})
    pr = pd.DataFrame(prow); pr.to_csv(OUT / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMean F1 at K=1:"); print(summ[summ.K == 1].pivot_table(index=["condition", "target"], columns="method", values="f1").round(3))
    print("\nPaired difference vs Independent (pp):")
    print(pr.pivot_table(index=["condition", "target"], columns=["K", "method"], values="diff_pp").round(2))
    print("\nJoint p:"); print(pr[pr.target == "JOINT"][["condition", "K", "method", "diff_pp", "wilcoxon_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
