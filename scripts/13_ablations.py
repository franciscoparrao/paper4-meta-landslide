#!/usr/bin/env python3
"""Paper 4 — ablation analysis on meta-learning hyperparameters.

One-at-a-time variation around baseline (inner=5, eps=0.1, K_meta=10):
- inner_steps: {1, 3, 5, 10}
- eps (Reptile only): {0.05, 0.1, 0.3}
- K_meta (training-time K-shot): {5, 10, 20}

For each config: train meta-model, evaluate K=5 K-shot on 4 targets × 3 seeds × 30 episodes.

Outputs:
- results/ablations/ablations_raw.csv
- results/ablations/ablations_summary.csv
- figures/ablation_inner_steps.png, ablation_eps.png, ablation_K_meta.png
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _meta_lib import (
    DEVICE, MLP, bootstrap_ci, evaluate_kshot_episode, fomaml_train,
    load_basin, load_features, reptile_train, standardize_basin,
)

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
OUT = ROOT / "results" / "ablations"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCE = ["chanaral", "taltal", "maule", "choapa", "tilviche", "caracoles", "maipo", "rapel", "bueno_puelo", "magallanes"]
TARGET = ["copiapo", "huasco", "elqui", "limari"]
SEEDS = [42, 123, 7]
N_EPISODES = 30
N_QUERY = 10
EVAL_K = 5  # representative K for ablation eval

# Baseline hyperparameters
BASELINE = {"inner_steps": 5, "eps": 0.1, "K_meta": 10}
META_OUTER = 300
META_INNER_LR = 1e-2
META_OUTER_LR = 1e-3

# Sweep grids
SWEEPS = {
    "inner_steps": [1, 3, 5, 10],
    "eps":         [0.05, 0.1, 0.3],
    "K_meta":      [5, 10, 20],
}


def run_config(method, hp, src_std, tgt_data, n_in, seed):
    """Train meta-model with given hyperparameters; return per-target eval rows."""
    rng = np.random.default_rng(seed)
    if method == "reptile":
        meta = reptile_train(
            src_std, n_in, META_OUTER, hp["K_meta"], hp["inner_steps"],
            META_INNER_LR, hp["eps"], rng,
        )
    elif method == "fomaml":
        meta = fomaml_train(
            src_std, n_in, META_OUTER, hp["K_meta"], N_QUERY, hp["inner_steps"],
            META_INNER_LR, META_OUTER_LR, rng,
        )
    else:
        raise ValueError(method)
    state = deepcopy(meta.state_dict())
    rows = []
    for tgt, (X, y) in tgt_data.items():
        X_std = standardize_basin(X)
        rng_ep = np.random.default_rng(seed * 1000 + hash(tgt + method) % 1000)
        for ep in range(N_EPISODES):
            m = evaluate_kshot_episode(state, X_std, y, EVAL_K, N_QUERY,
                                       n_in, rng_ep)
            rows.append({"method": method, "target": tgt, "seed": seed,
                         "episode": ep, "f1": m["f1"], "auc": m["auc"], **hp})
    return rows


def main():
    features = load_features()
    n_in = len(features)
    src_raw = {b: load_basin(b, features) for b in SOURCE}
    src_std = {b: (standardize_basin(X), y) for b, (X, y) in src_raw.items()}
    tgt_raw = {b: load_basin(b, features) for b in TARGET}

    rows = []
    for sweep_name, values in SWEEPS.items():
        for v in values:
            for method in ["reptile", "fomaml"]:
                if sweep_name == "eps" and method == "fomaml":
                    continue  # eps is Reptile-only
                hp = dict(BASELINE)
                hp[sweep_name] = v
                hp["sweep"] = sweep_name
                for seed in SEEDS:
                    torch.manual_seed(seed); np.random.seed(seed)
                    print(f"  {method:<8} sweep={sweep_name:<11} {sweep_name}={v} seed={seed}")
                    rows.extend(run_config(method, hp, src_std, tgt_raw, n_in, seed))

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "ablations_raw.csv", index=False)
    print(f"\nRaw → {OUT / 'ablations_raw.csv'} ({len(raw)} rows)")

    summary_rows = []
    for (sweep, method, val), grp in raw.groupby(["sweep", "method", "inner_steps"]):
        # Ad-hoc grouping: use the swept variable column as the value
        pass
    # Cleaner aggregation: group by sweep, value of swept variable, method
    out_rows = []
    for sweep, sub in raw.groupby("sweep"):
        for method, sub2 in sub.groupby("method"):
            for v, sub3 in sub2.groupby(sweep):
                f1_m, f1_lo, f1_hi = bootstrap_ci(sub3["f1"].values)
                auc_m, auc_lo, auc_hi = bootstrap_ci(sub3["auc"].values)
                out_rows.append({
                    "sweep": sweep, "value": v, "method": method, "n_runs": len(sub3),
                    "f1_mean": f1_m, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
                    "auc_mean": auc_m, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
                })
    summary = pd.DataFrame(out_rows)
    summary.to_csv(OUT / "ablations_summary.csv", index=False)
    print(f"Summary → {OUT / 'ablations_summary.csv'}")

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
    colors = {"reptile": "#d95f0e", "fomaml": "#c51b8a"}
    for sweep in SWEEPS:
        sub = summary[summary.sweep == sweep]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for method in sub["method"].unique():
            d = sub[sub.method == method].sort_values("value")
            ax.errorbar(d["value"], d["f1_mean"],
                        yerr=[d["f1_mean"] - d["f1_ci_lo"], d["f1_ci_hi"] - d["f1_mean"]],
                        fmt="o-", color=colors[method], label=method.capitalize(),
                        linewidth=2, markersize=7, capsize=4)
        baseline_v = BASELINE[sweep]
        ax.axvline(baseline_v, linestyle="--", color="gray", alpha=0.5, label="baseline")
        ax.set_xlabel(sweep)
        ax.set_ylabel(f"F1 (K={EVAL_K} K-shot, mean ± 95% CI)")
        ax.set_title(f"Ablation: {sweep}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        out = FIGS / f"ablation_{sweep}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out.name}")


if __name__ == "__main__":
    main()
