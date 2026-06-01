"""Spatial CNN backbone + meta-learning utilities for the U-Net/CNN baseline.

Mirrors the API of _meta_lib for the point-wise MLP, but operates on
multi-channel image patches [B, C, P, P]. Used by 26_unet_benchmark.py.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
PATCH_DIR = ROOT / "data" / "patches"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpatialCNN(nn.Module):
    """Small patch-classifier CNN. Input [B,C,32,32] -> binary logit.

    Architecture: 3 conv blocks (Conv-ReLU-Pool) + AdaptiveAvgPool + FC head.
    ~70k params. Treats the center pixel's class via global patch context.
    """
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_in, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                          # 32 -> 16
            nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                          # 16 -> 8
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                  # -> 1x1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def load_patches(basin: str):
    d = np.load(PATCH_DIR / f"{basin}.npz")
    return d["X"], d["y"].astype(np.float32)


def sample_kshot(y, K, n_query, rng):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    Kp = min(K, len(pos)); Kn = min(K, len(neg))
    sup_p = rng.choice(pos, Kp, replace=False)
    sup_n = rng.choice(neg, Kn, replace=False)
    sup = np.concatenate([sup_p, sup_n])
    rp = np.setdiff1d(pos, sup_p); rn = np.setdiff1d(neg, sup_n)
    nq = min(n_query, len(rp), len(rn))
    qry = np.concatenate([rng.choice(rp, nq, replace=False),
                          rng.choice(rn, nq, replace=False)])
    return sup, qry


def fit_inner_cnn(model, X, y, lr, steps, batch=64):
    """Adam minibatch training of the CNN backbone for `steps` updates."""
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(y).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    for _ in range(steps):
        idx = torch.randperm(n)[:batch] if n > batch else torch.arange(n)
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt[idx]), yt[idx])
        loss.backward(); optim.step()
    return model


def evaluate_query_cnn(model, X, y):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X).to(DEVICE))).cpu().numpy()
    pred = (p > 0.5).astype(int)
    return {"f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")}


def reptile_train_cnn(src_data, n_in, outer_steps, K, inner_steps, inner_lr,
                      eps, rng):
    """Reptile meta-training over CNN. src_data: {basin: (X[N,C,P,P], y[N])}."""
    meta = SpatialCNN(n_in).to(DEVICE)
    for _ in range(outer_steps):
        b = rng.choice(list(src_data.keys()))
        X, y = src_data[b]
        sup, _ = sample_kshot(y, K, n_query=10, rng=rng)
        inner = deepcopy(meta)
        inner = fit_inner_cnn(inner, X[sup], y[sup], inner_lr, inner_steps)
        with torch.no_grad():
            for pm, pi in zip(meta.parameters(), inner.parameters()):
                pm.add_(eps * (pi - pm))
    return meta


def fomaml_train_cnn(src_data, n_in, outer_steps, K, n_query, inner_steps,
                    inner_lr, outer_lr, rng):
    """FOMAML meta-training over CNN."""
    meta = SpatialCNN(n_in).to(DEVICE)
    meta_optim = torch.optim.Adam(meta.parameters(), lr=outer_lr)
    for _ in range(outer_steps):
        b = rng.choice(list(src_data.keys()))
        X, y = src_data[b]
        sup, qry = sample_kshot(y, K, n_query, rng)
        inner = deepcopy(meta)
        inner_optim = torch.optim.SGD(inner.parameters(), lr=inner_lr)
        Xs = torch.from_numpy(X[sup]).to(DEVICE)
        ys = torch.from_numpy(y[sup]).to(DEVICE)
        for _ in range(inner_steps):
            inner_optim.zero_grad()
            F.binary_cross_entropy_with_logits(inner(Xs), ys).backward()
            inner_optim.step()
        for p in inner.parameters():
            p.grad = None
        Xq = torch.from_numpy(X[qry]).to(DEVICE)
        yq = torch.from_numpy(y[qry]).to(DEVICE)
        F.binary_cross_entropy_with_logits(inner(Xq), yq).backward()
        meta_optim.zero_grad()
        for pm, pi in zip(meta.parameters(), inner.parameters()):
            if pi.grad is not None:
                pm.grad = pi.grad.detach().clone()
        meta_optim.step()
    return meta


def pretrain_concat_cnn(src_data, n_in, steps, lr, batch=128):
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    model = SpatialCNN(n_in).to(DEVICE)
    return fit_inner_cnn(model, Xs, ys, lr, steps, batch=batch)
