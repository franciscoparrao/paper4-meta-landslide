"""Shared utilities for meta-learning scripts (12, 13, 14, 15).

Contains MLP, sample_kshot, fit_inner, reptile_train, fomaml_train, evaluate_episode,
bootstrap_ci, and load helpers — to avoid duplication across the analysis scripts.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper4_meta_learning_transfer")
ML = ROOT / "data" / "ml_ready"
DEVICE = torch.device("cpu")


def read_h5(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def load_basin(basin: str, features: list[str]):
    df = read_h5(ML / f"{basin}.h5")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y


def load_features() -> list[str]:
    p = ML / "../aligned/cfs_selected_features.txt"
    return [l.strip() for l in p.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def standardize_basin(X):
    mu = X.mean(0); sd = np.maximum(X.std(0), 1e-6)
    return (X - mu) / sd


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_inner(model, X, y, lr, steps):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    for _ in range(steps):
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt), yt)
        loss.backward()
        optim.step()
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).cpu().numpy()


def sample_kshot(X, y, K, n_query, rng):
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    Kp = min(K, len(pos_idx)); Kn = min(K, len(neg_idx))
    sel_pos = rng.choice(pos_idx, size=Kp, replace=False)
    sel_neg = rng.choice(neg_idx, size=Kn, replace=False)
    sup = np.concatenate([sel_pos, sel_neg])
    rest_pos = np.setdiff1d(pos_idx, sel_pos); rest_neg = np.setdiff1d(neg_idx, sel_neg)
    nq = min(n_query, len(rest_pos), len(rest_neg))
    qpos = rng.choice(rest_pos, size=nq, replace=False)
    qneg = rng.choice(rest_neg, size=nq, replace=False)
    qry = np.concatenate([qpos, qneg])
    return sup, qry


def evaluate_query(model, Xq, yq):
    p = predict_proba(model, Xq)
    pred = (p > 0.5).astype(int)
    return {
        "f1":  f1_score(yq, pred, zero_division=0),
        "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan"),
    }


def reptile_train(src_data, n_in, outer_steps, K, inner_steps, inner_lr, eps, rng):
    meta = MLP(n_in).to(DEVICE)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        sup, _ = sample_kshot(X, y, K, n_query=10, rng=rng)
        inner = deepcopy(meta)
        inner = fit_inner(inner, X[sup], y[sup], lr=inner_lr, steps=inner_steps)
        with torch.no_grad():
            for pm, pi in zip(meta.parameters(), inner.parameters()):
                pm.add_(eps * (pi - pm))
    return meta


def fomaml_train(src_data, n_in, outer_steps, K, n_query, inner_steps,
                 inner_lr, outer_lr, rng):
    meta = MLP(n_in).to(DEVICE)
    meta_optim = torch.optim.Adam(meta.parameters(), lr=outer_lr)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        sup, qry = sample_kshot(X, y, K, n_query, rng)
        inner = deepcopy(meta)
        inner_optim = torch.optim.SGD(inner.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            inner_optim.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                inner(torch.from_numpy(X[sup])),
                torch.from_numpy(y[sup]),
            )
            loss.backward()
            inner_optim.step()
        for p in inner.parameters():
            p.grad = None
        loss_q = F.binary_cross_entropy_with_logits(
            inner(torch.from_numpy(X[qry])),
            torch.from_numpy(y[qry]),
        )
        loss_q.backward()
        meta_optim.zero_grad()
        for pm, pi in zip(meta.parameters(), inner.parameters()):
            if pi.grad is not None:
                pm.grad = pi.grad.detach().clone()
        meta_optim.step()
    return meta


def pretrain_concat(src_data, n_in, steps, lr):
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    Xs_std = standardize_basin(Xs).astype(np.float32)
    model = MLP(n_in).to(DEVICE)
    return fit_inner(model, Xs_std, ys, lr=lr, steps=steps)


def evaluate_kshot_episode(init_state, X_norm, y, K, n_query, n_in, rng,
                            adapt_lr=1e-2, adapt_steps=30):
    sup, qry = sample_kshot(X_norm, y, K, n_query, rng)
    Xs, ys = X_norm[sup], y[sup]
    Xq, yq = X_norm[qry], y[qry]
    model = MLP(n_in).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = fit_inner(model, Xs, ys, lr=adapt_lr, steps=adapt_steps)
    return evaluate_query(model, Xq, yq)


def bootstrap_ci(values, n_boot=1000, ci=0.95, rng=None):
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


# ---- DANN (Ganin & Lempitsky 2015) ----

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class DANN(nn.Module):
    def __init__(self, n_in, n_domains, hidden=(64, 32)):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
        )
        self.class_head = nn.Linear(hidden[1], 1)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden[1], 32), nn.ReLU(),
            nn.Linear(32, n_domains),
        )

    def forward(self, x, lambd=1.0):
        z = self.feat(x)
        return self.class_head(z).squeeze(-1), self.domain_head(grad_reverse(z, lambd))


def dann_to_mlp_state(dann: "DANN") -> dict:
    """Pack DANN feat + class_head into the MLP state-dict for adapt-time use."""
    mlp = MLP(dann.feat[0].in_features)
    mlp.net[0].load_state_dict(dann.feat[0].state_dict())
    mlp.net[3].load_state_dict(dann.feat[3].state_dict())
    mlp.net[6].load_state_dict(dann.class_head.state_dict())
    return deepcopy(mlp.state_dict())


def train_dann(src_data, n_in, epochs, batch, lr, rng):
    """Standard DANN training with annealed lambda."""
    domain_to_id = {b: i for i, b in enumerate(src_data)}
    Xs_list, ys_list, ds_list = [], [], []
    for b, (X, y) in src_data.items():
        Xs_list.append(X); ys_list.append(y)
        ds_list.append(np.full(len(y), domain_to_id[b], dtype=np.int64))
    X_all = np.concatenate(Xs_list, axis=0)
    y_all = np.concatenate(ys_list, axis=0)
    d_all = np.concatenate(ds_list, axis=0)
    mu = X_all.mean(0); sd = np.maximum(X_all.std(0), 1e-6)
    X_all = ((X_all - mu) / sd).astype(np.float32)

    model = DANN(n_in, n_domains=len(domain_to_id)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X_all)
    yt = torch.from_numpy(y_all.astype(np.float32))
    dt = torch.from_numpy(d_all)
    n = len(X_all)
    total_iters = epochs * (n // batch)
    iter_count = 0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            bi = idx[start:start + batch]
            x = Xt[bi]; y_ = yt[bi]; d_ = dt[bi]
            p = iter_count / max(1, total_iters)
            lambd = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            optim.zero_grad()
            cl, dl = model(x, lambd)
            loss = (F.binary_cross_entropy_with_logits(cl, y_)
                    + F.cross_entropy(dl, d_))
            loss.backward()
            optim.step()
            iter_count += 1
    return model


# ---- ProtoNet (Snell et al. 2017) — adapted for tabular binary classification ----

class ProtoEncoder(nn.Module):
    """Feature encoder shared with MLP backbone for fair comparison."""
    def __init__(self, n_in, hidden=(64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


def _prototypes(z, y):
    """Compute class prototypes given embeddings z and labels y."""
    p_pos = z[y == 1].mean(0) if (y == 1).any() else z.mean(0)
    p_neg = z[y == 0].mean(0) if (y == 0).any() else z.mean(0)
    return torch.stack([p_neg, p_pos])  # [2, d]


def _proto_predict_logits(z_query, prototypes):
    """Negative squared Euclidean distance to each prototype = logits."""
    d = ((z_query.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(-1)
    return -d  # higher is closer


def protonet_train(src_data, n_in, outer_steps, K, n_query, lr, rng):
    """Episodic training of a feature encoder via prototypical loss."""
    enc = ProtoEncoder(n_in).to(DEVICE)
    optim = torch.optim.Adam(enc.parameters(), lr=lr)
    for _ in range(outer_steps):
        basin = rng.choice(list(src_data.keys()))
        X, y = src_data[basin]
        sup, qry = sample_kshot(X, y, K, n_query, rng)
        Xs, ys = torch.from_numpy(X[sup]), torch.from_numpy(y[sup])
        Xq, yq = torch.from_numpy(X[qry]), torch.from_numpy(y[qry])
        zs = enc(Xs); zq = enc(Xq)
        protos = _prototypes(zs, ys)
        logits = _proto_predict_logits(zq, protos)
        loss = F.cross_entropy(logits, yq.long())
        optim.zero_grad(); loss.backward(); optim.step()
    return enc


def evaluate_protonet_episode(encoder, X_norm, y, K, n_query, n_in, rng,
                              support_idx=None, query_idx=None):
    """K-shot evaluation: extract features, compute prototypes from K support, classify query."""
    if support_idx is None:
        sup, qry = sample_kshot(X_norm, y, K, n_query, rng)
    else:
        sup, qry = support_idx, query_idx
    encoder.eval()
    with torch.no_grad():
        zs = encoder(torch.from_numpy(X_norm[sup]))
        zq = encoder(torch.from_numpy(X_norm[qry]))
        protos = _prototypes(zs, torch.from_numpy(y[sup]))
        logits = _proto_predict_logits(zq, protos)
        p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    pred = (p > 0.5).astype(int)
    yq = y[qry]
    return {"f1": f1_score(yq, pred, zero_division=0),
            "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan")}


# ---- Meta-Baseline (Chen et al. 2021) ----
# Step 1: pretrain encoder with standard classification on concat source.
# Step 2: at test time, compute cosine prototypes and classify by similarity.

def meta_baseline_train(src_data, n_in, steps, lr):
    """Pretrain encoder + classifier head with concat-source supervised learning."""
    Xs = np.concatenate([X for X, _ in src_data.values()], axis=0)
    ys = np.concatenate([y for _, y in src_data.values()], axis=0)
    Xs = standardize_basin(Xs).astype(np.float32)
    enc = ProtoEncoder(n_in).to(DEVICE)
    head = nn.Linear(32, 1).to(DEVICE)
    optim = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    Xt = torch.from_numpy(Xs); yt = torch.from_numpy(ys)
    for _ in range(steps):
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(enc(Xt)).squeeze(-1), yt)
        loss.backward(); optim.step()
    return enc  # discard head — use cosine prototypes at test time


def evaluate_meta_baseline_episode(encoder, X_norm, y, K, n_query, n_in, rng,
                                   support_idx=None, query_idx=None,
                                   temperature=10.0):
    """Cosine-similarity prototype classifier on encoder features."""
    if support_idx is None:
        sup, qry = sample_kshot(X_norm, y, K, n_query, rng)
    else:
        sup, qry = support_idx, query_idx
    encoder.eval()
    with torch.no_grad():
        zs = encoder(torch.from_numpy(X_norm[sup]))
        zq = encoder(torch.from_numpy(X_norm[qry]))
        ys = torch.from_numpy(y[sup])
        protos = _prototypes(zs, ys)  # [2, d]
        # cosine similarity
        zq_n = F.normalize(zq, dim=-1)
        protos_n = F.normalize(protos, dim=-1)
        logits = temperature * (zq_n @ protos_n.t())
        p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    pred = (p > 0.5).astype(int)
    yq = y[qry]
    return {"f1": f1_score(yq, pred, zero_division=0),
            "auc": roc_auc_score(yq, p) if len(np.unique(yq)) == 2 else float("nan")}


# ---- CDAN (Long et al. 2018) — conditional adversarial DA ----

class CDAN(nn.Module):
    """Domain classifier conditions on features × class predictions outer product."""
    def __init__(self, n_in, n_domains, hidden=(64, 32)):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
        )
        self.class_head = nn.Linear(hidden[1], 1)
        # Domain classifier sees outer product flattened: hidden[1] * 2 dims (binary class)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden[1] * 2, 64), nn.ReLU(),
            nn.Linear(64, n_domains),
        )

    def forward(self, x, lambd=1.0):
        z = self.feat(x)
        cl = self.class_head(z).squeeze(-1)
        # class probability vector [B, 2]
        p_pos = torch.sigmoid(cl)
        p_vec = torch.stack([1 - p_pos, p_pos], dim=-1)  # [B, 2]
        # outer product features × class probs → [B, hidden, 2]
        cond = torch.einsum("bd,bc->bdc", z, p_vec).flatten(1)  # [B, hidden*2]
        dl = self.domain_head(grad_reverse(cond, lambd))
        return cl, dl


def train_cdan(src_data, n_in, epochs, batch, lr, rng):
    """CDAN training: conditional adversarial DA."""
    domain_to_id = {b: i for i, b in enumerate(src_data)}
    Xs_list, ys_list, ds_list = [], [], []
    for b, (X, y) in src_data.items():
        Xs_list.append(X); ys_list.append(y)
        ds_list.append(np.full(len(y), domain_to_id[b], dtype=np.int64))
    X_all = np.concatenate(Xs_list, axis=0)
    y_all = np.concatenate(ys_list, axis=0)
    d_all = np.concatenate(ds_list, axis=0)
    mu = X_all.mean(0); sd = np.maximum(X_all.std(0), 1e-6)
    X_all = ((X_all - mu) / sd).astype(np.float32)

    model = CDAN(n_in, n_domains=len(domain_to_id)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X_all); yt = torch.from_numpy(y_all.astype(np.float32))
    dt = torch.from_numpy(d_all)
    n = len(X_all)
    total_iters = epochs * (n // batch)
    iter_count = 0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            bi = idx[start:start + batch]
            x = Xt[bi]; y_ = yt[bi]; d_ = dt[bi]
            p = iter_count / max(1, total_iters)
            lambd = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            optim.zero_grad()
            cl, dl = model(x, lambd)
            loss = (F.binary_cross_entropy_with_logits(cl, y_)
                    + F.cross_entropy(dl, d_))
            loss.backward(); optim.step()
            iter_count += 1
    return model


def cdan_to_mlp_state(cdan: "CDAN") -> dict:
    """Pack CDAN feat + class_head into MLP state for K-shot adapt."""
    mlp = MLP(cdan.feat[0].in_features)
    mlp.net[0].load_state_dict(cdan.feat[0].state_dict())
    mlp.net[3].load_state_dict(cdan.feat[3].state_dict())
    mlp.net[6].load_state_dict(cdan.class_head.state_dict())
    return deepcopy(mlp.state_dict())
