"""Probing + scoring (paper Appendix B).

Score curve S_{t,.} for a target t is produced per (split design, probe family):
one probe is fit per layer on a train-only-preprocessed subsample, and scored on
the held-out group split. Categorical: S = max(0,(acc-b)/(1-b)) with b the
held-out majority baseline (0 if b==1). Continuous (tree depth): S = max(0, rho)
with Spearman rho set to 0 under constant ranks.

Depth functionals are applied to these curves downstream (depth.py), so the three
functionals in the grid share one curve per (target, split, probe).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TargetArray

LOWRANK_MAX = 64


def group_split(groups: np.ndarray, seed: int, test_frac: float = 0.3):
    """One deterministic 70/30 split on the grouping key (positions into `groups`)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx = np.arange(len(groups))
    train, test = next(gss.split(idx, groups=groups))
    return train, test


def subsample(train_pos: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n is None or len(train_pos) <= n:
        return train_pos
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_pos, size=n, replace=False))


def _build_probe(kind: str, probe: str, n_train: int, d: int, seed: int) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if probe == "lowrank":
        rank = min(LOWRANK_MAX, n_train - 1, d - 1)
        rank = max(rank, 1)
        steps.append(
            ("svd", TruncatedSVD(n_components=rank, algorithm="randomized", n_iter=5, random_state=seed))
        )
    if kind == "categorical":
        steps.append(("clf", RidgeClassifier(alpha=1.0)))
    else:
        steps.append(("reg", Ridge(alpha=1.0)))
    return Pipeline(steps)


def _score_categorical(y_train, y_pred, y_test) -> float:
    _, counts = np.unique(y_test, return_counts=True)
    b = counts.max() / len(y_test)
    if b >= 1.0:
        return 0.0
    acc = float(np.mean(y_pred == y_test))
    return max(0.0, (acc - b) / (1.0 - b))


def _score_continuous(y_pred, y_test) -> float:
    if np.all(y_pred == y_pred[0]) or np.all(y_test == y_test[0]):
        return 0.0
    rho, _ = spearmanr(y_pred, y_test)
    if not np.isfinite(rho):
        return 0.0
    return max(0.0, float(rho))


def fit_score(X_tr, y_tr, X_te, y_te, kind: str, probe: str, seed: int) -> float:
    if kind == "categorical" and len(np.unique(y_tr)) < 2:
        return 0.0
    pipe = _build_probe(kind, probe, X_tr.shape[0], X_tr.shape[1], seed)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    if kind == "categorical":
        return _score_categorical(y_tr, y_pred, y_te)
    return _score_continuous(y_pred, y_te)


def score_curve(
    states: list[np.ndarray],
    target: TargetArray,
    group_key: np.ndarray,
    probe: str,
    seed: int,
    train_n: int,
) -> np.ndarray:
    """S over layers for one (target, split design, probe family)."""
    idx = np.flatnonzero(target.mask)
    g = group_key[idx]
    y = target.labels[idx]
    tr, te = group_split(g, seed)
    tr = subsample(tr, train_n, seed)
    y_tr, y_te = y[tr], y[te]
    curve = np.zeros(len(states), dtype=float)
    for layer, X in enumerate(states):
        Xl = X[idx]
        curve[layer] = fit_score(Xl[tr], y_tr, Xl[te], y_te, target.kind, probe, seed)
    return curve
