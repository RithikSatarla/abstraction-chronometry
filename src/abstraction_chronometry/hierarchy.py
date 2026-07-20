"""Hierarchy scores H(M,p) = A(rank(d_p(M)), pi0).

`A` is a rank-agreement statistic (Spearman rho or Kendall-style H_K); ties in
the ladder pi0 are handled with average ranks (Spearman) or the 0/0.5/1 tie
convention (partial-order score H_R, paper Prop. 6).
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.stats import rankdata


def average_ranks(x) -> np.ndarray:
    """Ranks with average-rank tie handling (Spearman convention)."""
    return rankdata(np.asarray(x, dtype=float), method="average")


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    if denom == 0.0:
        return 0.0
    return float((ac @ bc) / denom)


def spearman_from_ranks(depth_ranks, ladder_ranks) -> float:
    """Spearman rho = Pearson correlation of the two rank vectors."""
    return pearson(depth_ranks, ladder_ranks)


def kendall_from_ranks(depth_ranks, ladder_ranks) -> float:
    """Kendall-style H_K = 1 - 2V/C(m,2), V = # discordant pairs.

    Pairs tied in the ladder (ladder difference 0) do not count as violations,
    matching the H_K convention used in the paper's Mahonian discussion.
    """
    dr = np.asarray(depth_ranks, dtype=float)
    lr = np.asarray(ladder_ranks, dtype=float)
    m = dr.shape[0]
    violations = 0
    for i in range(m):
        for j in range(i + 1, m):
            if (dr[i] - dr[j]) * (lr[i] - lr[j]) < 0:
                violations += 1
    return 1.0 - 2.0 * violations / comb(m, 2)


def hierarchy_score(depths, ladder, stat: str = "spearman") -> float:
    """H(M,p): agreement between the depth-induced ranks and the ladder pi0."""
    dr = average_ranks(depths)
    lr = average_ranks(ladder)
    if stat == "spearman":
        return spearman_from_ranks(dr, lr)
    if stat == "kendall":
        return kendall_from_ranks(dr, lr)
    raise ValueError(f"unknown stat: {stat}")


def partial_order_score(depths, pairs) -> float:
    """H_R(d; R0) = |R0|^-1 sum_{(a,b) in R0} psi(d_a, d_b),
    psi(x,y) = 1{x<y} + 0.5*1{x=y}  (paper Sec. 4 / Prop. 6)."""
    d = np.asarray(depths, dtype=float)
    if len(pairs) == 0:
        raise ValueError("empty defended pair set")
    total = 0.0
    for a, b in pairs:
        if d[a] < d[b]:
            total += 1.0
        elif d[a] == d[b]:
            total += 0.5
    return total / len(pairs)
