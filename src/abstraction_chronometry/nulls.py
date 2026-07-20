"""Exact finite nulls (paper Sec. 5-6).

The random-ladder null fixes the estimated depths and enumerates the finite orbit
of target relabelings. Selection over a protocol grid is corrected with a *joint*
permutation applied to every cell before the selection functional F (Theorem 5).

Key invariant enforced here: the selected final max, selected init max, and the
contrast Gamma are all read off the SAME orbit machinery, so
    Gamma == Final_max - Init_max
holds by construction. (The paper's Appendix C Table 3 violated this; see
tests/test_criterion.py::test_gamma_matches_final_minus_init.)
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from .hierarchy import average_ranks, partial_order_score, pearson

EPS = 1e-9

# --------------------------------------------------------------------------- #
# Total-ladder orbit (six-target Spearman audit)
# --------------------------------------------------------------------------- #


def ladder_orbit(ladder) -> list[np.ndarray]:
    """Unique arrangements of the ladder rank multiset = the finite orbit Pi(pi0).

    For the tied six-target ladder [1,2,3,3,4,5] this yields 6!/2! = 360 vectors.
    """
    lr = tuple(average_ranks(ladder))
    return [np.asarray(p, dtype=float) for p in sorted(set(permutations(lr)))]


def _cell_depth_ranks(cells) -> list[np.ndarray | None]:
    return [None if d is None else average_ranks(d) for d in cells]


def _max_over_cells(depth_ranks: list[np.ndarray | None], ladder_vec: np.ndarray) -> float:
    best = -np.inf
    for dr in depth_ranks:
        if dr is None:
            continue
        h = pearson(dr, ladder_vec)
        if h > best:
            best = h
    return best


def _minmax_over_cells(depth_ranks, ladder_vec) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for dr in depth_ranks:
        if dr is None:
            continue
        h = pearson(dr, ladder_vec)
        lo = min(lo, h)
        hi = max(hi, h)
    return lo, hi


def max_selection_test(cells, ladder) -> dict:
    """Selected max H over a grid + its exact joint-orbit p-value (F = max)."""
    orbit = ladder_orbit(ladder)
    dranks = _cell_depth_ranks(cells)
    obs_vec = average_ranks(ladder)
    h_obs = _max_over_cells(dranks, obs_vec)
    tail = sum(1 for lv in orbit if _max_over_cells(dranks, lv) >= h_obs - EPS)
    return {"H_max": h_obs, "p": tail / len(orbit), "count": tail, "orbit": len(orbit)}


def diameter_test(cells, ladder) -> dict:
    """Protocol-diameter test (F = max - min)."""
    orbit = ladder_orbit(ladder)
    dranks = _cell_depth_ranks(cells)
    lo, hi = _minmax_over_cells(dranks, average_ranks(ladder))
    d_obs = hi - lo
    tail = 0
    for lv in orbit:
        clo, chi = _minmax_over_cells(dranks, lv)
        if (chi - clo) >= d_obs - EPS:
            tail += 1
    return {"H_min": lo, "H_max": hi, "diam": d_obs, "p": tail / len(orbit), "orbit": len(orbit)}


def contrast_test(final_cells, init_cells, ladder) -> dict:
    """Final-minus-initialization contrast Gamma and its exact p-value under the
    *same* joint orbit (paper Def. 1). Gamma = max_p H(M_f) - max_p H(M_0)."""
    orbit = ladder_orbit(ladder)
    df = _cell_depth_ranks(final_cells)
    di = _cell_depth_ranks(init_cells)
    obs_vec = average_ranks(ladder)
    final_max = _max_over_cells(df, obs_vec)
    init_max = _max_over_cells(di, obs_vec)
    gamma_obs = final_max - init_max
    tail = 0
    for lv in orbit:
        z = _max_over_cells(df, lv) - _max_over_cells(di, lv)
        if z >= gamma_obs - EPS:
            tail += 1
    return {
        "final_max": final_max,
        "init_max": init_max,
        "gamma": gamma_obs,
        "p": tail / len(orbit),
        "count": tail,
        "orbit": len(orbit),
    }


# --------------------------------------------------------------------------- #
# Partial-order (defended pair set) null  (paper Prop. 6)
# --------------------------------------------------------------------------- #


def partial_order_max_test(cells, pairs, active_targets) -> dict:
    """Max over a grid of H_R with the exact relabeling null over S(T_R)."""
    active = list(active_targets)
    perms = list(permutations(active))

    def hr(depths, perm) -> float:
        relabel = dict(zip(active, perm))
        d = np.asarray(depths, dtype=float)
        remapped = d.copy()
        for orig, slot in relabel.items():
            remapped[orig] = d[slot]
        return partial_order_score(remapped, pairs)

    def maxcell(perm) -> float:
        best = -np.inf
        for d in cells:
            if d is None:
                continue
            best = max(best, hr(d, perm))
        return best

    obs = maxcell(tuple(active))
    tail = sum(1 for p in perms if maxcell(p) >= obs - EPS)
    return {"H_max": obs, "p": tail / len(perms), "count": tail, "orbit": len(perms)}


def partial_order_contrast(final_cells, init_cells, pairs, active_targets) -> dict:
    active = list(active_targets)
    perms = list(permutations(active))

    def hr(depths, perm) -> float:
        relabel = dict(zip(active, perm))
        d = np.asarray(depths, dtype=float)
        remapped = d.copy()
        for orig, slot in relabel.items():
            remapped[orig] = d[slot]
        return partial_order_score(remapped, pairs)

    def maxcell(cells, perm) -> float:
        best = -np.inf
        for d in cells:
            if d is None:
                continue
            best = max(best, hr(d, perm))
        return best

    ident = tuple(active)
    final_max = maxcell(final_cells, ident)
    init_max = maxcell(init_cells, ident)
    gamma = final_max - init_max
    tail = sum(
        1 for p in perms if (maxcell(final_cells, p) - maxcell(init_cells, p)) >= gamma - EPS
    )
    return {
        "final_max": final_max,
        "init_max": init_max,
        "gamma": gamma,
        "p": tail / len(perms),
        "orbit": len(perms),
    }


# --------------------------------------------------------------------------- #
# Mahonian closed form (strict total ladder, Kendall H_K)  (paper Sec. 5)
# --------------------------------------------------------------------------- #


def mahonian_counts(m: int) -> np.ndarray:
    """Counts of permutations of m by inversion number V: coefficients of
    prod_{i=1..m}(1 + q + ... + q^{i-1}).  Index v -> #perms with V=v."""
    coeffs = np.array([1.0])
    for i in range(1, m + 1):
        coeffs = np.convolve(coeffs, np.ones(i))
    return coeffs


def mahonian_pvalue(v_obs: int, m: int) -> float:
    """Exact one-sided p for a strict total ladder: P(V <= v_obs) under the
    uniform permutation null (high agreement = few inversions)."""
    counts = mahonian_counts(m)
    return float(counts[: v_obs + 1].sum() / counts.sum())


# --------------------------------------------------------------------------- #
# Finite-resolution boundary  (paper Prop. 5)
# --------------------------------------------------------------------------- #


def rejects_at(p_value: float, orbit_size: int, alpha: float = 0.05) -> bool:
    """Level-alpha one-sided rejection: r(Z) <= floor(alpha*|Pi|)."""
    r = round(p_value * orbit_size)
    return r <= int(np.floor(alpha * orbit_size))


def min_attainable_p(orbit_size: int) -> float:
    return 1.0 / orbit_size
