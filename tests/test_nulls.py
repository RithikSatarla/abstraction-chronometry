"""Exact-null validation against brute force."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np

from abstraction_chronometry.nulls import (
    contrast_test,
    ladder_orbit,
    mahonian_counts,
    mahonian_pvalue,
    max_selection_test,
    min_attainable_p,
    rejects_at,
)


def _inversions(perm) -> int:
    return sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )


def test_mahonian_matches_bruteforce():
    for m in range(1, 8):
        counts = mahonian_counts(m)
        brute = np.zeros(int(counts.sum()) and len(counts))
        brute = np.zeros(len(counts))
        for perm in permutations(range(m)):
            brute[_inversions(perm)] += 1
        assert counts.sum() == factorial(m)
        assert np.array_equal(counts, brute)


def test_mahonian_min_pvalue():
    # Perfect one-sided total order (V=0): min exact p = 1/m!  (paper Sec. 5).
    assert abs(mahonian_pvalue(0, 3) - 1 / 6) < 1e-12
    assert abs(mahonian_pvalue(0, 4) - 1 / 24) < 1e-12
    # m=3 at alpha=.05 cannot reject: 1/6 > .05.
    assert not rejects_at(mahonian_pvalue(0, 3), 6, alpha=0.05)


def test_tied_ladder_orbit_is_360():
    ladder = [1, 2, 3, 3, 4, 5]  # word_shape<UPOS<Number(x2)<deprel<tree_depth
    assert len(ladder_orbit(ladder)) == 360  # 6!/2!


def test_strict_ladder_orbit_is_120():
    assert len(ladder_orbit([1, 2, 3, 4, 5])) == 120  # 5!


def test_max_selection_pvalue_is_orbit_fraction():
    # A single cell whose depths equal the ladder ranks -> perfect H=1.
    ladder = [1, 2, 3, 3, 4, 5]
    perfect = np.array([0.1, 0.2, 0.3, 0.3, 0.4, 0.5])  # ranks match ladder
    res = max_selection_test([perfect], ladder)
    assert abs(res["H_max"] - 1.0) < 1e-9
    # p must be an integer multiple of 1/360 (finite resolution, Prop. 5).
    assert abs(res["p"] * 360 - round(res["p"] * 360)) < 1e-9
    assert res["p"] >= min_attainable_p(360) - 1e-12


def test_resolution_boundary_matches_paper_counts():
    # 8/360 rejects at .05 (floor(.05*360)=18); 28/360 does not.
    assert rejects_at(8 / 360, 360, 0.05)
    assert not rejects_at(28 / 360, 360, 0.05)


def test_contrast_uses_same_orbit_denominator():
    ladder = [1, 2, 3, 3, 4, 5]
    final = [np.array([0.1, 0.2, 0.3, 0.35, 0.5, 0.6])]
    init = [np.array([0.3, 0.1, 0.5, 0.2, 0.4, 0.35])]
    res = contrast_test(final, init, ladder)
    assert res["orbit"] == 360
    assert abs(res["gamma"] - (res["final_max"] - res["init_max"])) < 1e-9
