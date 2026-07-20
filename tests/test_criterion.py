"""Learned-origin criterion + the arithmetic invariant the paper's Table 3 broke.

The submitted draft's Appendix C reported (size 4177 seed) e.g. Final max = .939,
Gamma = .826, which implies Init max = .113 -- inconsistent with every other init
value in the paper. Here we assert the engine can never emit that: Gamma is always
exactly Final_max - Init_max, and the reported Init max column equals it.
"""

from __future__ import annotations

import numpy as np

from abstraction_chronometry.criterion import learned_origin_criterion

LADDER = [1, 2, 3, 3, 4, 5]
RNG = np.random.default_rng(0)


def _random_grid(n_cells, undefined_frac=0.25):
    cells = []
    for _ in range(n_cells):
        if RNG.random() < undefined_frac:
            cells.append(None)
        else:
            cells.append(RNG.random(6))
    return cells


def test_gamma_equals_final_minus_init():
    for _ in range(200):
        final = _random_grid(12)
        init = _random_grid(12)
        # guarantee at least one defined cell per side
        final[0] = RNG.random(6)
        init[0] = RNG.random(6)
        res = learned_origin_criterion(final, init, LADDER)
        assert abs(res["gamma"] - (res["final_max"] - res["init_max"])) < 1e-9
        assert abs(res["final_max"] - res["Hf_max"]) < 1e-9
        assert abs(res["init_max"] - res["Hi_max"]) < 1e-9


def test_implied_init_max_is_in_range():
    # Whatever Gamma comes out, Init max = Final max - Gamma must stay a real
    # hierarchy score in [-1, 1]; the paper's .113/.171 implied inits could only
    # arise from a hand-typed Gamma, not this engine.
    for _ in range(200):
        final = _random_grid(12)
        init = _random_grid(12)
        final[0] = RNG.random(6)
        init[0] = RNG.random(6)
        res = learned_origin_criterion(final, init, LADDER)
        implied_init = res["final_max"] - res["gamma"]
        assert -1.0 - 1e-9 <= implied_init <= 1.0 + 1e-9
        assert abs(implied_init - res["init_max"]) < 1e-9


def test_criterion_two_part_gate():
    # Final passes but contrast fails -> criterion fails (the paper's whole point).
    strong_final = [np.array([0.05, 0.15, 0.3, 0.32, 0.6, 0.8])]  # matches ladder well
    strong_init = [np.array([0.05, 0.15, 0.3, 0.32, 0.6, 0.8])]  # identical -> Gamma=0
    res = learned_origin_criterion(strong_final, strong_init, LADDER)
    assert res["gamma"] == 0.0
    assert not res["contrast_rejects"]
    assert not res["passes"]
