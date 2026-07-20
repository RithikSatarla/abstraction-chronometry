"""Learned-origin evidence criterion (paper Def. 1).

A learned-origin claim passes at level alpha only if BOTH:
  (a) the selected final-checkpoint max statistic rejects under the joint
      target-permutation null, and
  (b) the selected final-minus-initialization contrast is positive AND rejects
      under the *same* orbit.

Per-protocol pass counts are descriptive, not the criterion.
"""

from __future__ import annotations

from .nulls import (
    contrast_test,
    max_selection_test,
    partial_order_contrast,
    partial_order_max_test,
)


def learned_origin_criterion(final_cells, init_cells, ladder, alpha: float = 0.05) -> dict:
    """Total-ladder learned-origin evaluation for one selection set (e.g. one
    model size's 12 protocols, or all sizes x protocols pooled)."""
    final = max_selection_test(final_cells, ladder)
    init = max_selection_test(init_cells, ladder)
    contrast = contrast_test(final_cells, init_cells, ladder)

    final_rejects = _rej(final["p"], alpha)
    contrast_rejects = contrast["gamma"] > 0 and _rej(contrast["p"], alpha)

    return {
        "Hf_max": final["H_max"],
        "pf": final["p"],
        "Hi_max": init["H_max"],
        "pi": init["p"],
        "init_max": contrast["init_max"],   # == Hi_max, kept explicit for the table
        "final_max": contrast["final_max"],  # == Hf_max
        "gamma": contrast["gamma"],
        "p_gamma": contrast["p"],
        "orbit": final["orbit"],
        "final_rejects": final_rejects,
        "contrast_rejects": contrast_rejects,
        "passes": final_rejects and contrast_rejects,
    }


def learned_origin_partial(final_cells, init_cells, pairs, active, alpha: float = 0.05) -> dict:
    """Defended-pair (partial-order) learned-origin evaluation."""
    final = partial_order_max_test(final_cells, pairs, active)
    init = partial_order_max_test(init_cells, pairs, active)
    contrast = partial_order_contrast(final_cells, init_cells, pairs, active)

    final_rejects = _rej(final["p"], alpha)
    contrast_rejects = contrast["gamma"] > 0 and _rej(contrast["p"], alpha)

    return {
        "HR_max": final["H_max"],
        "pf": final["p"],
        "HR_init_max": init["H_max"],
        "gamma": contrast["gamma"],
        "p_gamma": contrast["p"],
        "orbit": final["orbit"],
        "final_rejects": final_rejects,
        "contrast_rejects": contrast_rejects,
        "passes": final_rejects and contrast_rejects,
    }


def _rej(p_value: float, alpha: float) -> bool:
    return p_value <= alpha + 1e-12
