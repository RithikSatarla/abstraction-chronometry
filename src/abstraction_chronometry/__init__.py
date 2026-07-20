"""Abstraction Chronometry: finite-protocol invariance for layerwise probing.

BlackboxNLP 2026 reproducibility artifact. The `depth`, `hierarchy`, `nulls`, and
`criterion` modules are the model-agnostic statistics core (paper Sec. 4-6); the
`data`, `extract`, and `probes` modules produce the per-layer probe curves the
core consumes.
"""

from .criterion import learned_origin_criterion, learned_origin_partial
from .depth import apply_functional, center_of_mass, normalize, onset, peak
from .hierarchy import hierarchy_score, partial_order_score
from .nulls import (
    contrast_test,
    diameter_test,
    mahonian_counts,
    mahonian_pvalue,
    max_selection_test,
    min_attainable_p,
    partial_order_max_test,
    rejects_at,
)

__all__ = [
    "center_of_mass",
    "peak",
    "onset",
    "apply_functional",
    "normalize",
    "hierarchy_score",
    "partial_order_score",
    "max_selection_test",
    "diameter_test",
    "contrast_test",
    "partial_order_max_test",
    "mahonian_counts",
    "mahonian_pvalue",
    "rejects_at",
    "min_attainable_p",
    "learned_origin_criterion",
    "learned_origin_partial",
]
