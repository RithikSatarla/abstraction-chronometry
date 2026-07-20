"""Depth functionals and rank-agreement scores."""

from __future__ import annotations

import numpy as np

from abstraction_chronometry.depth import center_of_mass, normalize, onset, peak
from abstraction_chronometry.hierarchy import (
    hierarchy_score,
    partial_order_score,
)


def test_center_of_mass():
    assert center_of_mass([0, 0, 1, 0]) == 2.0
    assert abs(center_of_mass([1, 1, 1, 1]) - 1.5) < 1e-9  # uniform -> mean layer
    assert center_of_mass([0, 0, 0, 0]) is None  # no positive mass -> undefined


def test_peak_averages_maximizers():
    assert peak([0, 1, 0, 1]) == 2.0  # mean of layers 1 and 3
    assert peak([0, 0, 0, 0]) == 1.5  # all-layer tie -> mean layer (always defined)
    assert peak([0, 0, 5]) == 2.0


def test_onset():
    assert onset([0.0, 0.5, 1.0], frac=0.8) == 2.0  # first layer >= 0.8*max
    assert onset([0.0, 0.9, 1.0], frac=0.8) == 1.0
    assert onset([0.0, 0.0, 0.0]) is None


def test_normalize():
    assert normalize(3.0, 6) == 0.5
    assert normalize(None, 6) is None


def test_hierarchy_score_perfect_and_reversed():
    ladder = [1, 2, 3, 3, 4, 5]
    perfect = [0.1, 0.2, 0.3, 0.3, 0.4, 0.5]
    reversed_ = [0.5, 0.4, 0.3, 0.3, 0.2, 0.1]
    assert abs(hierarchy_score(perfect, ladder) - 1.0) < 1e-9
    assert hierarchy_score(reversed_, ladder) < 0


def test_partial_order_score():
    # pairs: word_shape(0) before UPOS(1), deprel(2), tree_depth(3)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    depths = [0.1, 0.2, 0.4, 0.5]  # perfectly ordered
    assert partial_order_score(depths, pairs) == 1.0
    tied = [0.1, 0.1, 0.4, 0.5]  # 0==1 tie -> 0.5 credit on (0,1)
    assert abs(partial_order_score(tied, pairs) - (4 + 0.5) / 5) < 1e-9
