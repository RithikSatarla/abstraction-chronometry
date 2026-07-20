"""Depth functionals D: map a layer-indexed score curve S_{t,.} to a scalar depth d_t.

The audit (paper Sec. 7) uses three functionals: center of mass, peak (averaging
maximizers), and 80% onset. Each returns a raw depth in layer units (0..Lmax) or
None when the depth is *undefined* (Appendix B: "undefined when the probe curve is
empty or the chosen center-of-mass/onset statistic has no positive score mass").
Normalization by Lmax(M) is applied separately (see `normalize`).

Scores S are already non-negative and anchored at 0 = held-out majority baseline
(see probes.score_categorical / score_continuous), so "positive score mass" means
sum(S) > 0 for center of mass and max(S) > 0 for onset.
"""

from __future__ import annotations

import numpy as np


def center_of_mass(scores: np.ndarray) -> float | None:
    """Score-weighted mean layer.  Undefined when total score mass is zero."""
    s = np.asarray(scores, dtype=float)
    total = s.sum()
    if not total > 0:
        return None
    layers = np.arange(s.shape[0], dtype=float)
    return float((layers * s).sum() / total)


def peak(scores: np.ndarray) -> float | None:
    """Mean of all argmax layers (averages ties, including all-layer ties).

    Always defined for a non-empty curve, per Appendix B ("peak depth averages
    all maximizers, including all-layer ties").
    """
    s = np.asarray(scores, dtype=float)
    if s.shape[0] == 0:
        return None
    maximizers = np.flatnonzero(s == s.max())
    return float(maximizers.mean())


def onset(scores: np.ndarray, frac: float = 0.8) -> float | None:
    """First layer reaching `frac` of the peak score.  Undefined when max <= 0."""
    s = np.asarray(scores, dtype=float)
    peak_val = s.max() if s.shape[0] else 0.0
    if not peak_val > 0:
        return None
    idx = np.flatnonzero(s >= frac * peak_val)
    return float(idx[0])


# Registry keyed by the short protocol names used in the tables (COM / Pk / On).
FUNCTIONALS = {
    "COM": center_of_mass,
    "Pk": peak,
    "On": onset,
}


def apply_functional(name: str, scores: np.ndarray) -> float | None:
    return FUNCTIONALS[name](scores)


def normalize(depth: float | None, lmax: int) -> float | None:
    """Normalize a raw layer depth by Lmax(M) (number of hidden states minus 1)."""
    if depth is None:
        return None
    return depth / lmax
