"""Turn saved cell groups into the paper's Table 1/2/3 rows + criterion verdicts.

All hierarchy scores, selected maxima, and contrasts flow through the stats core
in nulls.py / criterion.py, so the reported Init max, Final max, and Gamma are
mutually consistent by construction (Gamma == Final max - Init max exactly).
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .criterion import learned_origin_criterion, learned_origin_partial
from .data import ACTIVE_4, LADDER_6, PAIRS_4, TARGETS_4
from .depth import apply_functional, normalize
from .hierarchy import hierarchy_score
from .nulls import contrast_test, max_selection_test


# --------------------------------------------------------------------------- #
# Building depth vectors from stored curves (any target subset)
# --------------------------------------------------------------------------- #


def depth_vector(cg: dict, targets: list[str], functional: str, split: str, probe: str):
    """Depth vector (list of float|None) for one protocol over `targets`."""
    lmax = cg["lmax"]
    vec = []
    for tname in targets:
        curve = np.asarray(cg["curves"][f"{tname}|{split}|{probe}"], dtype=float)
        vec.append(normalize(apply_functional(functional, curve), lmax))
    return vec


def cells_for(cg: dict, targets: list[str]):
    """dict protocol_name -> (np.array or None). None if any target undefined."""
    out = {}
    for f, s, p in C.protocol_cells():
        vec = depth_vector(cg, targets, f, s, p)
        out[C.protocol_name(f, s, p)] = None if any(v is None for v in vec) else np.array(vec)
    return out


def selected_protocol(cells: dict, ladder) -> tuple[str | None, float]:
    best_name, best_h = None, -np.inf
    for name, v in cells.items():
        if v is None:
            continue
        h = hierarchy_score(v, ladder)
        if h > best_h:
            best_h, best_name = h, name
    return best_name, best_h


# --------------------------------------------------------------------------- #
# Six-target total-ladder audit (Table 1 + pooled result)
# --------------------------------------------------------------------------- #


def evaluate_size(cg_final: dict, cg_init: dict, targets: list[str], ladder=LADDER_6) -> dict:
    fcells = cells_for(cg_final, targets)
    icells = cells_for(cg_init, targets)
    res = learned_origin_criterion(list(fcells.values()), list(icells.values()), ladder)
    sel_name, _ = selected_protocol(fcells, ladder)
    res.update(
        sel=sel_name,
        def_init=sum(v is not None for v in icells.values()),
        def_final=sum(v is not None for v in fcells.values()),
        def_matched=sum(fcells[k] is not None and icells[k] is not None for k in fcells),
    )
    return res


def evaluate_pooled(finals: list[dict], inits: list[dict], targets: list[str], ladder=LADDER_6) -> dict:
    fpool, ipool = [], []
    for cg in finals:
        fpool.extend(cells_for(cg, targets).values())
    for cg in inits:
        ipool.extend(cells_for(cg, targets).values())
    res = learned_origin_criterion(fpool, ipool, ladder)
    # protocol diameter over all pooled final cells
    hs = [hierarchy_score(v, ladder) for v in fpool if v is not None]
    res.update(H_min=min(hs), H_max_all=max(hs), diam=max(hs) - min(hs), n_final_cells=len(fpool))
    return res


# --------------------------------------------------------------------------- #
# Trajectory (Table 2): matched max relative to step0's defined cells
# --------------------------------------------------------------------------- #


def evaluate_trajectory(record: dict, targets: list[str], ladder=LADDER_6) -> list[dict]:
    cps = record["checkpoints"]
    step0 = cps[C.INIT_REVISION]
    step0_cells = cells_for(step0, targets)
    matched_keys = [k for k, v in step0_cells.items() if v is not None]
    step0_matched_max = max(
        (hierarchy_score(step0_cells[k], ladder) for k in matched_keys), default=float("nan")
    )

    rows = []
    for rev, cg in cps.items():
        cells = cells_for(cg, targets)
        defined = {k: v for k, v in cells.items() if v is not None}
        full = max_selection_test(list(cells.values()), ladder)
        matched_vals = [cells[k] for k in matched_keys if cells[k] is not None]
        matched_max = max(
            (hierarchy_score(v, ladder) for v in matched_vals), default=float("nan")
        )
        row = {
            "checkpoint": rev,
            "def": f"{len(defined)}/{len(cells)}",
            "Hc_max": full["H_max"],
            "pc": full["p"],
            "matched_max": matched_max,
        }
        if rev != C.INIT_REVISION:
            # contrast of matched maxes vs step0, exact joint orbit
            ct = contrast_test(matched_vals, list(step0_cells.values()), ladder)
            row["gamma"] = matched_max - step0_matched_max
            row["p_gamma"] = ct["p"]
        else:
            row["gamma"] = None
            row["p_gamma"] = None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Four-target partial order
# --------------------------------------------------------------------------- #


def evaluate_partial(cg_final: dict, cg_init: dict) -> dict:
    fcells = list(cells_for(cg_final, TARGETS_4).values())
    icells = list(cells_for(cg_init, TARGETS_4).values())
    return learned_origin_partial(fcells, icells, PAIRS_4, ACTIVE_4)


def evaluate_partial_pooled(finals: list[dict], inits: list[dict]) -> dict:
    fpool, ipool = [], []
    for cg in finals:
        fpool.extend(cells_for(cg, TARGETS_4).values())
    for cg in inits:
        ipool.extend(cells_for(cg, TARGETS_4).values())
    return learned_origin_partial(fpool, ipool, PAIRS_4, ACTIVE_4)
