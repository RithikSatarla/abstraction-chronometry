"""Render Tables 1-3 (text + LaTeX) and the summary numbers used in Sec. 7 prose.

Every number is recomputed from saved cell groups via analysis.py, so the tables
cannot disagree with each other (in particular Gamma == Final max - Init max).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from .analysis import (
    evaluate_partial,
    evaluate_partial_pooled,
    evaluate_pooled,
    evaluate_size,
    evaluate_trajectory,
)
from .data import TARGETS_6


def f3(x) -> str:
    """Format like the paper: 3 decimals, no leading zero, '-' for None/NaN."""
    if x is None:
        return "--"
    try:
        if x != x:  # NaN
            return "--"
    except TypeError:
        return "--"
    s = f"{x:.3f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


# --------------------------------------------------------------------------- #
# Table 1: five-size scale sweep
# --------------------------------------------------------------------------- #


def table1(record: dict) -> dict:
    # Records written after the second-family run name their own revisions;
    # older Pythia records fall back to the config defaults.
    final_rev = record.get("final_revision", C.FINAL_REVISION)
    init_rev = record.get("init_revision", C.INIT_REVISION)
    sizes = list(record["groups"].keys())
    finals, inits, rows = [], [], []
    for size in sizes:
        g = record["groups"][size]
        fcg, icg = g[final_rev], g[init_rev]
        finals.append(fcg)
        inits.append(icg)
        r = evaluate_size(fcg, icg, TARGETS_6)
        rows.append((size, r))
    pooled = evaluate_pooled(finals, inits, TARGETS_6)
    return {"rows": rows, "pooled": pooled}


def render_table1_text(t1: dict) -> str:
    lines = [
        "Table 1  Five-size Pythia audit (all-six-defined total-ladder rule)",
        f"{'Size':>5} {'Def0/f/m':>9} {'Hfmax':>6} {'Sel.':>16} {'pf':>5} "
        f"{'Init':>5} {'Final':>6} {'Gamma':>6} {'pG':>5} {'pass':>5}",
    ]
    for size, r in t1["rows"]:
        defs = f"{r['def_init']}/{r['def_final']}/{r['def_matched']}"
        lines.append(
            f"{size.upper():>5} {defs:>9} {f3(r['Hf_max']):>6} {str(r['sel']):>16} "
            f"{f3(r['pf']):>5} {f3(r['init_max']):>5} {f3(r['final_max']):>6} "
            f"{f3(r['gamma']):>6} {f3(r['p_gamma']):>5} {str(r['passes']):>5}"
        )
    p = t1["pooled"]
    lines.append(
        f"POOLED  final max H={f3(p['final_max'])} p={f3(p['pf'])} | "
        f"init max={f3(p['init_max'])} p={f3(p['pi'])} | "
        f"Gamma={f3(p['gamma'])} p={f3(p['p_gamma'])} | "
        f"diam={f3(p['diam'])} (Hmin={f3(p['H_min'])}, Hmax={f3(p['H_max_all'])})"
    )
    return "\n".join(lines)


def render_table1_latex(t1: dict) -> str:
    head = (
        "\\begin{tabular}{lllllrrrr}\n\\toprule\n"
        "Size & Def. 0/f/m & $H^f_{\\max}$ & Sel. & $p_f$ & Init max & Final max "
        "& $\\Gamma$ & $p_\\Gamma$ \\\\\n\\midrule"
    )
    body = []
    for size, r in t1["rows"]:
        defs = f"{r['def_init']}/{r['def_final']}/{r['def_matched']}"
        body.append(
            f"{size.upper()} & {defs} & {f3(r['Hf_max'])} & {r['sel']} & {f3(r['pf'])} "
            f"& {f3(r['init_max'])} & {f3(r['final_max'])} & {f3(r['gamma'])} & {f3(r['p_gamma'])} \\\\"
        )
    return head + "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}"


# --------------------------------------------------------------------------- #
# Table 2: 410M trajectory
# --------------------------------------------------------------------------- #


def table2(record: dict) -> list[dict]:
    return evaluate_trajectory(record, TARGETS_6)


def render_table2_text(rows: list[dict]) -> str:
    lines = [
        "Table 2  Pythia-410M trajectory (full English-EWT dev)",
        f"{'Checkpoint':>10} {'Def':>6} {'Hcmax':>6} {'Match':>6} {'pc':>5} {'Gamma':>6} {'pG':>5}",
    ]
    for r in rows:
        lines.append(
            f"{r['checkpoint']:>10} {r['def']:>6} {f3(r['Hc_max']):>6} "
            f"{f3(r['matched_max']):>6} {f3(r['pc']):>5} {f3(r['gamma']):>6} {f3(r['p_gamma']):>5}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table 3: seed robustness (three sizes, two seeds)
# --------------------------------------------------------------------------- #


def table3(record_primary: dict, record_secondary: dict, sizes=("70m", "160m", "410m")) -> list[dict]:
    rows = []
    for rec, seed in ((record_primary, record_primary["seed"]), (record_secondary, record_secondary["seed"])):
        for size in sizes:
            if size not in rec["groups"]:
                continue
            g = rec["groups"][size]
            r = evaluate_size(g[C.FINAL_REVISION], g[C.INIT_REVISION], TARGETS_6)
            rows.append({"size": size, "seed": seed, **r})
    rows.sort(key=lambda x: (["70m", "160m", "410m", "1b", "1.4b"].index(x["size"]), x["seed"]))
    return rows


def render_table3_text(rows: list[dict]) -> str:
    lines = [
        "Table 3  Seed robustness (primary vs independent seed)",
        f"{'Size':>5} {'Seed':>5} {'Def0/f/m':>9} {'Init':>5} {'Final':>6} {'pf':>5} {'Gamma':>6} {'pG':>5}",
    ]
    for r in rows:
        defs = f"{r['def_init']}/{r['def_final']}/{r['def_matched']}"
        lines.append(
            f"{r['size'].upper():>5} {r['seed']:>5} {defs:>9} {f3(r['init_max']):>5} "
            f"{f3(r['final_max']):>6} {f3(r['pf']):>5} {f3(r['gamma']):>6} {f3(r['p_gamma']):>5}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #


def build_report(sweep_path: Path, traj_path: Path | None = None, seed2_path: Path | None = None) -> str:
    sweep = json.loads(Path(sweep_path).read_text())
    out = [render_table1_text(t1 := table1(sweep)), ""]

    finals = [sweep["groups"][s][C.FINAL_REVISION] for s in sweep["groups"]]
    inits = [sweep["groups"][s][C.INIT_REVISION] for s in sweep["groups"]]
    part_pool = evaluate_partial_pooled(finals, inits)
    best = max(
        (
            (s, evaluate_partial(sweep["groups"][s][C.FINAL_REVISION], sweep["groups"][s][C.INIT_REVISION]))
            for s in sweep["groups"]
        ),
        key=lambda kv: kv[1]["HR_max"],
    )
    out.append(
        f"Partial order: best final {best[0].upper()} HR_max={f3(best[1]['HR_max'])} "
        f"p={f3(best[1]['pf'])}; pooled Gamma_R={f3(part_pool['gamma'])} p={f3(part_pool['p_gamma'])}"
    )
    out.append("")

    if traj_path and Path(traj_path).exists():
        traj = json.loads(Path(traj_path).read_text())
        out.append(render_table2_text(table2(traj)))
        out.append("")
    if seed2_path and Path(seed2_path).exists():
        seed2 = json.loads(Path(seed2_path).read_text())
        out.append(render_table3_text(table3(sweep, seed2)))
        out.append("")

    out.append("LaTeX (Table 1):")
    out.append(render_table1_latex(t1))
    return "\n".join(out)
