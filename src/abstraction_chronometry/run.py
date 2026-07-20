"""Orchestration: extract -> probe -> depth curves -> per-cell depth vectors.

Produces a compact results record per (model size, checkpoint) that the stats
core and table builders consume. Heavy per-instance representations live only in
RAM for one (size, revision) at a time and are never persisted; only score curves
and depth vectors (small) are saved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from .data import TARGETS_6, Dataset, build_targets, group_keys, load_sentences
from .depth import apply_functional, normalize
from .extract import Extractor
from .probes import score_curve


@dataclass
class CellGroup:
    """All 12 protocol cells for one (size, revision)."""

    size: str
    revision: str
    n_layers: int
    lmax: int
    targets: list[str]
    # protocol_name -> depth vector (list of float|None over targets)
    cells: dict[str, list] = field(default_factory=dict)
    # (target, split, probe) joined by '|' -> per-layer score curve
    curves: dict[str, list] = field(default_factory=dict)


def _probe_cell_group(
    states,
    ds: Dataset,
    seed: int,
    train_n: int,
    target_names: list[str],
    lmax: int,
    n_layers: int,
    size: str,
    revision: str,
    targets=None,
    groups=None,
) -> CellGroup:
    """Probe already-extracted states for one seed. Extraction is seed-independent
    (deterministic model forward), so this is reused across seeds."""
    if targets is None:
        targets = build_targets(ds, target_names)
    if groups is None:
        groups = group_keys(ds)

    curves: dict[str, np.ndarray] = {}
    for tname in target_names:
        for split in C.SPLITS:
            for probe in C.PROBES:
                key = f"{tname}|{split}|{probe}"
                curves[key] = score_curve(
                    states, targets[tname], groups[split], probe, seed, train_n
                )

    cells: dict[str, list] = {}
    for f, s, p in C.protocol_cells():
        vec = []
        for tname in target_names:
            raw = apply_functional(f, curves[f"{tname}|{s}|{p}"])
            vec.append(normalize(raw, lmax))
        cells[C.protocol_name(f, s, p)] = vec

    return CellGroup(
        size=size,
        revision=revision,
        n_layers=n_layers,
        lmax=lmax,
        targets=list(target_names),
        cells=cells,
        curves={k: v.tolist() for k, v in curves.items()},
    )


def run_cell_group(
    size: str,
    revision: str,
    ds: Dataset,
    seed: int,
    train_n: int,
    target_names: list[str] = TARGETS_6,
    batch_size: int = 16,
    verbose: bool = True,
) -> CellGroup:
    ext = Extractor(size, revision=revision)
    if verbose:
        print(f"[extract] {size}@{revision}: {len(ds)} instances, {ext.n_states} states")
    states = ext.extract(ds, batch_size=batch_size, verbose=verbose)
    lmax, n_layers = ext.lmax, ext.n_layers
    ext.free()
    cg = _probe_cell_group(states, ds, seed, train_n, target_names, lmax, n_layers, size, revision)
    _release_states(states)
    ext.cleanup_states()
    return cg


def _release_states(states) -> None:
    """Close disk-backed memmap handles so their temp files can be deleted."""
    import gc

    for m in states:
        mm = getattr(m, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass
    gc.collect()


def _save(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def _batch_for(size: str, override: int | None) -> int:
    if override is not None:
        return override
    if "/" in size:  # explicit repo id (second family, ~1B) -> large-model batch
        return 8
    return 8 if size in ("1b", "1.4b") else 16  # RTX 3060 Ti (8 GB)


def run_scale_sweep(
    out_path: Path,
    sizes: list[str] = C.SIZES,
    seed: int = C.PRIMARY_SEED,
    n_sentences: int | None = C.SCALE_SWEEP_SENTENCES,
    train_n: int = C.TRAIN_N_SWEEP,
    batch_size: int | None = None,
    target_names: list[str] = TARGETS_6,
    data_path=None,
    init_revision: str = C.INIT_REVISION,
    final_revision: str = C.FINAL_REVISION,
) -> dict:
    out_path = Path(out_path)
    ds = load_sentences(path=data_path, limit=n_sentences)
    # resume: reuse any (size, revision) already saved
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {
            "kind": "scale_sweep",
            "seed": seed,
            "n_sentences": len(ds.sentences),
            "n_instances": len(ds),
            "targets": list(target_names),
            # recorded so analysis need not assume the Pythia revision names
            "init_revision": init_revision,
            "final_revision": final_revision,
            "groups": {},
        }
    for size in sizes:
        record["groups"].setdefault(size, {})
        for rev in (init_revision, final_revision):
            if rev in record["groups"][size]:
                print(f"[skip] {size}@{rev} already done")
                continue
            cg = run_cell_group(size, rev, ds, seed, train_n, target_names, _batch_for(size, batch_size))
            record["groups"][size][rev] = asdict(cg)
            _save(record, out_path)  # checkpoint after each (size, revision)
    return record


def run_scale_sweep_multiseed(
    out_dir: Path,
    seeds: list[int],
    sizes: list[str] = C.SIZES,
    n_sentences: int | None = C.SCALE_SWEEP_SENTENCES,
    train_n: int = C.TRAIN_N_SWEEP,
    batch_size: int | None = None,
    target_names: list[str] = TARGETS_6,
) -> dict[int, Path]:
    """Multi-seed scale sweep sharing one extraction per (size, revision).

    Writes one standard-format ``sweep_<seed>.json`` per seed (so all existing
    analysis/table code reads them unchanged). Resumable at the (size, revision,
    seed) granularity.
    """
    out_dir = Path(out_dir)
    ds = load_sentences(limit=n_sentences)
    targets = build_targets(ds, target_names)
    groups = group_keys(ds)

    paths = {s: out_dir / f"sweep_{s}.json" for s in seeds}
    records = {}
    for s in seeds:
        if paths[s].exists():
            records[s] = json.loads(paths[s].read_text())
        else:
            records[s] = {
                "kind": "scale_sweep",
                "seed": s,
                "n_sentences": len(ds.sentences),
                "n_instances": len(ds),
                "targets": list(target_names),
                "groups": {},
            }

    for size in sizes:
        for s in seeds:
            records[s]["groups"].setdefault(size, {})
        for rev in (C.INIT_REVISION, C.FINAL_REVISION):
            todo = [s for s in seeds if rev not in records[s]["groups"][size]]
            if not todo:
                print(f"[skip] {size}@{rev} (all {len(seeds)} seeds done)")
                continue
            ext = Extractor(size, revision=rev)
            print(f"[extract] {size}@{rev}: {len(ds)} instances, {ext.n_states} states "
                  f"-> probing seeds {todo}")
            states = ext.extract(ds, batch_size=_batch_for(size, batch_size))
            lmax, n_layers = ext.lmax, ext.n_layers
            ext.free()
            for s in todo:
                cg = _probe_cell_group(
                    states, ds, s, train_n, target_names, lmax, n_layers, size, rev,
                    targets=targets, groups=groups,
                )
                records[s]["groups"][size][rev] = asdict(cg)
                _save(records[s], paths[s])
            _release_states(states)
            ext.cleanup_states()
    return paths


def run_trajectory(
    out_path: Path,
    seed: int = C.PRIMARY_SEED,
    size: str = C.TRAJECTORY_SIZE,
    checkpoints: list[str] = C.TRAJECTORY_CHECKPOINTS,
    n_sentences: int | None = C.TRAJECTORY_SENTENCES,
    train_n: int = C.TRAIN_N_TRAJ,
    batch_size: int | None = None,
    target_names: list[str] = TARGETS_6,
) -> dict:
    out_path = Path(out_path)
    ds = load_sentences(limit=n_sentences)
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {
            "kind": "trajectory",
            "seed": seed,
            "size": size,
            "n_sentences": len(ds.sentences),
            "n_instances": len(ds),
            "targets": list(target_names),
            "checkpoints": {},
        }
    for rev in checkpoints:
        if rev in record["checkpoints"]:
            print(f"[skip] {size}@{rev} already done")
            continue
        cg = run_cell_group(size, rev, ds, seed, train_n, target_names, _batch_for(size, batch_size))
        record["checkpoints"][rev] = asdict(cg)
        _save(record, out_path)
    return record
