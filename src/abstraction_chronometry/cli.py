"""Command-line entry point (`ac-run`) reproducing Tables 1-3 and the manifest.

Examples:
  ac-run scale-sweep --out results/sweep_2026.json --seed 2026
  ac-run scale-sweep --out results/sweep_4177.json --seed 4177 --sizes 70m 160m 410m
  ac-run trajectory  --out results/traj_2026.json  --seed 2026
  ac-run tables      --sweep results/sweep_2026.json --traj results/traj_2026.json \
                     --seed2 results/sweep_4177.json
  ac-run manifest    --results results
  ac-run all         --results results        # full paper run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as C


def cmd_scale_sweep(args):
    from .run import run_scale_sweep

    # batch_size=None -> run.py picks a per-size default (shrinks for 1b/1.4b on 8GB cards)
    run_scale_sweep(
        Path(args.out),
        sizes=args.sizes,
        seed=args.seed,
        n_sentences=args.sentences,
        batch_size=args.batch,
    )
    print(f"wrote {args.out}")


def cmd_olmo(args):
    """Second architecture family: OLMo-2-1B, matched init vs final checkpoint."""
    from .run import run_scale_sweep

    run_scale_sweep(
        Path(args.out),
        sizes=[C.OLMO_MODEL],
        seed=args.seed,
        n_sentences=args.sentences,
        batch_size=args.batch,
        init_revision=C.OLMO_INIT_REVISION,
        final_revision=C.OLMO_FINAL_REVISION,
    )
    print(f"wrote {args.out}")


def cmd_trajectory(args):
    from .run import run_trajectory

    run_trajectory(Path(args.out), seed=args.seed, batch_size=args.batch)
    print(f"wrote {args.out}")


def cmd_multiseed(args):
    from .run import run_scale_sweep_multiseed

    seeds = args.seeds if args.seeds else C.SWEEP_SEEDS
    paths = run_scale_sweep_multiseed(
        Path(args.out_dir), seeds=seeds, sizes=args.sizes, batch_size=args.batch
    )
    print("wrote:", *[str(p) for p in paths.values()], sep="\n  ")


def cmd_causal(args):
    from .causal import run_causal

    run_causal(Path(args.out), sizes=args.sizes, revision=args.revision,
               max_pairs=args.pairs, seed=args.seed)
    print(f"wrote {args.out}")


def cmd_tables(args):
    from .tables import build_report

    report = build_report(Path(args.sweep), args.traj, args.seed2)
    print(report)
    if args.save:
        Path(args.save).write_text(report)
        print(f"\nsaved report to {args.save}")


def cmd_manifest(args):
    from .manifest import build_manifest

    m = build_manifest(Path(args.results))
    print(f"manifest: {len(m['files'])} files, python {m['software']['python']}")


def cmd_all(args):
    from .manifest import build_manifest
    from .run import run_scale_sweep, run_trajectory
    from .tables import build_report

    root = Path(args.results)
    sweep = root / "sweep_2026.json"
    sweep2 = root / "sweep_4177.json"
    traj = root / "traj_2026.json"
    run_scale_sweep(sweep, seed=C.PRIMARY_SEED)
    run_trajectory(traj, seed=C.PRIMARY_SEED)
    run_scale_sweep(sweep2, sizes=["70m", "160m", "410m"], seed=C.SECONDARY_SEED)
    report = build_report(sweep, str(traj), str(sweep2))
    (root / "report.txt").write_text(report)
    build_manifest(root)
    print(report)
    print(f"\nAll artifacts in {root}/")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ac-run", description="Abstraction Chronometry reproduction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ss = sub.add_parser("scale-sweep")
    ss.add_argument("--out", required=True)
    ss.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ss.add_argument("--sizes", nargs="+", default=C.SIZES)
    ss.add_argument("--sentences", type=int, default=C.SCALE_SWEEP_SENTENCES)
    ss.add_argument("--batch", type=int, default=None)
    ss.set_defaults(func=cmd_scale_sweep)

    ol = sub.add_parser("olmo", help="second architecture family (OLMo-2-1B)")
    ol.add_argument("--out", required=True)
    ol.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ol.add_argument("--sentences", type=int, default=C.SCALE_SWEEP_SENTENCES)
    ol.add_argument("--batch", type=int, default=None)
    ol.set_defaults(func=cmd_olmo)

    tr = sub.add_parser("trajectory")
    tr.add_argument("--out", required=True)
    tr.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    tr.add_argument("--batch", type=int, default=None)
    tr.set_defaults(func=cmd_trajectory)

    ms = sub.add_parser("multiseed")
    ms.add_argument("--out-dir", required=True)
    ms.add_argument("--seeds", nargs="+", type=int, default=None)
    ms.add_argument("--sizes", nargs="+", default=C.SIZES)
    ms.add_argument("--batch", type=int, default=None)
    ms.set_defaults(func=cmd_multiseed)

    ca = sub.add_parser("causal")
    ca.add_argument("--out", required=True)
    ca.add_argument("--sizes", nargs="+", default=C.SIZES)
    ca.add_argument("--revision", default=C.FINAL_REVISION)
    ca.add_argument("--pairs", type=int, default=240)
    ca.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ca.set_defaults(func=cmd_causal)

    tb = sub.add_parser("tables")
    tb.add_argument("--sweep", required=True)
    tb.add_argument("--traj", default=None)
    tb.add_argument("--seed2", default=None)
    tb.add_argument("--save", default=None)
    tb.set_defaults(func=cmd_tables)

    mf = sub.add_parser("manifest")
    mf.add_argument("--results", required=True)
    mf.set_defaults(func=cmd_manifest)

    al = sub.add_parser("all")
    al.add_argument("--results", default="results")
    al.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
