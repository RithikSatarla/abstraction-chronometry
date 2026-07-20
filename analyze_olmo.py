"""Second-architecture-family analysis (OLMo-2-1B), Sec. 7 / Appendix F.

Applies the *same* learned-origin criterion used for Table 1 to the OLMo-2
sweep, and re-checks the identity Gamma == final_max - init_max that the
criterion enforces by construction.

    uv run python analyze_olmo.py results/sweep_olmo.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from abstraction_chronometry import config as C
from abstraction_chronometry.analysis import evaluate_partial, evaluate_size
from abstraction_chronometry.data import TARGETS_6


def main(path: str = "results/sweep_olmo.json") -> None:
    rec = json.loads(Path(path).read_text())
    init_rev = rec.get("init_revision", C.OLMO_INIT_REVISION)
    final_rev = rec.get("final_revision", C.OLMO_FINAL_REVISION)

    print(f"file        : {path}")
    print(f"seed        : {rec['seed']}   sentences={rec['n_sentences']} "
          f"instances={rec['n_instances']}")
    print(f"revisions   : init={init_rev}  final={final_rev}\n")

    for model, g in rec["groups"].items():
        if init_rev not in g or final_rev not in g:
            print(f"{model}: INCOMPLETE (have {sorted(g)})")
            continue
        fcg, icg = g[final_rev], g[init_rev]
        r = evaluate_size(fcg, icg, TARGETS_6)
        p = evaluate_partial(fcg, icg)

        print(f"=== {model}  ({fcg['n_layers']} layers, L_max={fcg['lmax']}) ===")
        print(f"  defined cells init/final/matched : "
              f"{r['def_init']}/{r['def_final']}/{r['def_matched']}")
        print(f"  selected protocol               : {r['sel']}")
        print(f"  H_max final                     : {r['final_max']:.3f}  p={r['pf']:.3f}")
        print(f"  H_max init                      : {r['init_max']:.3f}  p={r['pi']:.3f}")
        print(f"  Gamma (final - init)            : {r['gamma']:.3f}  p={r['p_gamma']:.3f}")
        print(f"  orbit                           : {r['orbit']}")
        print(f"  final rejects / contrast rejects: "
              f"{r['final_rejects']} / {r['contrast_rejects']}")
        print(f"  LEARNED-ORIGIN CRITERION        : "
              f"{'PASSES' if r['passes'] else 'FAILS'}")
        print(f"  four-target partial order       : HR_max={p['HR_max']:.3f} "
              f"p={p['pf']:.3f}  HR_init_max={p['HR_init_max']:.3f}  "
              f"Gamma={p['gamma']:.3f} p={p['p_gamma']:.3f}  "
              f"{'PASSES' if p['passes'] else 'FAILS'}")
        assert abs(p["gamma"] - (p["HR_max"] - p["HR_init_max"])) < 1e-12, "partial Gamma identity"

        # the identity that the .826-class bug violated
        lhs, rhs = r["gamma"], r["final_max"] - r["init_max"]
        ok = abs(lhs - rhs) < 1e-12
        print(f"  CHECK Gamma == final-init       : {lhs:.6f} == {rhs:.6f}  "
              f"{'OK' if ok else 'MISMATCH'}\n")
        if not ok:
            raise SystemExit("Gamma identity violated")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
