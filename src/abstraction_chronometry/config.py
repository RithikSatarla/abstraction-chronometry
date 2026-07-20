"""Grid, model, and checkpoint definitions (paper Sec. 7 / Appendix B)."""

from __future__ import annotations

SIZES = ["70m", "160m", "410m", "1b", "1.4b"]

INIT_REVISION = "step0"
FINAL_REVISION = "main"

# Second architecture family (Sec. 7). OLMo-2 is chosen over OLMo-1 because it
# publishes the *matched* random initialization the learned-origin criterion
# needs (OLMo-1-hf's earliest public branch is step1000-tokens4B, i.e. already
# trained on 4B tokens, which cannot serve as M_0).
OLMO_MODEL = "allenai/OLMo-2-0425-1B"
OLMO_INIT_REVISION = "stage1-step0-tokens0B"
OLMO_FINAL_REVISION = "main"

TRAJECTORY_SIZE = "410m"
TRAJECTORY_CHECKPOINTS = ["step0", "step1000", "step4000", "step16000", "step64000", "main"]

PRIMARY_SEED = 2026
SECONDARY_SEED = 4177  # Appendix C seed-robustness check
# Full seed set for the multi-seed robustness sweep (governs split + subsample + SVD).
SWEEP_SEEDS = [2026, 4177, 5023, 6180, 8191]

SCALE_SWEEP_SENTENCES = 1000
TRAJECTORY_SENTENCES = None  # all (2001)
TRAIN_N_SWEEP = 4000
TRAIN_N_TRAJ = 5000

# 12-protocol grid = 3 depth functionals x 2 split designs x 2 probe families.
FUNCTIONALS = ["COM", "Pk", "On"]
SPLITS = ["sentence", "type"]
PROBES = ["ridge", "lowrank"]

_SPLIT_TAG = {"sentence": "sent", "type": "type"}
_PROBE_TAG = {"ridge": "ridge", "lowrank": "low"}


def protocol_cells() -> list[tuple[str, str, str]]:
    return [(f, s, p) for f in FUNCTIONALS for s in SPLITS for p in PROBES]


def protocol_name(functional: str, split: str, probe: str) -> str:
    return f"{functional}-{_SPLIT_TAG[split]}-{_PROBE_TAG[probe]}"
