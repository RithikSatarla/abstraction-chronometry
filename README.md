# Abstraction Chronometry — Reproducibility Artifact

Code and results for the BlackboxNLP 2026 reproducibility-track submission
*"Abstraction Chronometry: Finite-Protocol Invariance for Layerwise Probing"*.

This artifact audits the "BERT rediscovers the classical NLP pipeline" claim
family on the Pythia model suite, using exact finite-permutation nulls and a
two-part learned-origin evidence criterion (final-checkpoint significance
*and* a rejecting contrast against initialization, under the same joint
permutation orbit).

## What's here

- `src/abstraction_chronometry/` — the statistics core (depth functionals,
  exact Mahonian/joint-permutation nulls, the learned-origin criterion) plus
  the data, extraction, and probing layers.
- `tests/` — unit tests for the statistics core, independent of any model run,
  including an explicit invariant test that `Gamma == Final_max - Init_max`
  always holds.
- `results/` — the real experimental outputs: the primary five-size scale
  sweep (seed 2026), the Pythia-410M training trajectory (6 checkpoints, full
  2001-sentence UD English-EWT dev set), a five-seed robustness sweep
  (`sweep_{2026,4177,5023,6180,8191}.json`), and the causal-patching experiment
  (`causal_main.json`), plus a checksummed manifest (`manifest.json`).
- `src/.../causal.py` — the causal-chronometry experiment: NNsight activation
  patching on subject-verb agreement minimal pairs, localizing where subject
  number is *causally* used vs. where it is *decodable*.
- `paper/main.tex` — the paper source, with all reported numbers regenerated
  from `results/`.

## Reproducing

```
uv sync
uv run ac-run all --results results
```

The second architecture family (OLMo-2-1B) is a separate command, since it is a
~11GB download and is not part of the Pythia grid:

```
uv run ac-run olmo --out results/sweep_olmo.json --seed 2026
uv run python analyze_olmo.py results/sweep_olmo.json
```

OLMo-2 is used rather than OLMo-1 because it publishes the *matched* random
initialization the learned-origin criterion requires
(`stage1-step0-tokens0B`); OLMo-1-hf's earliest public branch is
`step1000-tokens4B`, i.e. already trained on 4B tokens, so it cannot serve
as `M_0`.

Hidden states are extracted via [NNsight](https://nnsight.net) module-level
interventions, verified to exact (zero max absolute difference) numerical
parity against HuggingFace's `output_hidden_states` reference path before any
probing was run (`tests/test_extract_parity.py`, and Appendix B of the paper).
Parity is checked per model family, since GPTNeoX and OLMo-2 name their
modules differently; run the OLMo half with `AC_TEST_OLMO=1 uv run pytest`.

## Environment

Python 3.12, PyTorch 2.6.0+cu124, NNsight 0.7.0, Transformers 5.14.1,
NVIDIA RTX 3060 Ti (8GB). Full pinned versions in `pyproject.toml` /
`uv.lock`, and recorded per-run in `results/manifest.json`.
