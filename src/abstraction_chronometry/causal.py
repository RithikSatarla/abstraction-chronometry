"""Causal chronometry: where is subject number *causally* carried, vs where it is
*decodable*? (NNsight activation patching.)

The probing audit measures a decodability depth for each target. Here we add a
causal-depth protocol for one target (number), via activation patching on
subject-verb agreement minimal pairs (Linzen-style). For each layer we patch the
subject-token residual stream from the plural run into the singular run and
measure how much of the agreement logit gap is restored; the resulting
causal-effect curve has a center-of-mass depth directly comparable to the
probe's decodability depth for number.

This uses NNsight interventions (reading and writing module activations inside a
trace), not just hidden-state extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .extract import MODEL_TEMPLATE, enable_system_certs

# Regular noun singular/plural pairs; filtered at runtime to those whose single
# leading-space token count matches (so minimal pairs stay position-aligned).
_NOUNS = [
    ("boy", "boys"), ("girl", "girls"), ("dog", "dogs"), ("cat", "cats"),
    ("car", "cars"), ("book", "books"), ("author", "authors"), ("player", "players"),
    ("student", "students"), ("teacher", "teachers"), ("farmer", "farmers"),
    ("doctor", "doctors"), ("singer", "singers"), ("writer", "writers"),
    ("king", "kings"), ("actor", "actors"), ("worker", "workers"), ("runner", "runners"),
    ("driver", "drivers"), ("painter", "painters"), ("dancer", "dancers"), ("banker", "bankers"),
]
_ATTRACTORS = ["manager", "senator", "officer", "pilot", "guard", "captain", "lawyer", "reporter"]
_PREPS = ["near", "behind", "beside"]
_VERBS = [("is", "are"), ("was", "were")]  # (singular form, plural form)


@dataclass
class Pair:
    plural_ids: list[int]
    singular_ids: list[int]
    subj_pos: int
    last_pos: int
    sing_verb_id: int
    plur_verb_id: int


def _single_tok(tok, word: str) -> int | None:
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def build_pairs(tok, max_pairs: int = 240, seed: int = 2026) -> list[Pair]:
    """Minimal pairs: 'The <subj> <prep> the <attr>' -> next token is a copula.
    Only subject number differs between plural_ids and singular_ids."""
    rng = np.random.default_rng(seed)
    # keep noun pairs whose sing/plural are both single leading-space tokens
    noun_pairs = []
    for s, p in _NOUNS:
        si, pi = _single_tok(tok, s), _single_tok(tok, p)
        if si is not None and pi is not None:
            noun_pairs.append((s, p, si, pi))
    verbs = [(sv, pv, _single_tok(tok, sv), _single_tok(tok, pv)) for sv, pv in _VERBS]
    verbs = [v for v in verbs if v[2] is not None and v[3] is not None]

    pairs: list[Pair] = []
    for s, p, si_id, pi_id in noun_pairs:
        for attr in _ATTRACTORS:
            prep = _PREPS[rng.integers(len(_PREPS))]
            sv, pv, sv_id, pv_id = verbs[rng.integers(len(verbs))]
            prefix_s = f"The {s} {prep} the {attr}"
            prefix_p = f"The {p} {prep} the {attr}"
            ids_s = tok.encode(prefix_s, add_special_tokens=False)
            ids_p = tok.encode(prefix_p, add_special_tokens=False)
            if len(ids_s) != len(ids_p):
                continue  # keep aligned
            # subject is the 2nd word -> its first subword is at index 1 (after "The")
            pairs.append(Pair(
                plural_ids=ids_p, singular_ids=ids_s, subj_pos=1,
                last_pos=len(ids_s) - 1, sing_verb_id=sv_id, plur_verb_id=pv_id,
            ))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _as_hidden_write(module_output, pos, value):
    """Assign `value` into the residual stream at token `pos`. Handles the
    transformers-5.x GPTNeoX layer whose output is a bare tensor."""
    module_output[:, pos, :] = value


class CausalPatcher:
    def __init__(self, model_size: str, revision: str = "main",
                 dtype: torch.dtype = torch.float16, device: str = "cuda"):
        enable_system_certs()
        from nnsight import LanguageModel

        self.name = MODEL_TEMPLATE.format(size=model_size)
        self.lm = LanguageModel(self.name, revision=revision, dtype=dtype,
                                dispatch=True, device_map=device, low_cpu_mem_usage=True)
        self.tok = self.lm.tokenizer
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.n_layers = int(self.lm.config.num_hidden_layers)

    def _agreement_gap(self, logits_last, sing_id, plur_id) -> float:
        # Delta = logit(plural verb) - logit(singular verb); >0 favours plural.
        return float(logits_last[plur_id] - logits_last[sing_id])

    @torch.no_grad()
    def causal_curve(self, pairs: list[Pair], verbose: bool = True) -> np.ndarray:
        """Per-layer fraction of the agreement gap restored by patching the
        subject-token residual from the plural run into the singular run.
        Averaged over pairs. Returns array over layers (0..n_layers-1)."""
        effects = np.zeros(self.n_layers, dtype=float)
        counts = 0
        for k, pr in enumerate(pairs):
            sing = torch.tensor([pr.singular_ids])
            plur = torch.tensor([pr.plural_ids])

            def val(x):
                x = getattr(x, "value", x)
                return x[0] if isinstance(x, (tuple, list)) else x

            # clean (plural) subject activations at every layer + both baselines.
            # NOTE: explicit loop (not comprehension) inside the trace -- nnsight
            # does not bind list-comprehension results.
            plur_acts = []
            with self.lm.trace(plur):
                for i in range(self.n_layers):
                    plur_acts.append(self.lm.gpt_neox.layers[i].output.save())
                plur_logits = self.lm.output.logits[0, pr.last_pos].save()
            with self.lm.trace(sing):
                sing_logits = self.lm.output.logits[0, pr.last_pos].save()

            d_plur = self._agreement_gap(val(plur_logits), pr.sing_verb_id, pr.plur_verb_id)
            d_sing = self._agreement_gap(val(sing_logits), pr.sing_verb_id, pr.plur_verb_id)
            denom = d_plur - d_sing
            if abs(denom) < 1e-4:
                continue

            for L in range(self.n_layers):
                clean_subj = val(plur_acts[L])[:, pr.subj_pos, :]
                with self.lm.trace(sing):
                    self.lm.gpt_neox.layers[L].output[:, pr.subj_pos, :] = clean_subj
                    patched = self.lm.output.logits[0, pr.last_pos].save()
                d_patched = self._agreement_gap(val(patched), pr.sing_verb_id, pr.plur_verb_id)
                effects[L] += (d_patched - d_sing) / denom
            counts += 1
            if verbose and k % 50 == 0:
                print(f"  [causal {self.name}] pair {k}/{len(pairs)}")
        if counts == 0:
            return effects
        return effects / counts

    def free(self):
        import gc
        try:
            del self.lm
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def causal_depth(curve: np.ndarray) -> float:
    """Center-of-mass depth of the (nonnegative part of the) causal-effect curve,
    on the same 0..n_layers-1 block scale, normalized to [0,1] by (n_layers-1)."""
    c = np.clip(np.asarray(curve, dtype=float), 0.0, None)
    if c.sum() <= 0:
        return float("nan")
    layers = np.arange(len(c))
    com = (layers * c).sum() / c.sum()
    return com / (len(c) - 1)


def run_causal(out_path, sizes: list[str], revision: str = "main",
               max_pairs: int = 240, seed: int = 2026) -> dict:
    """Run the causal-patching experiment across sizes, saving incrementally.
    Resilient per size (skips a size that fails to load, e.g. OOM on 8GB RAM)."""
    out_path = Path(out_path)
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {"kind": "causal", "revision": revision, "max_pairs": max_pairs,
                  "seed": seed, "sizes": {}}
    for size in sizes:
        if size in record["sizes"]:
            print(f"[skip causal] {size} already done", flush=True)
            continue
        try:
            cp = CausalPatcher(size, revision=revision)
            pairs = build_pairs(cp.tok, max_pairs=max_pairs, seed=seed)
            print(f"[causal {size}] {len(pairs)} pairs, {cp.n_layers} layers", flush=True)
            curve = cp.causal_curve(pairs)
            n_layers = cp.n_layers
            cp.free()
            record["sizes"][size] = {
                "n_layers": n_layers, "n_pairs": len(pairs),
                "curve": [float(x) for x in curve], "causal_depth": causal_depth(curve),
                "peak_layer": int(np.argmax(curve)),
            }
            out_path.write_text(json.dumps(record, indent=2))
            print(f"[causal {size}] depth={causal_depth(curve):.3f} "
                  f"peak_layer={int(np.argmax(curve))}/{n_layers}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[causal {size}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return record
