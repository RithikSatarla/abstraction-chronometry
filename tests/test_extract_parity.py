"""NNsight-vs-HuggingFace parity for the assembled residual stream.

`Extractor.trace_states` claims to reproduce HF `output_hidden_states` exactly.
That claim is what licenses reading our depth curves as layer depths, so it is
checked per model family: GPTNeoX (Pythia) always, OLMo-2 when the weights are
available locally (it is a ~2GB download, so it is opt-in via
AC_TEST_OLMO=1 rather than run by default).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from abstraction_chronometry.extract import Extractor, enable_system_certs

SENTENCES = ["The dog beside the manager is asleep .", "Colorless green ideas sleep ."]


def _parity(model: str, revision: str = "main") -> float:
    """Max abs diff between our assembled states and HF output_hidden_states."""
    enable_system_certs()
    ext = Extractor(model, revision=revision, device="cpu")

    ids_list = [ext.tok.encode(s, add_special_tokens=False) for s in SENTENCES]
    maxT = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), maxT), ext.tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), maxT), dtype=torch.long)
    for r, ids in enumerate(ids_list):
        input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[r, : len(ids)] = 1

    ours = [a.float().cpu().numpy() for a in ext.trace_states(input_ids, attn)]

    # reference: the underlying HF module, same weights, same inputs
    hf = ext.lm._model if hasattr(ext.lm, "_model") else ext.lm.model
    with torch.no_grad():
        out = hf(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    ref = [h.float().cpu().numpy() for h in out.hidden_states]

    assert len(ours) == len(ref) == ext.n_states, (
        f"state count {len(ours)} vs HF {len(ref)} vs expected {ext.n_states}"
    )
    # compare only real (non-pad) positions
    mask = attn.numpy().astype(bool)
    diffs = [np.abs(o[mask] - r[mask]).max() for o, r in zip(ours, ref)]
    ext.free()
    return float(max(diffs))


def test_parity_gpt_neox():
    """Pythia-70m: assembled states must equal HF hidden states exactly."""
    assert _parity("70m") == 0.0


@pytest.mark.skipif(
    os.environ.get("AC_TEST_OLMO") != "1",
    reason="set AC_TEST_OLMO=1 to run the OLMo-2 parity check (~2GB download)",
)
def test_parity_olmo2():
    """OLMo-2-1B: the second family must use the same assembly rule."""
    from abstraction_chronometry import config as C

    assert _parity(C.OLMO_MODEL, revision=C.OLMO_FINAL_REVISION) == 0.0
