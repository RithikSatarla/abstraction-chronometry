"""NNsight hidden-state extraction (paper Sec. 7 / Appendix B).

The NDIF-award requirement is that model internals are accessed through NNsight.
We capture the residual stream via NNsight module envoys (the embedding, each
`layers[i]`, and the final norm) and assemble them in the order that exactly
reproduces HuggingFace `output_hidden_states` (verified to 0.0 max abs diff in
tests/test_extract_parity.py): [embed, block_0 .. block_{n-2}, final_norm].

Two decoder-only families are supported, differing only in module names:
GPTNeoX (Pythia) and OLMo-2. Both build `output_hidden_states` the same way --
embeddings, then each block output, with the *last* entry passed through the
final norm -- so one assembly rule serves both; `_MODULE_PATHS` supplies the
names and the parity test checks the assumption per family.

Per-token instances are aligned to the first subword of each word. Right padding
plus a passed attention mask keeps real-token representations exact under the
causal model.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import torch

from .data import Dataset

MODEL_TEMPLATE = "EleutherAI/pythia-{size}"

# config.model_type -> (embedding, block list, final norm) module paths.
_MODULE_PATHS = {
    "gpt_neox": ("gpt_neox.embed_in", "gpt_neox.layers", "gpt_neox.final_layer_norm"),
    "olmo2": ("model.embed_tokens", "model.layers", "model.norm"),
}


def _resolve(root, path: str):
    """Walk a dotted module path on an NNsight envoy (or plain module)."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def enable_system_certs() -> None:
    """Route Python SSL through the OS trust store (needed behind TLS proxies)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass


def _as_hidden(x):
    x = getattr(x, "value", x)
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x


class Extractor:
    """Loads one (model size, checkpoint revision) and extracts per-instance,
    per-layer representations aligned to `Dataset.instances` order."""

    def __init__(
        self,
        model_size: str,
        revision: str = "main",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        enable_system_certs()
        from nnsight import LanguageModel

        # A '/' means an explicit HF repo id (e.g. the OLMo-2 second family);
        # a bare token is a Pythia size shorthand.
        self.name = model_size if "/" in model_size else MODEL_TEMPLATE.format(size=model_size)
        self.revision = revision
        # Staged load. Dispatching straight to CUDA makes accelerate hold a large
        # transient CPU allocation while it plans/copies, which the OS kills
        # outright (exit 5, no traceback) on an 8GB box with a released-in-fp32
        # checkpoint. Loading to CPU first streams shard-by-shard through
        # reclaimable page cache, then .to(cuda) moves an already-fp16 model.
        want_cuda = device == "cuda" and torch.cuda.is_available()
        self.lm = LanguageModel(
            self.name,
            revision=revision,
            dtype=dtype,
            dispatch=True,
            device_map="cpu" if want_cuda else device,
            low_cpu_mem_usage=True,  # stream weights instead of materializing full fp32 on CPU RAM first
        )
        if want_cuda:
            self._hf_model().to("cuda")
            import gc

            gc.collect()
        self.tok = self.lm.tokenizer
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.family = str(self.lm.config.model_type)
        if self.family not in _MODULE_PATHS:
            raise ValueError(
                f"{self.name}: unsupported model_type {self.family!r}; "
                f"add its module paths to _MODULE_PATHS and extend the parity test."
            )
        self.n_layers = int(self.lm.config.num_hidden_layers)
        self.hidden = int(self.lm.config.hidden_size)
        self.n_states = self.n_layers + 1  # embed + blocks + final_ln
        self.lmax = self.n_layers  # index of the deepest state

    def _hf_model(self):
        """The underlying HuggingFace module behind the NNsight envoy."""
        for attr in ("_model", "model"):
            m = getattr(self.lm, attr, None)
            if isinstance(m, torch.nn.Module):
                return m
        raise AttributeError("could not locate the underlying HF model on LanguageModel")

    @property
    def device(self) -> torch.device:
        return next(self._hf_model().parameters()).device

    # -- alignment ---------------------------------------------------------- #

    def align_sentence(self, words: list[str]) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        first_pos: list[int] = []
        for wi, w in enumerate(words):
            piece = (" " + w) if wi > 0 else w
            sub = self.tok.encode(piece, add_special_tokens=False)
            if not sub:  # e.g. a form that tokenizes to nothing
                sub = self.tok.encode(" " + w, add_special_tokens=False) or [
                    self.tok.eos_token_id
                ]
            first_pos.append(len(ids))
            ids.extend(sub)
        return ids, first_pos

    # -- extraction --------------------------------------------------------- #

    @torch.no_grad()
    def trace_states(self, input_ids, attn) -> list:
        """Per-layer residual states for one batch, ordered to match HuggingFace
        `output_hidden_states`: [embed, block_0 .. block_{n-2}, final_norm].

        Both supported families append the embedding, then each block output,
        then replace the tail with the normed final block, so this single
        assembly is family-independent given the right module names.
        tests/test_extract_parity.py checks that per family.
        """
        emb_path, blocks_path, norm_path = _MODULE_PATHS[self.family]
        layer_saves = []
        with self.lm.trace(input_ids, attention_mask=attn):
            embed = _resolve(self.lm, emb_path).output.save()
            blocks = _resolve(self.lm, blocks_path)
            # explicit loop: a comprehension does not bind saves inside a trace
            for i in range(self.n_layers):
                layer_saves.append(blocks[i].output.save())
            final_ln = _resolve(self.lm, norm_path).output.save()

        assembled = [_as_hidden(embed)]
        for i in range(self.n_layers - 1):
            assembled.append(_as_hidden(layer_saves[i]))
        assembled.append(_as_hidden(final_ln))
        return assembled

    @torch.no_grad()
    def extract(
        self, ds: Dataset, batch_size: int = 16, max_len: int = 128, verbose: bool = True
    ) -> list[np.ndarray]:
        n_inst = len(ds)
        # Disk-backed fp16 memmaps (one per layer) rather than a hard RAM alloc:
        # on an 8GB-RAM machine the 1.4b states array (~1.4GB) plus the torch base
        # tips into OOM. Memmap pages are reclaimable page cache, so extraction can't
        # OOM, and probing reads one layer-slice at a time. sklearn upcasts per-fit,
        # so fp16 storage does not affect probe precision.
        import tempfile

        self.states_dir = tempfile.mkdtemp(prefix="ac_states_")
        states = [
            np.memmap(
                os.path.join(self.states_dir, f"L{i}.dat"),
                dtype=np.float16, mode="w+", shape=(n_inst, self.hidden),
            )
            for i in range(self.n_states)
        ]

        aligns = [self.align_sentence([t.form for t in s]) for s in ds.sentences]
        offsets = np.zeros(len(ds.sentences) + 1, dtype=int)
        for si, sent in enumerate(ds.sentences):
            offsets[si + 1] = offsets[si] + len(sent)

        n_sent = len(ds.sentences)
        for b0 in range(0, n_sent, batch_size):
            batch = list(range(b0, min(b0 + batch_size, n_sent)))
            ids_list = [aligns[si][0][:max_len] for si in batch]
            maxT = max(len(x) for x in ids_list)
            input_ids = torch.full((len(batch), maxT), self.tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxT), dtype=torch.long)
            for r, ids in enumerate(ids_list):
                input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, : len(ids)] = 1

            assembled = [
                a.float().cpu().numpy() for a in self.trace_states(input_ids, attn)
            ]  # (B, T, d) each

            for r, si in enumerate(batch):
                first_pos = aligns[si][1]
                base = offsets[si]
                for wi in range(len(ds.sentences[si])):
                    pos = min(first_pos[wi], maxT - 1)
                    for s in range(self.n_states):
                        states[s][base + wi] = assembled[s][r, pos]

            if verbose and (b0 // batch_size) % 20 == 0:
                print(f"  [{self.name}@{self.revision}] {min(b0 + batch_size, n_sent)}/{n_sent}")

        for m in states:
            m.flush()
        return states

    def cleanup_states(self) -> None:
        """Delete the on-disk memmap temp dir. Call after probing is done with the
        states returned by extract()."""
        d = getattr(self, "states_dir", None)
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            self.states_dir = None

    def free(self) -> None:
        import gc

        try:
            del self.lm
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
