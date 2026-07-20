"""UD English-EWT loading and target construction (paper Sec. 7, Appendix B).

Targets are per-token (word-level) instances aligned to a single global token
order; the extraction layer produces one representation per instance (first
subword). Two grouping keys are exposed for the split designs: sentence id
(sentence-heldout) and lowercased word form (type-heldout).
"""

from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from conllu import parse_incr

EWT_DEV_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_English-EWT/master/en_ewt-ud-dev.conllu"
)
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "ud" / "en_ewt-ud-dev.conllu"

# Six-target total ladder (paper Sec. 7): word_shape < UPOS < Number(x2) < deprel < tree_depth.
TARGETS_6 = ["word_shape", "upos", "number_withno", "number_only", "deprel", "tree_depth"]
LADDER_6 = [1, 2, 3, 3, 4, 5]

# Four-target defended partial order (avoids the ambiguous Number rung).
TARGETS_4 = ["word_shape", "upos", "deprel", "tree_depth"]
# indices into TARGETS_4: ws=0, upos=1, deprel=2, tree_depth=3
PAIRS_4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
ACTIVE_4 = [0, 1, 2, 3]


@dataclass
class Token:
    form: str
    upos: str
    deprel: str
    feats: dict
    head: int
    tid: int


@dataclass
class Instance:
    sent_idx: int
    form: str


@dataclass
class Dataset:
    sentences: list[list[Token]]
    instances: list[Instance] = field(default_factory=list)

    def __post_init__(self):
        if not self.instances:
            self.instances = [
                Instance(si, tok.form)
                for si, sent in enumerate(self.sentences)
                for tok in sent
            ]

    def __len__(self):
        return len(self.instances)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def ensure_ewt(cache: Path = DEFAULT_CACHE) -> Path:
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(EWT_DEV_URL, context=ctx) as resp:  # noqa: S310
        cache.write_bytes(resp.read())
    return cache


def load_sentences(path: Path | None = None, limit: int | None = None) -> Dataset:
    path = Path(path) if path else ensure_ewt()
    sentences: list[list[Token]] = []
    with open(path, encoding="utf-8") as fh:
        for tl in parse_incr(fh):
            toks: list[Token] = []
            for t in tl:
                tid = t["id"]
                if not isinstance(tid, int):  # skip multiword ranges / empty nodes
                    continue
                feats = t["feats"] or {}
                head = t["head"] if isinstance(t["head"], int) else 0
                toks.append(
                    Token(
                        form=t["form"],
                        upos=t["upos"] or "_",
                        deprel=(t["deprel"] or "_"),
                        feats=dict(feats),
                        head=head,
                        tid=tid,
                    )
                )
            if toks:
                sentences.append(toks)
            if limit and len(sentences) >= limit:
                break
    return Dataset(sentences)


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #


def word_shape(form: str) -> str:
    """Collapsed orthographic pattern: X/x/d/other, consecutive duplicates merged."""
    out = []
    for ch in form:
        if ch.isupper():
            c = "X"
        elif ch.islower():
            c = "x"
        elif ch.isdigit():
            c = "d"
        else:
            c = ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def _tree_depths(sent: list[Token]) -> dict[int, int]:
    by_id = {t.tid: t for t in sent}
    depth: dict[int, int] = {}

    def d(tid: int, seen: frozenset = frozenset()) -> int:
        if tid in depth:
            return depth[tid]
        tok = by_id.get(tid)
        if tok is None or tok.head == 0 or tok.head not in by_id or tid in seen:
            depth[tid] = 0
            return 0
        depth[tid] = 1 + d(tok.head, seen | {tid})
        return depth[tid]

    for t in sent:
        d(t.tid)
    return depth


@dataclass
class TargetArray:
    kind: str  # "categorical" | "continuous"
    labels: np.ndarray  # object (categorical) or float (continuous)
    mask: np.ndarray  # bool: instance is an included target instance


def build_targets(ds: Dataset, names: list[str] = TARGETS_6) -> dict[str, TargetArray]:
    n = len(ds)
    out: dict[str, TargetArray] = {}
    # precompute tree depths per sentence
    depths_per_sent = [_tree_depths(s) for s in ds.sentences]

    def alloc_cat():
        return np.empty(n, dtype=object), np.zeros(n, dtype=bool)

    buffers: dict[str, tuple] = {}
    for name in names:
        if name == "tree_depth":
            buffers[name] = (np.zeros(n, dtype=float), np.zeros(n, dtype=bool))
        else:
            buffers[name] = alloc_cat()

    i = 0
    for si, sent in enumerate(ds.sentences):
        for tok in sent:
            for name in names:
                labels, mask = buffers[name]
                if name == "word_shape":
                    labels[i] = word_shape(tok.form)
                    mask[i] = True
                elif name == "upos":
                    labels[i] = tok.upos
                    mask[i] = True
                elif name == "deprel":
                    labels[i] = tok.deprel.split(":")[0]
                    mask[i] = True
                elif name == "number_withno":
                    labels[i] = tok.feats.get("Number", "NoNumber")
                    mask[i] = True
                elif name == "number_only":
                    if "Number" in tok.feats:
                        labels[i] = tok.feats["Number"]
                        mask[i] = True
                elif name == "tree_depth":
                    labels[i] = float(depths_per_sent[si][tok.tid])
                    mask[i] = True
            i += 1

    for name in names:
        labels, mask = buffers[name]
        kind = "continuous" if name == "tree_depth" else "categorical"
        out[name] = TargetArray(kind=kind, labels=labels, mask=mask)
    return out


def group_keys(ds: Dataset) -> dict[str, np.ndarray]:
    """Grouping keys for the two split designs."""
    sent_group = np.array([inst.sent_idx for inst in ds.instances])
    type_group = np.array([inst.form.lower() for inst in ds.instances], dtype=object)
    return {"sentence": sent_group, "type": type_group}
