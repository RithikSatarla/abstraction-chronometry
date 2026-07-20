"""Artifact manifest: checksums + environment, as promised in the paper's
Reproducibility Statement."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PKGS = ["torch", "nnsight", "transformers", "datasets", "scikit-learn", "numpy", "scipy", "conllu"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _PKGS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    try:
        import torch

        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["cuda_device"] = None
    return out


def build_manifest(results_dir: Path, seeds: dict | None = None) -> dict:
    results_dir = Path(results_dir)
    files = {}
    for p in sorted(results_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(results_dir)).replace("\\", "/")] = {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
    manifest = {
        "software": software_versions(),
        "seeds": seeds or {"primary": 2026, "secondary": 4177},
        "files": files,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
