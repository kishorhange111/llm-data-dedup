"""A code corpus that needs no download: the ``.py`` files of installed Python environments.

Site-packages directories are full of real near-duplicates: vendored copies of libraries
(``pip/_vendor``, ``setuptools/_vendor``), generated files, and files copied between
projects. That is the kind of redundancy code-pretraining pipelines remove.
"""

from __future__ import annotations

import glob
import hashlib
import os
import site
import sysconfig
from dataclasses import dataclass


@dataclass
class Doc:
    root: str
    path: str  # relative to root
    text: str


def default_roots() -> list[str]:
    roots = [sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"]]
    roots += [site.getusersitepackages()] if os.path.isdir(site.getusersitepackages()) else []
    seen, out = set(), []
    for r in roots:  # purelib is inside stdlib on some layouts: keep each file once
        r = os.path.realpath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def load(roots: list[str], max_bytes: int = 1 << 20) -> list[Doc]:
    docs, seen = [], set()
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
            real = os.path.realpath(f)
            if real in seen or os.path.getsize(f) > max_bytes:
                continue
            seen.add(real)
            with open(f, encoding="utf-8", errors="replace") as fh:
                docs.append(Doc(os.path.basename(root), os.path.relpath(f, root), fh.read()))
    return docs


def exact_dedup(docs: list[Doc]) -> tuple[list[Doc], int]:
    """Keep the first copy of each byte-identical text; returns (kept, number removed)."""
    seen, kept = set(), []
    for d in docs:
        h = hashlib.sha256(d.text.encode("utf-8", "surrogatepass")).digest()
        if h not in seen:
            seen.add(h)
            kept.append(d)
    return kept, len(docs) - len(kept)
