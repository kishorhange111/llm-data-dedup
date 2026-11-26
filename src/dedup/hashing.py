"""Tokens, n-gram shingles and a 64-bit mixing hash, all vectorised with NumPy.

A document becomes the **set** of hashes of its token n-grams ("shingles"). Two documents'
Jaccard similarity is ``|A ∩ B| / |A ∪ B|`` over those sets. Code is tokenised into
identifiers, numbers and single punctuation characters, so reformatting whitespace does
not change the set but renaming a variable does.

``splitmix64`` is the finaliser of Steele et al.'s SplitMix generator: a bijection on 64-bit
integers with good avalanche. XOR-ing the input with a per-permutation seed before mixing
gives the hash family MinHash uses. NumPy's uint64 arithmetic wraps modulo 2^64, which is
what the mixer needs.
"""

from __future__ import annotations

import re

import numpy as np

TOKEN = re.compile(r"\w+|[^\w\s]")
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
MAX64 = np.uint64(2**64 - 1)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + _C1
        x = (x ^ (x >> np.uint64(30))) * _C2
        x = (x ^ (x >> np.uint64(27))) * _C3
    return x ^ (x >> np.uint64(31))


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


class Vocab:
    """Maps token strings to dense int ids (first-seen order)."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, tokens: list[str]) -> np.ndarray:
        ids = self.ids
        return np.fromiter((ids.setdefault(t, len(ids)) for t in tokens), np.int64, len(tokens))


def shingles(ids: np.ndarray, n: int = 5) -> np.ndarray:
    """Sorted unique uint64 hashes of the token n-grams of ``ids``; empty if len < n."""
    ids = np.asarray(ids, dtype=np.uint64)
    m = len(ids) - n + 1
    if m <= 0:
        return np.empty(0, np.uint64)
    tok = splitmix64(ids)
    h = np.zeros(m, np.uint64)
    for k in range(n):  # order-sensitive combine: h = mix(h) ^ tok
        h = splitmix64(h) ^ tok[k : k + m]
    return np.unique(splitmix64(h))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard of two sorted unique arrays."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)
