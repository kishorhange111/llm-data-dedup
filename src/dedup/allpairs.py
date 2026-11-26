"""Exact all-pairs Jaccard similarity join by prefix filtering (Chaudhuri et al. 2006,
Bayardo et al. 2007). This is the ground truth that MinHash-LSH is measured against.

Fix a global order on shingles, rarest first. If ``J(x, y) ≥ t`` then
``|x ∩ y| ≥ t·|x ∪ y| ≥ α = max(⌈t|x|⌉, ⌈t|y|⌉)``. Any two sets that share α elements
must share one among the first ``|x| − α + 1`` elements of x and the first ``|y| − α + 1``
of y. So it is enough to index each set's **prefix** of ``|x| − ⌈t|x|⌉ + 1`` elements,
pair up sets whose prefixes meet, drop pairs whose sizes differ by more than a factor of
t, and verify the rest exactly. Nothing above the threshold can be missed. Rare-first
ordering keeps each prefix's posting lists short.
"""

from __future__ import annotations

import math

import numpy as np

from .hashing import jaccard


def prefix_len(n: int, t: float) -> int:
    return n - math.ceil(t * n - 1e-9) + 1


def similar_pairs(sets: list[np.ndarray], t: float, max_group: int = 20000):
    """Returns (codes i·N + j with i < j, jaccard values, number of candidates verified)."""
    N = len(sets)
    sizes = np.array([len(s) for s in sets])
    flat = np.concatenate([s for s in sets if len(s)]) if sizes.sum() else np.empty(0, np.uint64)
    _, inv, df = np.unique(flat, return_inverse=True, return_counts=True)
    rank = np.empty(len(df), np.int64)
    rank[np.lexsort((np.arange(len(df)), df))] = np.arange(len(df))  # rarest first
    toks, docs = [], []
    off = 0
    for d, n in enumerate(sizes):
        if n == 0:
            continue
        r = np.sort(rank[inv[off : off + n]])
        p = prefix_len(int(n), t)
        toks.append(r[:p])
        docs.append(np.full(p, d, np.int64))
        off += n
    toks, docs = np.concatenate(toks), np.concatenate(docs)
    order = np.lexsort((docs, toks))
    toks, docs = toks[order], docs[order]
    starts = np.flatnonzero(np.r_[True, toks[1:] != toks[:-1]])
    lens = np.diff(np.r_[starts, len(toks)])
    codes = []
    for s, g in zip(starts[lens > 1], lens[lens > 1], strict=True):
        if g > max_group:
            raise ValueError(f"posting list of {g} sets; deduplicate exact copies first")
        m = docs[s : s + g]
        i, j = np.triu_indices(g, 1)
        codes.append(m[i] * N + m[j])
    if not codes:
        return np.empty(0, np.int64), np.empty(0), 0
    cand = np.unique(np.concatenate(codes))
    a, b = cand // N, cand % N
    sa, sb = sizes[a], sizes[b]
    keep = np.minimum(sa, sb) >= t * np.maximum(sa, sb) - 1e-9  # size filter
    cand = cand[keep]
    J = np.array([jaccard(sets[i], sets[j]) for i, j in zip(cand // N, cand % N, strict=True)])
    hit = J >= t - 1e-12
    return cand[hit], J[hit], len(cand)
