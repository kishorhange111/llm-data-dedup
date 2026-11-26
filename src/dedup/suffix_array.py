"""Exact-substring deduplication with a suffix array (Lee et al. 2022, "ExactSubstr").

MinHash finds documents that are mostly the same. It misses a 60-line license header or
a vendored helper pasted into an otherwise original file. ExactSubstr finds every token
span of length ≥ k that occurs more than once anywhere in the corpus.

**Suffix array by prefix doubling** (Manber & Myers). ``rank_h[i]`` orders suffixes by
their first h tokens. Sorting by the pair ``(rank_h[i], rank_h[i + h])`` gives
``rank_2h``, so ⌈log₂ L⌉ rounds suffice, where L is the longest repeat. Each round is one
NumPy sort.

**LCP without a Python loop.** Kasai's algorithm is O(n) but sequential. Since the
doubling rounds already stored ``rank_h`` for h = 2^j, the longest common prefix of two
suffixes a, b follows by binary lifting. For h from largest to smallest, if
``rank_h[a + ℓ] == rank_h[b + ℓ]`` the next h tokens match, so ℓ += h. That runs
vectorised over all adjacent pairs of the suffix array at once and is exact.

**Marking.** Adjacent suffixes with LCP ≥ k share a k-token span. In each run of such
suffixes this keeps the occurrence that comes first in the corpus and marks
``[p, p + k)`` for the others. Every document ends in a unique separator, so no match
crosses a document boundary.
"""

from __future__ import annotations

import numpy as np


def suffix_array(s: np.ndarray):
    """Returns (sa, ranks) where ranks[j] is the rank array for prefix length 2^j."""
    s = np.asarray(s)
    n = len(s)
    rank = np.unique(s, return_inverse=True)[1].astype(np.int64)
    ranks = [rank.astype(np.int32)]
    h = 1
    while True:
        second = np.zeros(n, np.int64)
        second[: n - h] = rank[h:] + 1 if h < n else 0
        key = rank * (n + 1) + second
        sa = np.argsort(key, kind="stable")
        k = key[sa]
        new = np.empty(n, np.int64)
        new[sa] = np.concatenate([[0], np.cumsum(k[1:] != k[:-1])])
        rank = new
        ranks.append(rank.astype(np.int32))
        h *= 2
        if rank.max() == n - 1:
            return sa, ranks


def lcp_adjacent(sa: np.ndarray, ranks: list[np.ndarray]) -> np.ndarray:
    """lcp[i] = longest common prefix of suffixes sa[i] and sa[i + 1]."""
    n = len(sa)
    a, b = sa[:-1].astype(np.int64), sa[1:].astype(np.int64)
    ell = np.zeros(n - 1, np.int64)
    for j in range(len(ranks) - 1, -1, -1):
        h = 1 << j
        pa, pb = a + ell, b + ell
        ok = (pa + h <= n) & (pb + h <= n)  # both blocks of h tokens lie inside the text
        idx = np.flatnonzero(ok)
        r = ranks[j]
        same = r[pa[idx]] == r[pb[idx]]
        ell[idx[same]] += h
    return ell


def concat(docs: list[np.ndarray]):
    """Joins token-id arrays with a unique separator after each; returns (text, starts)."""
    base = max((int(d.max()) for d in docs if len(d)), default=-1) + 1
    parts, starts, off = [], [], 0
    for i, d in enumerate(docs):
        parts += [np.asarray(d, np.int64), np.array([base + i], np.int64)]
        starts.append(off)
        off += len(d) + 1
    return np.concatenate(parts), np.array(starts + [off])


def duplicate_mask(text: np.ndarray, k: int, sa=None, lcp=None) -> np.ndarray:
    """Boolean mask of tokens inside a ≥ k-token span that also occurs earlier."""
    if sa is None:
        sa, ranks = suffix_array(text)
        lcp = lcp_adjacent(sa, ranks)
    n = len(text)
    long = lcp >= k
    if not long.any():
        return np.zeros(n, bool)
    # suffixes sa[i] and sa[i + 1] share a k-token prefix; runs of such links form groups
    # of equal k-grams. Keep each group's earliest position, mark the rest.
    edge = np.diff(np.r_[0, long.astype(np.int8), 0])  # +1 where a run starts
    member = np.r_[long, False] | np.r_[False, long]
    rid = np.cumsum(edge[:n] == 1)[member]
    pos = sa[member].astype(np.int64)
    order = np.lexsort((pos, rid))
    pos, rid = pos[order], rid[order]
    first = np.r_[True, rid[1:] != rid[:-1]]
    dup = pos[~first]
    cover = np.zeros(n + 1, np.int64)
    np.add.at(cover, dup, 1)
    np.add.at(cover, np.minimum(dup + k, n), -1)
    return np.cumsum(cover)[:n] > 0
