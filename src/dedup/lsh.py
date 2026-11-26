"""Locality-sensitive hashing over MinHash signatures (banding).

Split the K = b·r signature slots into b bands of r rows. Two documents become a
**candidate pair** if all r rows agree in at least one band. With per-slot agreement
probability s (their Jaccard), that happens with probability

    P(s) = 1 − (1 − s^r)^b,

an S-curve whose steep part sits near ``(1/b)^(1/r)``. ``optimal_params`` picks (b, r) for
a target threshold by minimising the weighted area of false positives (P(s) for s < t) and
false negatives (1 − P(s) for s ≥ t), the rule datasketch uses.

Clustering does not need the candidate pairs themselves. Sorting the band keys puts each
bucket's members next to each other, and linking consecutive members gives a graph whose
connected components are the duplicate clusters. That costs O(N log N) per band even when
one bucket holds thousands of copies, where listing its pairs would be quadratic.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .hashing import splitmix64


def s_curve(s, b: int, r: int):
    return 1.0 - (1.0 - np.asarray(s, dtype=np.float64) ** r) ** b


def optimal_params(threshold: float, num_perm: int, fp_weight=0.5, fn_weight=0.5):
    """(b, r) with b·r ≤ num_perm minimising fp_weight·FP area + fn_weight·FN area."""
    lo = np.linspace(0.0, threshold, 201)
    hi = np.linspace(threshold, 1.0, 201)
    best, arg = np.inf, None
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = np.trapezoid(s_curve(lo, b, r), lo)
            fn = np.trapezoid(1.0 - s_curve(hi, b, r), hi)
            err = fp_weight * fp + fn_weight * fn
            if err < best:
                best, arg = err, (b, r)
    return arg


def band_keys(sigs: np.ndarray, b: int, r: int) -> np.ndarray:
    """(N, b) uint64: one hash of the r slots of each band."""
    sigs = np.asarray(sigs, dtype=np.uint64)
    keys = np.zeros((len(sigs), b), np.uint64)
    for j in range(b):
        k = np.full(len(sigs), np.uint64(j), np.uint64)
        for c in range(j * r, (j + 1) * r):
            k = splitmix64(k ^ sigs[:, c])
        keys[:, j] = k
    return keys


def clusters(keys: np.ndarray) -> np.ndarray:
    """Component label per document from band keys (union over bands of shared buckets)."""
    N, b = keys.shape
    src, dst = [], []
    for j in range(b):
        order = np.argsort(keys[:, j], kind="stable")
        same = keys[order[1:], j] == keys[order[:-1], j]
        src.append(order[:-1][same])
        dst.append(order[1:][same])
    src, dst = np.concatenate(src), np.concatenate(dst)
    g = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(N, N))
    return connected_components(g, directed=False)[1]


def candidate_pairs(keys: np.ndarray) -> np.ndarray:
    """All pairs (i < j) sharing a bucket in some band, as int64 codes ``i·N + j``."""
    N, b = keys.shape
    codes = []
    for j in range(b):
        order = np.argsort(keys[:, j], kind="stable")
        k = keys[order, j]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        sizes = np.diff(np.r_[starts, N])
        for s, g in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
            m = np.sort(order[s : s + g])
            i, jj = np.triu_indices(g, 1)
            codes.append(m[i] * N + m[jj])
    return np.unique(np.concatenate(codes)) if codes else np.empty(0, np.int64)
