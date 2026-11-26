"""MinHash (Broder 1997) and one-permutation hashing with densification.

**MinHash.** For a random hash h, ``P[min h(A) = min h(B)] = J(A, B)``: the minimum over
A ∪ B is equally likely to be any element, and it is shared exactly when that element is
in A ∩ B. K independent hashes give K Bernoulli(J) trials, so the fraction of equal
signature slots is an unbiased estimate of J with standard deviation ``√(J(1 − J)/K)``.
Shingles are already well-mixed 64-bit hashes, so each of the K functions is a cheap
bijection ``a_k·x + b_k mod 2^64`` (a_k odd): one multiply and one add per element instead
of a full mixer. The tests check that bias and spread stay binomial. Cost: K hash
evaluations per shingle.

**One-permutation hashing (Li, Owen & Zhang 2012).** Hash each shingle once, use the hash
to pick one of K bins, and keep the minimum per bin. That is K times cheaper. A bin that
receives no shingle (likely when a document has fewer shingles than about K) must be
**densified** so that every slot is comparable:

- ``"rotation"`` (Shrivastava & Li 2014): borrow from the nearest non-empty bin to the
  right, circularly, plus ``t · C`` for distance t. Unbiased, but neighbouring empty bins
  borrow from the same bin, so their agreements are correlated and the variance grows when
  many bins are empty.
- ``"optimal"`` (Shrivastava 2017): empty bin j probes bins ``g(j, 0), g(j, 1), …`` given
  by a fixed hash until one is non-empty and borrows its value. Different empty bins mostly
  borrow from different bins, which restores roughly MinHash's variance. The tests measure
  both.
"""

from __future__ import annotations

import numpy as np

from .hashing import MAX64, splitmix64

_OFFSET = np.uint64(0x5851F42D4C957F2D)  # odd constant added per step of borrowing
_PROBE, _SALT = np.uint64(0x100000001B3), np.uint64(0xD1B54A32D192ED03)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 0, max_elems: int = 4_000_000):
        rng = np.random.default_rng(seed)
        self.K = num_perm
        self.a = rng.integers(0, 2**63, num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.integers(0, 2**63, num_perm, dtype=np.uint64) * np.uint64(2)
        self.max_elems = max_elems  # cap on the (rows × shingles) block held in memory

    def signature(self, h: np.ndarray) -> np.ndarray:
        out = np.full(self.K, MAX64, np.uint64)
        if len(h) == 0:
            return out
        rows = max(1, min(self.K, self.max_elems // len(h)))
        with np.errstate(over="ignore"):
            for i in range(0, self.K, rows):
                a, b = self.a[i : i + rows, None], self.b[i : i + rows, None]
                out[i : i + rows] = (h[None, :] * a + b).min(1)
        return out


class OnePermHasher:
    def __init__(self, num_bins: int = 128, seed: int = 0, densify: str = "optimal"):
        if densify not in ("optimal", "rotation"):
            raise ValueError(densify)
        self.K, self.densify = num_bins, densify
        self.seed = np.uint64(np.random.default_rng(seed).integers(0, 2**63))

    def signature(self, h: np.ndarray) -> np.ndarray:
        K = self.K
        sig = np.full(K, MAX64, np.uint64)
        if len(h) == 0:
            return sig
        v = splitmix64(h ^ self.seed)
        b = (v % np.uint64(K)).astype(np.int64)
        np.minimum.at(sig, b, v)
        filled = sig != MAX64
        if filled.all():
            return sig
        if self.densify == "optimal":
            return self._optimal(sig, filled)
        # nearest filled bin at or to the right of j, circularly: search a doubled index line
        pos = np.arange(2 * K)
        cand = np.where(np.concatenate([filled, filled]), pos, 2 * K)
        nxt = np.minimum.accumulate(cand[::-1])[::-1][:K]
        t = (nxt - np.arange(K)).astype(np.uint64)
        with np.errstate(over="ignore"):
            return sig[nxt % K] + t * _OFFSET

    def _optimal(self, sig, filled):
        K = np.uint64(self.K)
        out = sig.copy()
        todo = np.flatnonzero(~filled).astype(np.uint64)
        attempt = np.uint64(0)
        while len(todo):  # the probe sequence depends only on (j, attempt), not on the data
            probe = splitmix64(todo * _PROBE ^ attempt ^ _SALT)
            k = (probe % K).astype(np.int64)
            ok = filled[k]
            out[todo[ok].astype(np.int64)] = sig[k[ok]]
            todo = todo[~ok]
            attempt += np.uint64(1)
        return out


def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def signatures(hasher, sets) -> np.ndarray:
    return np.stack([hasher.signature(h) for h in sets]) if sets else np.empty((0, hasher.K))
