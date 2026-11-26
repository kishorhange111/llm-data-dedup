import itertools

import numpy as np
import pytest

from dedup.allpairs import similar_pairs
from dedup.hashing import Vocab, jaccard, shingles, splitmix64, tokenize
from dedup.lsh import band_keys, candidate_pairs, clusters, optimal_params, s_curve
from dedup.minhash import MinHasher, OnePermHasher, estimate
from dedup.suffix_array import concat, duplicate_mask, lcp_adjacent, suffix_array


def pair_with_jaccard(shared, diff, rng):
    """Two sets of hashed ints with exact Jaccard shared / (shared + 2·diff)."""
    u = rng.choice(10**9, shared + 2 * diff, replace=False).astype(np.uint64)
    a = np.unique(splitmix64(np.r_[u[:shared], u[shared : shared + diff]]))
    b = np.unique(splitmix64(np.r_[u[:shared], u[shared + diff :]]))
    return a, b


def test_splitmix_is_injective_on_a_sample_and_deterministic():
    x = np.arange(200_000, dtype=np.uint64)
    assert len(np.unique(splitmix64(x))) == len(x)
    assert (splitmix64(x) == splitmix64(x.copy())).all()


def test_shingles_ignore_whitespace_but_not_renames():
    v = Vocab()
    a = shingles(v.encode(tokenize("def f(x):\n    return x + 1\n")), 3)
    b = shingles(v.encode(tokenize("def f( x ):  return x+1")), 3)
    c = shingles(v.encode(tokenize("def f(y):\n    return y + 1\n")), 3)
    assert (a == b).all() and jaccard(a, c) < 1.0
    assert len(shingles(v.encode(["a", "b"]), 5)) == 0


def _estimates(make, shared, diff, trials=200):
    a, b = pair_with_jaccard(shared, diff, np.random.default_rng(0))
    est = np.array([estimate(h.signature(a), h.signature(b)) for h in map(make, range(trials))])
    return jaccard(a, b), est


HASHERS = {
    "minhash": lambda s: MinHasher(128, seed=s),
    "oph-optimal": lambda s: OnePermHasher(128, seed=s, densify="optimal"),
    "oph-rotation": lambda s: OnePermHasher(128, seed=s, densify="rotation"),
}


@pytest.mark.parametrize("name", list(HASHERS))
@pytest.mark.parametrize("shared,diff", [(800, 100), (300, 300), (40, 5)])
def test_estimators_are_unbiased(name, shared, diff):
    """Mean over 200 independent seeds is within 4 standard errors of J. (40, 5) has fewer
    elements than bins, which exercises densification."""
    J, est = _estimates(HASHERS[name], shared, diff)
    assert abs(est.mean() - J) < 4 * est.std() / np.sqrt(len(est)) + 1e-3


@pytest.mark.parametrize("name", ["minhash", "oph-optimal"])
@pytest.mark.parametrize("shared,diff", [(800, 100), (300, 300), (40, 5)])
def test_spread_is_binomial(name, shared, diff):
    J, est = _estimates(HASHERS[name], shared, diff)
    sd = np.sqrt(J * (1 - J) / 128)
    assert 0.7 * sd < est.std() < 1.35 * sd


def test_rotation_densification_is_noisier_on_sparse_sets():
    J, rot = _estimates(HASHERS["oph-rotation"], 40, 5)
    _, opt = _estimates(HASHERS["oph-optimal"], 40, 5)
    assert rot.std() > 1.3 * opt.std()


def test_lsh_candidate_rate_follows_the_s_curve():
    rng = np.random.default_rng(1)
    b, r = 16, 8
    for shared, diff in [(700, 150), (850, 75), (950, 25)]:
        hits, trials = 0, 300
        for s in range(trials):
            a, c = pair_with_jaccard(shared, diff, rng)
            h = MinHasher(b * r, seed=s)
            keys = band_keys(np.stack([h.signature(a), h.signature(c)]), b, r)
            hits += bool((keys[0] == keys[1]).any())
        p = s_curve(jaccard(a, c), b, r)
        assert abs(hits / trials - p) < 4 * np.sqrt(p * (1 - p) / trials) + 0.01


def test_optimal_params_put_the_threshold_on_the_steep_part():
    for t in (0.5, 0.7, 0.85):
        b, r = optimal_params(t, 128)
        assert b * r <= 128
        assert abs((1 / b) ** (1 / r) - t) < 0.1


def test_clusters_are_components_of_candidate_pairs():
    rng = np.random.default_rng(2)
    base = [rng.choice(10**6, 200, replace=False).astype(np.uint64) for _ in range(6)]
    sets = []
    for s in base:  # three noisy copies of each base set
        for _ in range(3):
            keep = s[rng.random(len(s)) > 0.05]
            sets.append(np.unique(splitmix64(np.r_[keep, rng.integers(10**7, 10**8, 8)])))
    h = MinHasher(64, seed=0)
    keys = band_keys(np.stack([h.signature(x) for x in sets]), 16, 4)
    lab = clusters(keys)
    N = len(sets)
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for c in candidate_pairs(keys):
        parent[find(int(c // N))] = find(int(c % N))
    roots = [find(i) for i in range(N)]
    for i, j in itertools.combinations(range(N), 2):
        assert (lab[i] == lab[j]) == (roots[i] == roots[j])
    assert len(set(lab)) == 6  # each base set's copies form one cluster


@pytest.mark.parametrize("t", [0.5, 0.8])
def test_prefix_filter_join_is_exact(t):
    rng = np.random.default_rng(3)
    sets = []
    for _ in range(40):
        s = rng.choice(400, rng.integers(5, 60), replace=False).astype(np.uint64)
        sets.append(np.unique(s))
        if rng.random() < 0.5:  # a near copy
            keep = s[rng.random(len(s)) > rng.uniform(0, 0.4)]
            extra = rng.integers(400, 500, rng.integers(0, 4))
            sets.append(np.unique(np.r_[keep, extra].astype(np.uint64)))
    N = len(sets)
    truth = {
        i * N + j
        for i, j in itertools.combinations(range(N), 2)
        if jaccard(sets[i], sets[j]) >= t - 1e-12
    }
    got, J, _ = similar_pairs(sets, t)
    assert set(got.tolist()) == truth and len(truth) > 5
    for c, j in zip(got, J, strict=True):
        assert abs(jaccard(sets[c // N], sets[c % N]) - j) < 1e-12


def _text(rng, n=600, alphabet=4):
    t = rng.integers(0, alphabet, n)
    for _ in range(5):  # plant long repeats
        a, b, L = rng.integers(0, n - 40, 3)
        L = int(L % 30) + 10
        t[b : b + L] = t[a : a + L].copy()
    return t


@pytest.mark.parametrize("seed", range(3))
def test_suffix_array_and_lcp_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    t = _text(rng)
    sa, ranks = suffix_array(t)
    suffixes = [tuple(t[i:]) for i in range(len(t))]
    assert sa.tolist() == sorted(range(len(t)), key=lambda i: suffixes[i])
    lcp = lcp_adjacent(sa, ranks)
    for i in range(len(t) - 1):
        a, b = suffixes[sa[i]], suffixes[sa[i + 1]]
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        assert lcp[i] == n


def test_periodic_text_is_handled_exactly():
    t = np.tile([0, 1], 64)  # worst case for hashing-based LCP; ranks are exact
    sa, ranks = suffix_array(t)
    lcp = lcp_adjacent(sa, ranks)
    assert lcp.max() == len(t) - 2


@pytest.mark.parametrize("k", [5, 12])
def test_duplicate_mask_marks_every_repeat_after_its_first_occurrence(k):
    rng = np.random.default_rng(4)
    t = _text(rng, alphabet=6)
    seen, want = set(), np.zeros(len(t), bool)
    for p in range(len(t) - k + 1):
        g = tuple(t[p : p + k])
        if g in seen:
            want[p : p + k] = True
        seen.add(g)
    assert (duplicate_mask(t, k) == want).all() and want.any()


def test_separators_stop_matches_at_document_boundaries():
    a = np.array([1, 2, 3, 4, 5, 6])
    text, starts = concat([a[:3], a[3:], a])  # "123" + "456" + "123456"
    m = duplicate_mask(text, 4)
    assert not m.any()  # no 4-token span repeats without crossing a boundary
    assert duplicate_mask(text, 3)[starts[2] : starts[3] - 1].all()
