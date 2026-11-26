# llm-data-dedup

[![CI](https://github.com/kishorhange111/llm-data-dedup/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/llm-data-dedup/actions/workflows/ci.yml)

Near-duplicate removal for pretraining data, written from scratch in NumPy. It covers
MinHash, one-permutation hashing with two densification schemes, and LSH banding with
the S-curve. It also includes an **exact all-pairs Jaccard join** (prefix filtering).
Because that join returns the exact answer, every recall and precision number below is
measured, not estimated.

The corpus needs no download: it is the 22k `.py` files of the local Python
environments. Site-packages is full of real near-duplicates (vendored libraries,
generated modules, copied helpers), the same kind of redundancy code-pretraining
pipelines remove.

## Method

| step | file | what it does |
|---|---|---|
| shingles | [`hashing.py`](src/dedup/hashing.py) | identifier / number / punctuation tokens → hashed token 5-grams → a set per file |
| MinHash | [`minhash.py`](src/dedup/minhash.py) | K = 128 bijections `a·x + b mod 2^64`; `P[min agrees] = J` |
| one-permutation hashing | [`minhash.py`](src/dedup/minhash.py) | one hash per shingle, binned into K slots: K times fewer hash evaluations |
| densification | [`minhash.py`](src/dedup/minhash.py) | fills OPH's empty bins: *rotation* (2014) or *optimal* (2017) |
| LSH | [`lsh.py`](src/dedup/lsh.py) | b bands × r rows. Candidate if a band matches, with `P = 1 − (1 − s^r)^b`. Clusters are connected components of the shared buckets |
| (b, r) choice | [`lsh.py`](src/dedup/lsh.py) | minimises weighted false-positive and false-negative area under the S-curve |
| exact join | [`allpairs.py`](src/dedup/allpairs.py) | rarest-first global order. Sets with J ≥ t must share an element in their first `|x| − ⌈t|x|⌉ + 1`. Index the prefixes, apply the size filter, verify |

## Results: 22,457 Python files, 295 MB

[`bench.py`](src/dedup/bench.py) writes [results/bench.md](results/bench.md). The corpus
is 22,457 `.py` files of up to 1 MB each. Dropping 1,257 byte-identical copies leaves
21,191 files with at least 5 tokens and 44.2 M distinct shingles. The **exact** join
finds every pair with J ≥ 0.5 in 94 s on one core, verifying 163k candidates. 42,908
pairs have J ≥ 0.8. Keeping one file per connected cluster of those pairs removes
**1,242 files**, and that is the target every method below is measured against.

LSH with K = 128 on one core. *Recall* and *precision* are over pairs (J ≥ 0.8) among LSH candidates. The *est ≥ t* columns keep only candidates whose signature estimate is ≥ 0.8. The last two columns are files removed by keeping one per cluster, against **1,242** for the exact answer.

| hasher | b × r | signatures (s) | candidates | recall | precision | recall, est ≥ t | precision, est ≥ t | removed, LSH clusters | removed, verified |
|---|---|---|---|---|---|---|---|---|---|
| MinHash | 9 × 13 | 46.2 | 34,867 | 0.801 | 0.986 | 0.795 | 0.998 | 1,294 | 1,158 |
| MinHash | 14 × 9 | 46.2 | 46,738 | 0.982 | 0.902 | 0.962 | 0.995 | 1,715 | 1,232 |
| OPH, optimal densification | 9 × 13 | 8.3 | 34,650 | 0.788 | 0.976 | 0.786 | 0.999 | 1,278 | 1,153 |
| OPH, optimal densification | 14 × 9 | 8.3 | 44,572 | 0.963 | 0.927 | 0.950 | 0.997 | 1,667 | 1,231 |
| OPH, rotation densification | 9 × 13 | 2.7 | 36,753 | 0.791 | 0.923 | 0.788 | 0.997 | 1,680 | 1,180 |
| OPH, rotation densification | 14 × 9 | 2.7 | 49,065 | 0.964 | 0.843 | 0.950 | 0.996 | 2,288 | 1,235 |

- **Banding matters more than the hasher.** The equal-weight optimum (9 × 13) centres the
  S-curve at 0.84, above the 0.8 threshold, so it misses ~20% of true pairs. Weighting
  false negatives at 0.9 (14 × 9) moves it to 0.75 and lifts pair recall to 0.96–0.98.
- **Missing pairs barely matters; false links do.** Near-duplicates form clusters, and a
  cluster stays connected even when some of its pairs are missed. With 14 × 9, the
  verified pipelines remove 1,231–1,235 of the 1,242 files, at least 99.1%. Clustering
  raw LSH buckets without a check chains false positives into clusters: it removes
  1,667–2,288 files, up to 84% too many. Verification can be cheap. Keeping a candidate
  only if its *estimated* Jaccard is ≥ t gives 0.995+ precision at almost the same
  recall, with no exact computation.
- **One-permutation hashing is the cheaper hasher.** It needs one hash per shingle instead
  of 128. Signatures take 8.3 s against 46.2 s (5.6×, not 128×, because per-file Python
  overhead and the scatter-min take the rest). With optimal densification it matches
  MinHash on files removed (1,231 vs 1,232). Rotation densification is faster still, but its correlated bins put more pairs
  below the threshold into buckets: precision 0.84 against 0.93 for optimal densification
  at 14 × 9. That is the same effect the variance test measures on synthetic sets.
- **The biggest clusters are generated code.** 267 near-identical `plotly/graph_objs`
  property classes form the largest cluster, and five of the next six are plotly too.

**Does LSH follow its S-curve on real data?** Per pair, not at first sight: MinHash at
9 × 13 catches 54% of the pairs with J in [0.85, 0.90), where the curve predicts 80%. The
pairs are not independent trials, though. 96% of that bin's pairs sit inside one cluster
of near-identical plotly files, and those pairs share their signatures. Averaging the rate
per cluster instead gives **0.793 against the predicted 0.795**. The cluster average is
within 0.03 of the curve in five of the other seven bins and 0.12 off at worst, which is
the noise expected from 64–212 clusters per bin. The synthetic test, which uses
independent pairs, matches the curve directly.

| true J | pairs | clusters | largest cluster's share | measured, per pair | measured, cluster average | 1 − (1 − s^r)^b |
|---|---|---|---|---|---|---|
| [0.50, 0.60) | 13,532 | 212 | 26% | 0.005 | 0.006 | 0.005 |
| [0.60, 0.70) | 8,217 | 169 | 52% | 0.012 | 0.060 | 0.034 |
| [0.70, 0.75) | 1,860 | 75 | 30% | 0.077 | 0.087 | 0.128 |
| [0.75, 0.80) | 901 | 64 | 39% | 0.211 | 0.293 | 0.273 |
| [0.80, 0.85) | 6,046 | 70 | 92% | 0.547 | 0.505 | 0.626 |
| [0.85, 0.90) | 11,551 | 70 | 96% | 0.537 | 0.793 | 0.795 |
| [0.90, 0.95) | 15,612 | 69 | 85% | 0.970 | 0.949 | 0.973 |
| [0.95, 1.00) | 9,699 | 116 | 79% | 1.000 | 1.000 | 1.000 |

## Results: exact-substring dedup

MinHash compares whole files, so it cannot see a license header or a helper pasted into
an otherwise original file. [`suffix_array.py`](src/dedup/suffix_array.py) finds every
span of ≥ k tokens that already occurred earlier in the corpus, as in Lee et al. (2022).
It builds the suffix array by prefix doubling (one NumPy sort per round). The LCP array
comes **exactly and without a Python loop**, by binary lifting over the ranks the doubling
rounds already stored. Every file ends in a unique separator, so no match crosses a
boundary. Tests compare the suffix array, the LCP array and the mask against brute force,
including periodic text.

[`substr.py`](src/dedup/substr.py) runs this on what is left after exact and near-duplicate
removal: a random 8 M-token sample of 2,502 files. The suffix array and LCP take 41 s
(12 doubling rounds), and the longest repeat is 2,526 tokens
([results/substr.md](results/substr.md)).

| k (tokens) | tokens in a repeated span | share | share, rulers excluded | files touched |
|---|---|---|---|---|
| 50 | 647,983 | 8.1% | 7.1% | 1,018 |
| 100 | 286,224 | 3.6% | 3.4% | 491 |
| 200 | 111,297 | 1.4% | 1.4% | 119 |
| 400 | 50,823 | 0.6% | 0.6% | 31 |

- **Near-duplicate removal leaves plenty of repeated text behind.** After it, 7.1% of
  tokens still sit in a ≥ 50-token span seen earlier, in 41% of the files. Runs of a
  single repeated token (comment rulers such as `# -----`) are excluded from that share
  and add only about 1 point.
- **Long repeats are concentrated.** At k = 200, 1.4% of tokens are repeated, in 119
  files; at k = 400 it is 0.6%, in 31 files.
- **The most copied content is the Apache-2.0 license header**, about 80 copies in the
  sample. That boilerplate does not make two files near-duplicates, but it is exactly what
  substring dedup removes.

## Tests

`pytest -q` runs 30 tests. They check:
- MinHash and both OPH variants are **unbiased** (200 seeds, three Jaccard levels,
  including sets smaller than K);
- MinHash and optimal densification have **binomial spread** `√(J(1 − J)/K)`, and
  rotation densification is measurably noisier on sparse sets;
- the measured LSH candidate rate follows `1 − (1 − s^r)^b` at three similarity levels;
- `optimal_params` puts the threshold on the steep part of the curve;
- bucket-chain clustering gives the same components as listing every candidate pair;
- the prefix-filter join returns **exactly** the brute-force pair set at t = 0.5 and 0.8;
- the suffix array and LCP equal brute force, including periodic text, the worst case for
  hash-based LCP;
- the substring mask marks exactly the repeats after their first occurrence, and
  separators stop matches at document boundaries;
- shingles ignore whitespace but not renames.

```
pip install -e ".[dev]" && pytest -q
python -m dedup.bench --out results      # scans this machine's Python environments
python -m dedup.substr --out results
```
