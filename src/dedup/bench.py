"""Near-duplicate detection on a real code corpus, measured against the exact answer.

    python -m dedup.bench --out results

Pipeline: load ``.py`` files → drop byte-identical copies → 5-gram token shingles → the
exact set of pairs with Jaccard ≥ t (prefix-filter join) → for each hasher (MinHash,
one-permutation hashing with each densification), signatures + LSH banding at the
(b, r) that ``optimal_params`` picks for t. Reported per hasher:

- signature time;
- **recall** of the exact pairs ≥ t among LSH candidates, and precision of the candidates;
- the same after the usual cheap check "estimated Jaccard ≥ t" (no exact verification);
- the measured candidate rate by true Jaccard against the S-curve ``1 − (1 − s^r)^b``;
- documents a cluster-and-keep-one pass would remove, against the exact graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .allpairs import similar_pairs
from .corpus import default_roots, exact_dedup, load
from .hashing import Vocab, shingles, tokenize
from .lsh import band_keys, candidate_pairs, clusters, optimal_params, s_curve
from .minhash import MinHasher, OnePermHasher, estimate

BINS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0001]


def components(codes, N):
    g = coo_matrix((np.ones(len(codes), np.int8), (codes // N, codes % N)), shape=(N, N))
    return connected_components(g, directed=False)[1]


def removed_by(labels):
    """Docs dropped when each cluster keeps one member."""
    return len(labels) - len(np.unique(labels))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5, help="shingle size in tokens")
    ap.add_argument("--t", type=float, default=0.8, help="Jaccard threshold")
    ap.add_argument("--K", type=int, default=128, help="signature length")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--fn-weights", type=float, nargs="+", default=[0.5, 0.9])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log = {}
    T = time.perf_counter

    t0 = T()
    raw = load(a.roots or default_roots())
    if a.max_docs:
        raw = raw[: a.max_docs]
    docs, n_exact = exact_dedup(raw)
    log["files"], log["exact_duplicates"] = len(raw), n_exact
    log["bytes"] = sum(len(d.text) for d in raw)
    vocab = Vocab()
    sets, keep = [], []
    for d in docs:
        s = shingles(vocab.encode(tokenize(d.text)), a.n)
        if len(s):
            sets.append(s)
            keep.append(d)
    docs, N = keep, len(sets)
    log.update(docs=N, shingles=int(sum(map(len, sets))), load_s=T() - t0)
    print(json.dumps(log), flush=True)

    t0 = T()
    lo = min(BINS[0], a.t)
    codes_lo, J_lo, n_verified = similar_pairs(sets, lo)
    log["exact_join_s"], log["exact_join_verified"] = T() - t0, n_verified
    truth = codes_lo[J_lo >= a.t]
    log["exact_pairs"] = len(truth)
    exact_lab = components(truth, N)
    lo_lab = components(codes_lo, N)  # clusters of the J ≥ 0.5 graph, to group correlated pairs
    pair_comp = lo_lab[codes_lo // N]
    log["exact_removed"] = removed_by(exact_lab)
    print(json.dumps(log), flush=True)

    rows = []
    hashers = {
        "MinHash": MinHasher(a.K),
        "OPH, optimal densification": OnePermHasher(a.K, densify="optimal"),
        "OPH, rotation densification": OnePermHasher(a.K, densify="rotation"),
    }
    settings = [(fw, *optimal_params(a.t, a.K, 1 - fw, fw)) for fw in a.fn_weights]
    log["settings"] = [{"fn_weight": fw, "b": b, "r": r} for fw, b, r in settings]
    for name, h in hashers.items():
        t0 = T()
        sigs = np.stack([h.signature(s) for s in sets])
        sig_s = T() - t0
        for fw, b, r in settings:
            keys = band_keys(sigs[:, : b * r], b, r)
            cand = candidate_pairs(keys)
            true_cand = cand[np.isin(cand, truth)]
            pairs = zip(cand // N, cand % N, strict=True)
            est = np.array([estimate(sigs[i], sigs[j]) for i, j in pairs])
            flagged = cand[est >= a.t] if len(cand) else cand
            tp_est = int(np.isin(flagged, truth).sum())
            in_lo = np.isin(codes_lo, cand)
            by_bin = []
            for lo_b, hi_b in zip(BINS[:-1], BINS[1:], strict=True):
                m = (J_lo >= lo_b) & (J_lo < hi_b)
                if m.any():
                    # pairs inside one big cluster are far from independent trials, so also
                    # average the rate per cluster, and report the largest cluster's share
                    u = np.unique(pair_comp[m], return_inverse=True, return_counts=True)
                    comp, inv_c, cnt = u
                    per = np.bincount(inv_c, in_lo[m]) / cnt
                    by_bin.append({
                        "J": [lo_b, min(hi_b, 1.0)], "pairs": int(m.sum()),
                        "clusters": len(comp), "largest_share": float(cnt.max() / m.sum()),
                        "measured": float(in_lo[m].mean()),
                        "measured_cluster_avg": float(per.mean()),
                        "s_curve": float(s_curve(J_lo[m], b, r).mean()),
                    })  # fmt: skip
            row = {
                "hasher": name, "K": a.K, "fn_weight": fw, "b": b, "r": r,
                "signature_s": sig_s, "candidates": len(cand),
                "recall": len(true_cand) / max(1, len(truth)),
                "precision": len(true_cand) / max(1, len(cand)),
                "est_flagged": len(flagged), "est_recall": tp_est / max(1, len(truth)),
                "est_precision": tp_est / max(1, len(flagged)),
                "lsh_removed": removed_by(clusters(keys)),
                "verified_removed": removed_by(components(true_cand, N)),
                "by_jaccard": by_bin,
            }  # fmt: skip
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "by_jaccard"}), flush=True)

    # the biggest exact near-duplicate clusters, by top-level package
    sizes = Counter(exact_lab.tolist())
    top = []
    for lab_id, n in sizes.most_common(8):
        if n < 2:
            break
        members = np.flatnonzero(exact_lab == lab_id)
        pk = Counter(docs[m].path.replace("\\", "/").split("/")[0] for m in members)
        top.append({"size": int(n), "packages": dict(pk.most_common(4)),
                    "example": docs[members[0]].path.replace("\\", "/")})  # fmt: skip
    log["top_clusters"] = top
    with open(os.path.join(a.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump({"corpus": log, "hashers": rows}, f, indent=1)
    write_md(log, rows, a, os.path.join(a.out, "bench.md"))


def write_md(log, rows, a, path):
    L = [
        f"Corpus: {log['files']:,} `.py` files ({log['bytes'] / 2**20:.0f} MB), "
        f"{log['exact_duplicates']:,} byte-identical copies removed, {log['docs']:,} documents "
        f"with ≥ {a.n} tokens, {log['shingles']:,} distinct {a.n}-gram shingles in total.",
        "",
        f"Exact join (prefix filter, J ≥ {min(BINS[0], a.t)}): {log['exact_join_s']:.1f} s, "
        f"{log['exact_join_verified']:,} candidates verified. Pairs with J ≥ {a.t}: "
        f"**{log['exact_pairs']:,}**; keeping one document per connected cluster removes "
        f"{log['exact_removed']:,}.",
        "",
        f"K = {a.K}. (b, r) from `optimal_params({a.t}, {a.K}, fp_weight, fn_weight)`: "
        + ", ".join(f"fn_weight {s['fn_weight']} → {s['b']} × {s['r']}" for s in log["settings"])
        + ".",
        "",
        "| hasher | b × r | signatures (s) | candidates | recall | precision "
        "| recall, est ≥ t | precision, est ≥ t | removed, LSH clusters | removed, verified "
        f"(exact: {log['exact_removed']:,}) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['hasher']} | {r['b']} × {r['r']} | {r['signature_s']:.1f} "
            f"| {r['candidates']:,} | {r['recall']:.3f} | {r['precision']:.3f} "
            f"| {r['est_recall']:.3f} | {r['est_precision']:.3f} | {r['lsh_removed']:,} "
            f"| {r['verified_removed']:,} |"
        )
    for r in rows:
        L += ["", f"Candidate rate by true Jaccard, {r['hasher']}, {r['b']} × {r['r']}:", "",
              "| true J | pairs | clusters | largest cluster's share | measured, per pair "
              "| measured, cluster average | 1 − (1 − s^r)^b |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
        for x in r["by_jaccard"]:
            L.append(
                f"| [{x['J'][0]:.2f}, {x['J'][1]:.2f}) | {x['pairs']:,} | {x['clusters']:,} "
                f"| {100 * x['largest_share']:.0f}% | {x['measured']:.3f} "
                f"| {x['measured_cluster_avg']:.3f} | {x['s_curve']:.3f} |"
            )
    L += ["", "Largest near-duplicate clusters (exact, J ≥ t):", ""]
    for c in log["top_clusters"]:
        L.append(f"- {c['size']} files, {c['packages']}, e.g. `{c['example']}`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
