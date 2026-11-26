"""Stage 2: exact-substring dedup on what near-duplicate removal leaves.

    python -m dedup.substr --out results

Byte-identical files are dropped, then near-duplicate clusters (exact J ≥ t) keep one
file each. A random sample of the survivors, up to ``--tokens`` tokens, is concatenated,
and a suffix array finds every span of ≥ k tokens that occurs earlier in the sample.
Reports the fraction of tokens such spans cover for several k, and the most repeated
spans.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .allpairs import similar_pairs
from .bench import components
from .corpus import default_roots, exact_dedup, load
from .hashing import Vocab, shingles, tokenize
from .suffix_array import concat, duplicate_mask, lcp_adjacent, suffix_array


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    T = time.perf_counter
    docs, _ = exact_dedup(load(a.roots or default_roots()))
    vocab = Vocab()
    ids, sets, kept = [], [], []
    for d in docs:
        x = vocab.encode(tokenize(d.text))
        s = shingles(x, 5)
        if len(s):
            ids.append(x)
            sets.append(s)
            kept.append(d)
    N = len(sets)
    codes, _, _ = similar_pairs(sets, a.t)
    lab = components(codes, N)
    first = np.unique(lab, return_index=True)[1]  # one file per near-duplicate cluster
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(first)
    take, total = [], 0
    for i in order:
        if total + len(ids[i]) > a.tokens:
            continue
        take.append(i)
        total += len(ids[i])
    text, starts = concat([ids[i] for i in take])
    t0 = T()
    sa, ranks = suffix_array(text)
    lcp = lcp_adjacent(sa, ranks)
    sa_s = T() - t0
    inv = np.empty(len(vocab.ids), dtype=object)
    for tok, j in vocab.ids.items():
        inv[j] = tok
    out = {
        "files_after_near_dedup": len(first), "sample_files": len(take),
        "sample_tokens": total, "suffix_array_s": sa_s, "doubling_rounds": len(ranks) - 1,
        "max_lcp": int(lcp.max()), "by_k": [],
    }  # fmt: skip
    real = np.ones(len(text), bool)
    real[starts[1:] - 1] = False  # separators
    # runs of ≥ 8 identical tokens (comment rulers like "# -----", "=====") are repeated
    # everywhere; report shares with and without them
    starts_run = np.flatnonzero(np.r_[True, text[1:] != text[:-1]])
    run_len = np.diff(np.r_[starts_run, len(text)])
    ruler = np.repeat(run_len >= 8, run_len)
    for k in a.k:
        m = duplicate_mask(text, k, sa, lcp) & real
        doc_of = np.searchsorted(starts, np.flatnonzero(m), side="right") - 1
        out["by_k"].append({
            "k": k, "tokens_marked": int(m.sum()), "fraction": float(m.sum() / total),
            "files_touched": int(len(np.unique(doc_of))),
            "fraction_without_rulers": float((m & ~ruler).sum() / (real & ~ruler).sum()),
        })  # fmt: skip
    # most repeated k-token spans. A repeated block of length L shows up as L − k + 1
    # groups of shifted k-grams, so skip groups that start inside a block already listed.
    k = min(a.k)
    edge = np.diff(np.r_[0, (lcp >= k).astype(np.int8), 0])
    s_idx, e_idx = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    sizes = e_idx - s_idx + 1
    top, chosen = [], []
    for j in np.argsort(-sizes, kind="stable"):
        p = int(sa[s_idx[j] : e_idx[j] + 1].min())
        if any(c <= p < c + 4 * k for c in chosen) or len(np.unique(text[p : p + k])) < 8:
            continue  # already listed, or a ruler-like low-variety span
        chosen.append(p)
        top.append({"copies": int(sizes[j]), "starts_with": " ".join(inv[text[p : p + 14]])})
        if len(top) == 6:
            break
    out["top_spans"] = top
    with open(os.path.join(a.out, "substr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    L = [
        f"{out['sample_files']:,} files ({total:,} tokens) sampled from the "
        f"{out['files_after_near_dedup']:,} left after exact and near-duplicate removal "
        f"(J ≥ {a.t}). Suffix array + LCP: {sa_s:.0f} s, {out['doubling_rounds']} doubling "
        f"rounds, longest repeat {out['max_lcp']:,} tokens.",
        "",
        "| k (tokens) | tokens in a repeated span | share | share, rulers excluded "
        "| files touched |",
        "|---|---|---|---|---|",
    ]
    for r in out["by_k"]:
        L.append(
            f"| {r['k']} | {r['tokens_marked']:,} | {100 * r['fraction']:.1f}% "
            f"| {100 * r['fraction_without_rulers']:.1f}% | {r['files_touched']:,} |"
        )
    L += ["", f"Most repeated {k}-token spans with ≥ 8 distinct tokens:", ""]
    for x in top:
        L.append(f"- {x['copies']} copies: `{x['starts_with']} …`")
    with open(os.path.join(a.out, "substr.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
