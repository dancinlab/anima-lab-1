#!/usr/bin/env python3
"""bench_grounded_semantics.py — semantics from a corpus that is actually language.

QD-10/QD-11 concluded that semantic understanding was blocked because
`data/corpus_v2.txt` has no natural language (type/token 0.14–0.24%) and the
only other source, the 170 one-line descriptions, failed a shuffled-label
control. That conclusion was premature: it never looked outside those two
files.

The sibling `anima` repository's own markdown is natural language, written by
people:

    corpus_v2.txt Korean prose   870,211 tokens   2,074 types   0.24%
    anima/**/*.md Korean         1,144,691 tokens  82,873 types  6.50%

Twenty-seven times the lexical richness, and 91% of the 170 concepts have at
least one description word appearing in it three or more times.

This builds real distributional vectors from that corpus and re-asks the goal's
question, with the control that killed the previous attempt applied from the
start:

  concept vector = mean PPMI context vector of its description's words
  combination    = the same over BOTH descriptions' words
  ground truth   = the category a person assigned
  baseline       = shuffled category labels, never the naive 1/n_categories

Centroids exclude the concepts being classified — without that the concept sits
inside its own centroid and the score is leakage.
"""

import argparse
import collections
import math
import pathlib
import pickle
import re

import numpy as np

from bench_semantic_composition import concepts

CACHE = ("/private/tmp/claude-501/-Users-mini-dancinlab-anima-lab-1/"
         "d0396916-1383-4a82-b215-02ece85f6789/scratchpad/ko_corpus.pkl")
CORPUS_DIR = pathlib.Path("/Users/mini/dancinlab/anima")
CONTEXT_VOCAB = 4000
WINDOW = 5
MIN_COUNT = 3
SHUFFLES = 30


def load_corpus(path=None):
    if path:
        sents, freq = [], collections.Counter()
        for line in pathlib.Path(path).open(encoding="utf-8", errors="replace"):
            w = re.findall(r"[가-힣]{2,}", line)
            if len(w) >= 3:
                sents.append(w)
                freq.update(w)
        return sents, freq
    p = pathlib.Path(CACHE)
    if p.exists():
        d = pickle.load(open(p, "rb"))
        return d["sents"], d["freq"]
    sents, freq = [], collections.Counter()
    for f in CORPUS_DIR.rglob("*.md"):
        s = str(f)
        if ".git" in s or "node_modules" in s:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            w = re.findall(r"[가-힣]{2,}", line)
            if len(w) >= 3:
                sents.append(w)
                freq.update(w)
    return sents, freq


def build_vectors(sents, freq, targets):
    """PPMI context vectors for `targets` over the top-CONTEXT_VOCAB words."""
    ctx = [w for w, _ in freq.most_common(CONTEXT_VOCAB)]
    ctx_idx = {w: i for i, w in enumerate(ctx)}
    tgt = [w for w in targets if freq[w] >= MIN_COUNT]
    tgt_idx = {w: i for i, w in enumerate(tgt)}

    M = np.zeros((len(tgt), len(ctx)))
    for s in sents:
        for i, w in enumerate(s):
            ti = tgt_idx.get(w)
            if ti is None:
                continue
            for j in range(max(0, i - WINDOW), min(len(s), i + WINDOW + 1)):
                if j == i:
                    continue
                cj = ctx_idx.get(s[j])
                if cj is not None:
                    M[ti, cj] += 1.0

    total = M.sum()
    if total == 0:
        return {}, tgt
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)                       # PPMI
    n = np.linalg.norm(pmi, axis=1, keepdims=True)
    pmi = pmi / np.where(n < 1e-12, 1.0, n)
    return {w: pmi[i] for w, i in tgt_idx.items()}, tgt


def concept_vec(desc, V):
    ws = [w for w in re.findall(r"[가-힣]{2,}", desc) if w in V]
    if not ws:
        return None
    v = np.mean([V[w] for w in ws], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def grounding(names, labels, vecs):
    same, cross = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(vecs[a] - vecs[b]))
            (same if labels[a] == labels[b] else cross).append(d)
    same, cross = np.array(same), np.array(cross)
    se = math.sqrt(same.var(ddof=1) / len(same) + cross.var(ddof=1) / len(cross))
    return same.mean(), cross.mean(), (cross.mean() - same.mean()) / se


def compose_score(names, labels, vecs, combos):
    """Leave-the-pair-out centroid classification of each combination."""
    hits_same = tot_same = hits_cross = tot_cross = 0
    by_cat = collections.defaultdict(list)
    for n in names:
        by_cat[labels[n]].append(n)

    for (a, b), v in combos.items():
        cents = {}
        for cat, members in by_cat.items():
            m = [vecs[x] for x in members if x not in (a, b)]
            if m:
                c = np.mean(m, axis=0)
                nn = np.linalg.norm(c)
                if nn > 1e-12:
                    cents[cat] = c / nn
        if not cents:
            continue
        got = min(cents, key=lambda k: float(np.linalg.norm(v - cents[k])))
        if labels[a] == labels[b]:
            tot_same += 1
            hits_same += int(got == labels[a])
        else:
            tot_cross += 1
            hits_cross += int(got in (labels[a], labels[b]))
    return (hits_same / max(tot_same, 1), tot_same,
            hits_cross / max(tot_cross, 1), tot_cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=SHUFFLES)
    ap.add_argument("--corpus", default=None,
                    help="read a corpus file instead of the anima markdown tree")
    args = ap.parse_args()

    C = concepts()
    sents, freq = load_corpus(args.corpus)
    src = args.corpus or "anima/**/*.md"
    print(f"\n  근거 코퍼스 — {src} · 문장 {len(sents):,} · 어휘 {len(freq):,}")

    desc_words = {w for _, d in C.values() for w in re.findall(r"[가-힣]{2,}", d)}
    V, tgt = build_vectors(sents, freq, desc_words)
    print(f"  분포 벡터 구축 — 대상 단어 {len(V)}종 × 문맥 {CONTEXT_VOCAB}차원 (PPMI)")

    vecs, labels = {}, {}
    for n, (cat, d) in C.items():
        v = concept_vec(d, V)
        if v is not None:
            vecs[n] = v
            labels[n] = cat
    names = sorted(vecs)
    cats = sorted({labels[n] for n in names})
    print(f"  근거를 얻은 개념 {len(names)}/{len(C)} · 범주 {len(cats)}\n")

    s, x, z = grounding(names, labels, vecs)
    rng = np.random.default_rng(0)
    zs = []
    for _ in range(args.shuffles):
        vals = [labels[n] for n in names]
        rng.shuffle(vals)
        sh = dict(zip(names, vals))
        zs.append(grounding(names, sh, vecs)[2])
    zs = np.array(zs)
    print("  ① 같은 범주가 더 가까운가 — 뒤섞기 대조군 기준")
    print(f"     같은 {s:.4f} · 다른 {x:.4f} · z = {z:+.2f}")
    print(f"     뒤섞은 라벨 z = {zs.mean():+.2f} ± {zs.std():.2f}  →  "
          f"초과 {(z - zs.mean()) / zs.std():+.1f}σ")

    combos = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = concept_vec(C[a][1] + " " + C[b][1], V)
            if v is not None:
                combos[(a, b)] = v
    hs, ns_, hc, nc = compose_score(names, labels, vecs, combos)

    hss, hcs = [], []
    for _ in range(args.shuffles):
        vals = [labels[n] for n in names]
        rng.shuffle(vals)
        sh = dict(zip(names, vals))
        a_, _, b_, _ = compose_score(names, sh, vecs, combos)
        hss.append(a_)
        hcs.append(b_)
    hss, hcs = np.array(hss), np.array(hcs)

    print(f"\n  ② 결합이 올바른 범주로 가는가 — 쌍 제외 중심점, 뒤섞기 대조군 기준")
    print(f"     같은 범주끼리 → 그 범주   {hs:>6.1%}  (n={ns_})   "
          f"뒤섞음 {hss.mean():>6.1%} ± {hss.std():.1%}  →  "
          f"{(hs - hss.mean()) / hss.std():+.1f}σ")
    print(f"     다른 범주끼리 → 부모 적중 {hc:>6.1%}  (n={nc})   "
          f"뒤섞음 {hcs.mean():>6.1%} ± {hcs.std():.1%}  →  "
          f"{(hc - hcs.mean()) / hcs.std():+.1f}σ")
    print()


if __name__ == "__main__":
    main()
