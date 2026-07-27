#!/usr/bin/env python3
"""bench_semantic_composition.py — semantics from the one honest source left.

QD-10 found data/corpus_v2.txt has no natural language anywhere, so semantic
understanding looked blocked. But the repo holds a second semantic source that
QD-10 overlooked: `ALL_DATA_TYPES` gives every one of 170 concepts a
human-written one-line Korean description and a human-assigned category.

    "서예":   ("✒️", "호흡과 획")          category 예술
    "만다라": ("☸️", "우주의 설계도")       category 예술
    "매운맛": ("🌶️", "통증의 쾌락")        category 미각

Small, but written by a person about what the thing means — not templated, not
duplicated, not arithmetic. That is enough to ask the goal's question with real
ground truth:

  1. GROUNDING   does a description-based encoding put same-category concepts
                 closer than cross-category ones? (validates the encoder)
  2. FORM ALONE  do the form encodings — the ones used all series — have any
                 category structure at all? (expected: none)
  3. COMPOSITION when two concepts of the SAME category are combined, does the
                 combination stay in that category? when they come from
                 DIFFERENT categories, does it land between them?

(3) is semantic composition with human labels as ground truth: the combination
is understood if what it is made of determines where it belongs.
"""

import argparse
import itertools
import numpy as np

from bench_emergence_junction import bigrams, pool_sum
from bench_consciousness_universe import ALL_DATA_TYPES
from qualia_sense import sense


def concepts():
    """name → (category, description). Descriptions are the semantic source."""
    out = {}
    for cat, items in ALL_DATA_TYPES.items():
        for name, (emoji, desc) in items.items():
            out[name] = (cat, desc)
    return out


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def enc_meaning(name, C):
    """Grounded in the human description, not in the name's spelling."""
    return norm(pool_sum(bigrams(C[name][1])))


def enc_form(name, C):
    return norm(pool_sum(bigrams(name)))


def enc_qualia(name, C):
    return norm(np.array(sense(name).vector()))


ENCODERS = (("meaning (description)", enc_meaning),
            ("form (name bigrams)", enc_form),
            ("form (qualia_sense)", enc_qualia))


def grounding(names, C, enc):
    """same-category distance vs cross-category distance, and the gap's z."""
    same, cross = [], []
    for a, b in itertools.combinations(names, 2):
        d = float(np.linalg.norm(enc(a, C) - enc(b, C)))
        (same if C[a][0] == C[b][0] else cross).append(d)
    same, cross = np.array(same), np.array(cross)
    se = np.sqrt(same.var(ddof=1) / len(same) + cross.var(ddof=1) / len(cross))
    return same.mean(), cross.mean(), (cross.mean() - same.mean()) / se


def category_centroids(names, C, enc):
    cats = {}
    for n in names:
        cats.setdefault(C[n][0], []).append(enc(n, C))
    return {k: norm(np.mean(v, axis=0)) for k, v in cats.items()}


def composition(names, C, enc, cents):
    """Where does a combination land, by human category labels?

    same-category pairs  → does the combination stay in that category?
    cross-category pairs → is it nearest to one of the two parents' categories
                           (rather than to some third, unrelated category)?
    """
    def classify(v):
        return min(cents, key=lambda k: float(np.linalg.norm(v - cents[k])))

    same_hits = same_tot = cross_hits = cross_tot = 0
    for a, b in itertools.combinations(names, 2):
        ca, cb = C[a][0], C[b][0]
        # the combination's description is the two descriptions joined
        v = norm(pool_sum(bigrams(C[a][1] + C[b][1])))
        got = classify(v)
        if ca == cb:
            same_tot += 1
            same_hits += int(got == ca)
        else:
            cross_tot += 1
            cross_hits += int(got in (ca, cb))
    n_cat = len(cents)
    return (same_hits / same_tot, 1.0 / n_cat,
            cross_hits / cross_tot, 2.0 / n_cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    C = concepts()
    names = list(C)
    if args.limit:
        names = names[:args.limit]
    cats = {C[n][0] for n in names}

    print(f"\n  의미 결합 — {len(names)} concepts, {len(cats)} human-assigned categories")
    print("  ground truth = the category a person put each concept in\n")

    print("  ① 무엇이 범주 구조를 갖고 있나 (same vs cross category distance)")
    print(f"  {'encoding':<24} {'same':>8} {'cross':>8} {'gap z':>9} {'structure?':>12}")
    best = None
    for name, enc in ENCODERS:
        s, x, z = grounding(names, C, enc)
        print(f"  {name:<24} {s:>8.4f} {x:>8.4f} {z:>9.2f} "
              f"{'yes' if z > 2 else 'no':>12}")
        if name.startswith("meaning"):
            best = enc

    cents = category_centroids(names, C, best)
    sh, sc, ch, cc = composition(names, C, best, cents)
    print(f"\n  ② 결합이 어디에 속하나 — 설명문 기반 표현, 사람이 매긴 범주로 채점")
    print(f"     같은 범주끼리 결합 → 그 범주에 남는다   "
          f"{sh:>6.1%}   (우연 {sc:.1%})")
    print(f"     다른 범주끼리 결합 → 두 부모 중 하나로   "
          f"{ch:>6.1%}   (우연 {cc:.1%})")
    print()


if __name__ == "__main__":
    main()
