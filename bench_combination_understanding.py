#!/usr/bin/env python3
"""bench_combination_understanding.py — can the whole say what it was made of?

"Understanding a combined concept" needs meaning, and this repo has no channel
for meaning. The one source that could have supplied it — data/corpus_v2.txt,
70MB — cannot:

    41.7% of lines are arithmetic drills
    the dialogue portion is 360,398 lines and 190 UNIQUE ones (99.95% duplicate)
    the Korean prose has 870,211 tokens over 2,074 word types, and its top
    bigrams each occur exactly 3000 times — templates, not language

So semantic understanding is not merely unmeasured here; there is nothing for
it to be made of. A weaker but real sense of understanding is still testable
without any semantics:

    a representation understands a combination if the combination alone is
    enough to say what it was combined from.

That is analysability. A bag of parts loses the seam and cannot place it; an
encoding that keeps the join should recover both the parts and where they met.
Three questions, each against an explicit chance level:

  1. PARTS    given pool(AB), identify (A, B) among all N² candidate pairs
  2. SEAM     given pool(AB), identify where A ended — among all split points
  3. ORDER    given pool(AB), tell it from pool(BA)

`bag` cannot do 2 or 3 even in principle — pooling characters discards
adjacency. That is the control: an encoding that provably cannot understand a
combination, to measure the others against.
"""

import argparse
import itertools
import numpy as np

from bench_hash_revival import _unit, DIMS
from bench_emergence_junction import bigrams, pool_sum
from bench_consciousness_universe import ALL_DATA_TYPES
from qualia_sense import sense


def enc_bag(s):
    return pool_sum(list(s))


def enc_bigram(s):
    return pool_sum(bigrams(s))


def enc_qualia(s):
    return np.array(sense(s).vector())


ENCODERS = (("bag", enc_bag), ("bigram", enc_bigram), ("qualia_sense", enc_qualia))


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def q1_parts(words, enc):
    """Identify (A,B) from the joined representation — via the PARTS.

    The candidate table is built from enc(a) + enc(b), never from enc(a+b).
    Matching a re-encoding of the same string against itself is an identity
    lookup with distance 0, not a test; the first version of this function did
    exactly that and scored 100% for two encodings on no evidence.
    """
    pairs = [(a, b) for a in words for b in words if a != b]
    table = np.array([norm(enc(a) + enc(b)) for a, b in pairs])
    hits = 0
    for i, (a, b) in enumerate(pairs):
        q = norm(enc(a + b))
        hits += int(np.argmin(np.linalg.norm(table - q, axis=1)) == i)
    return hits / len(pairs), 1.0 / len(pairs)


def q2_seam(words, enc, rng):
    """Given the joined string, find where A ended.

    Every split of the same character sequence is compared; only the true one
    corresponds to the pair that was actually joined. An encoding that ignores
    adjacency scores at chance.
    """
    hits = tot = 0
    for a, b in itertools.permutations(words, 2):
        s = a + b
        if len(s) < 3:
            continue
        q = norm(enc(s))
        cands = [(s[:k], s[k:]) for k in range(1, len(s))]
        # the encoding of a split is the pooled sum of its two halves;
        # the true split is the one whose halves reconstruct the whole.
        d = [np.linalg.norm(q - norm(enc(x) + enc(y))) for x, y in cands]
        hits += int(cands[int(np.argmin(d))] == (a, b))
        tot += 1
    return hits / tot, np.mean([1.0 / (len(a + b) - 1)
                                for a, b in itertools.permutations(words, 2)
                                if len(a + b) >= 3])


def q3_order(words, enc):
    """Can AB be told from BA at all?"""
    gaps = [float(np.linalg.norm(norm(enc(a + b)) - norm(enc(b + a))))
            for a, b in itertools.combinations(words, 2)]
    return float(np.mean(gaps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=14)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    words = [n for items in ALL_DATA_TYPES.values() for n in items][:args.words]

    print(f"\n  결합 이해 — {len(words)} words, {len(words) * (len(words) - 1)} ordered pairs")
    print("  understanding here = the combination alone says what it was made of\n")
    print(f"  {'encoding':<14} {'① parts':>9} {'chance':>8} "
          f"{'② seam':>9} {'chance':>8} {'③ order gap':>12}")

    for name, enc in ENCODERS:
        p, pc = q1_parts(words, enc)
        s, sc = q2_seam(words, enc, rng)
        o = q3_order(words, enc)
        print(f"  {name:<14} {p:>8.1%} {pc:>8.2%} {s:>8.1%} {sc:>8.1%} {o:>12.4f}")

    print("\n  ① identify (A,B) from the joined representation")
    print("  ② identify where A ended — needs adjacency, not just contents")
    print("  ③ 0.0000 means AB and BA are the same object to this encoding")
    print()


if __name__ == "__main__":
    main()
