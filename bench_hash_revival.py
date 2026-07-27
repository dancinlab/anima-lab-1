#!/usr/bin/env python3
"""bench_hash_revival.py — is the hash beyond saving, or applied wrong?

QD-7 showed sha256 of a WHOLE string carries nothing: over 1600 samples the
distance between two hashed words sits exactly at the random null (z = +0.60),
and repetition never makes `서예서예` any more like `서예` than `빅뱅` is.

That is a fact about hashing the whole string, not about hashing. A hash
applied to PIECES and pooled is a different object: each piece contributes a
fixed random direction, so two strings sharing pieces share directions. The
structure comes back for free, and it is compositional by construction rather
than stored.

Four granularities, same 8-dimensional output, same measurements:

  whole    sha256(s)                       — what the bench shipped with
  bag      Σ over characters               — shared characters ⇒ shared direction
  bigram   Σ over adjacent pairs           — adjacency survives pooling
  pos      Σ over (index, character)       — position survives pooling

`qualia_sense` is carried alongside as the hand-built comparison. It has one
structural limit these can be tested against: it is a bag of character
statistics, so `AB` and `BA` are bit-identical to it — measured 0.0000 in
bench_concept_combination.py. A positional hash should not have that limit.
"""

import argparse
import hashlib
import itertools
import math
import numpy as np

from qualia_sense import sense, FEATURE_NAMES

DIMS = 8
WORDS = ("서예", "만다라", "검은사각형", "빅뱅", "공", "용", "단맛", "빨강")
STEM_GROUPS = (("서예", "서예체", "서예가"), ("만다라", "모래만다라"))


def _unit(token):
    """A token → a fixed direction on the unit sphere, via sha256."""
    h = hashlib.sha256(token.encode()).digest()
    v = np.frombuffer(h[:DIMS], dtype=np.uint8).astype(np.float64) / 255.0 - 0.5
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def h_whole(s):
    return _unit(s)


def _pool(tokens):
    if not tokens:
        return np.zeros(DIMS)
    v = np.sum([_unit(t) for t in tokens], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def h_bag(s):
    return _pool(list(s))


def h_bigram(s):
    return _pool([s[i:i + 2] for i in range(len(s) - 1)] or list(s))


def h_pos(s):
    return _pool([f"{i}:{c}" for i, c in enumerate(s)])


def h_qualia(s):
    v = np.array(sense(s).vector())
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


VARIANTS = (("whole", h_whole), ("bag", h_bag), ("bigram", h_bigram),
            ("pos", h_pos), ("qualia_sense", h_qualia))


def d(f, a, b):
    return float(np.linalg.norm(f(a) - f(b)))


def null_distance(f, trials=2000, seed=0):
    """Distance between two unrelated strings under this encoding."""
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(trials):
        a = "".join(chr(rng.integers(0xAC00, 0xD7A3)) for _ in range(rng.integers(1, 6)))
        b = "".join(chr(rng.integers(0xAC00, 0xD7A3)) for _ in range(rng.integers(1, 6)))
        ds.append(d(f, a, b))
    return float(np.mean(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=40)
    args = ap.parse_args()

    print("\n  해시 회생 — is the hash beyond saving?")
    print(f"  all encodings normalised to the unit sphere, {DIMS} dims, "
          f"so distances are comparable\n")

    print(f"  {'encoding':<14} {'null':>7} {'stem kin':>9} {'kin/null':>9} "
          f"{'AB vs BA':>9} {'repeat mono':>12} {'saturates':>10}")

    for name, f in VARIANTS:
        null = null_distance(f)

        kin = np.mean([d(f, g[i], g[j]) for g in STEM_GROUPS
                       for i in range(len(g)) for j in range(i + 1, len(g))])

        order = np.mean([d(f, a + b, b + a) for a, b in itertools.combinations(WORDS, 2)])

        seq = [d(f, "서예", "서예" * n) for n in range(2, args.repeats + 1)]
        mono = all(y >= x - 1e-9 for x, y in zip(seq, seq[1:]))
        tail_step = np.mean([abs(y - x) for x, y in zip(seq[-10:], seq[-9:])])

        print(f"  {name:<14} {null:>7.3f} {kin:>9.3f} {kin / null:>9.2f} "
              f"{order:>9.4f} {'yes' if mono else 'no':>12} "
              f"{('yes ' + f'{tail_step:.4f}') if tail_step < 1e-3 else 'no':>10}")

    print("\n  kin/null < 1  = related words land closer than unrelated ones (structure)")
    print("  AB vs BA > 0  = the encoding can tell 'A then B' from 'B then A'")
    print("  saturates     = the encoding stops distinguishing further repeats")

    print("\n  ── what the whole-string hash cannot do, and the pooled ones can " + "─" * 3)
    for a, b in (("서예", "서예체"), ("만다라", "모래만다라"), ("서예", "빅뱅")):
        row = "  ".join(f"{n}={d(f, a, b):.3f}" for n, f in VARIANTS)
        print(f"    {a + ' ↔ ' + b:<22} {row}")
    print()


if __name__ == "__main__":
    main()
