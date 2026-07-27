#!/usr/bin/env python3
"""bench_emergence_junction.py — a piece that exists only in the whole.

QD-8 found combination to be pure composition, but that test used
`qualia_sense`, which scores `AB` against `BA` at exactly 0.0000. An
order-blind encoding cannot show emergence from joining, because the join is
invisible to it. The null result was foreordained.

A bigram-pooled encoding is different, and the difference is provable rather
than hoped for. Pooling is a sum of one unit direction per token, so:

    bag:      tokens(AB) = chars(A) + chars(B)
              pool(AB) = pool(A) + pool(B)                    exactly
              → the whole IS the sum of its parts, by construction

    bigram:   tokens(AB) = bigrams(A) + [JUNCTION] + bigrams(B)
              pool(AB) = pool(A) + pool(B) + unit(junction)
              → one unit direction present in the whole and in NEITHER part

The junction is the last character of A followed by the first character of B.
It is a real token that neither part contains. That is the emergence candidate:
not a bigger number, a piece that did not exist before the join.

Three things have to hold for it to count:
  1. the residual is exactly unit(junction) — verified numerically, not assumed;
  2. the junction is genuinely absent from both parts;
  3. it survives the pipeline — the engine that swallowed every other signal
     this session either carries it or does not.

Requires torch for step 3 — run under the repo .venv.
"""

import argparse
import hashlib
import itertools
import math
import numpy as np

from bench_hash_revival import _unit, DIMS
from bench_consciousness_universe import ALL_DATA_TYPES

ENGINE_STEPS = 300
# 18 stimuli give 153 pairs. The first run used 8 words / 28 pairs, which is
# too few to resolve a 5-point effect.
WORDS = tuple([n for items in ALL_DATA_TYPES.values() for n in items][:18])


def bigrams(s):
    """Boundary-padded bigrams: ^ab$ → ['^a', 'ab', 'b$'].

    Padding is not decoration. Without it a one-character string has no bigrams
    at all, and the previous fallback (`or list(s)`) substituted CHARACTER
    tokens into a bigram pool — mixing token types, which broke the junction
    algebra for 서예+공 and 서예+용 with an error of exactly 1.0. With
    boundaries every string has bigrams and the algebra closes.

    Joining then both creates and destroys:
        pool(AB) = pool(A) + pool(B) + unit(a_last·b_first)
                                     − unit(a_last·$) − unit(^·b_first)
    One token appears that was in neither part; two tokens that were in the
    parts are gone from the whole. That is a stronger statement than "the whole
    is more than its parts" — the whole is not a superset of them.
    """
    t = "^" + s + "$"
    return [t[i:i + 2] for i in range(len(t) - 1)]


def pool_sum(tokens):
    """Unnormalised pooled sum — one unit direction per token."""
    return np.sum([_unit(t) for t in tokens], axis=0) if tokens else np.zeros(DIMS)


def junction(a, b):
    return a[-1] + b[0]


def lost_tokens(a, b):
    """The two boundary bigrams the join destroys."""
    return [a[-1] + "$", "^" + b[0]]


def expected_residual(a, b):
    return _unit(junction(a, b)) - sum(_unit(t) for t in lost_tokens(a, b))


def step1_residual_is_the_junction():
    print("\n  ① 잔차 = 새로 생긴 이음쌍 − 사라진 경계쌍 2개 · 수치 검증 " + "─" * 6)
    print(f"  {'pair':<20} {'junction':>10} {'|residual - unit(junction)|':>29} "
          f"{'novel?':>8}")
    worst = 0.0
    novel_all = True
    for a, b in itertools.combinations(WORDS, 2):
        j = junction(a, b)
        resid = pool_sum(bigrams(a + b)) - pool_sum(bigrams(a)) - pool_sum(bigrams(b))
        err = float(np.linalg.norm(resid - expected_residual(a, b)))
        novel = j not in bigrams(a) and j not in bigrams(b)
        novel_all &= novel
        worst = max(worst, err)
        if (a, b) in list(itertools.combinations(WORDS, 2))[:6]:
            print(f"  {a + '+' + b:<20} {j:>10} {err:>29.2e} "
                  f"{'yes' if novel else 'NO':>8}")
    print(f"  … worst error across all {len(list(itertools.combinations(WORDS, 2)))} "
          f"pairs: {worst:.2e}")
    print(f"  junction absent from both parts in every pair: "
          f"{'yes' if novel_all else 'NO'}")
    return worst, novel_all


def step2_how_much_does_it_move():
    print("\n  ② 그 새 조각이 얼마나 움직이나 — 부품 간 차이 대비 " + "─" * 12)
    print(f"  {'encoding':<10} {'mean |residual|':>16} {'mean |A-B|':>12} "
          f"{'residual / |A-B|':>18}")
    out = {}
    for name, tok in (("bag", list), ("bigram", bigrams)):
        rs, ds = [], []
        for a, b in itertools.combinations(WORDS, 2):
            pa, pb = pool_sum(tok(a)), pool_sum(tok(b))
            resid = pool_sum(tok(a + b)) - pa - pb
            rs.append(float(np.linalg.norm(resid)))
            ds.append(float(np.linalg.norm(pa - pb)))
        out[name] = (float(np.mean(rs)), float(np.mean(ds)))
        print(f"  {name:<10} {np.mean(rs):>16.4f} {np.mean(ds):>12.4f} "
              f"{np.mean(rs) / np.mean(ds):>18.3f}")
    print("  bag's residual is 0 by construction — pooling chars is exactly additive.")
    return out


def encode_for_engine(s, dim):
    """bigram-pooled, normalised, tiled to the engine's input width."""
    v = pool_sum(bigrams(s))
    n = np.linalg.norm(v)
    v = (v / n if n > 1e-12 else v).astype(np.float32)
    reps = int(math.ceil(dim / len(v)))
    return np.tile(v, reps)[:dim]


def step3_does_it_survive(steps):
    try:
        import torch
        from bench_mitosis_calibration import calibrate, make_engine
        from bench_psi_mitosis import stimulus_draw, DIM
    except ImportError as exc:
        print(f"\n  (③ skipped — {exc})")
        return None

    derived, _ = calibrate(stimulus_draw())

    def settle(s):
        x = torch.tensor(encode_for_engine(s, DIM)).unsqueeze(0)
        eng = make_engine("absolute", derived, 0)
        for _ in range(steps):
            eng.process(x)
        st = torch.stack([c.hidden.squeeze(0) for c in eng.cells]).mean(dim=0)
        return st.detach().numpy()

    print("\n  ③ 그 조각이 엔진까지 살아남나 " + "─" * 30)
    S = {w: settle(w) for w in WORDS}

    def offline(p, a, b):
        d = S[b] - S[a]
        dn = float(np.linalg.norm(d))
        if dn < 0.05:
            return None
        t = float(np.dot(p - S[a], d) / dn ** 2)
        return float(np.linalg.norm(p - (S[a] + t * d)) / dn)

    real, ablated, ctrl = [], [], []
    for a, b in itertools.combinations(WORDS, 2):
        r = offline(settle(a + b), a, b)
        if r is None:
            continue
        # ablation: same string, junction bigram deleted from the pool
        toks = [t for t in bigrams(a + b) if t != junction(a, b)]
        v = pool_sum(toks)
        n = np.linalg.norm(v)
        v = (v / n if n > 1e-12 else v).astype(np.float32)
        x = torch.tensor(np.tile(v, int(math.ceil(DIM / len(v))))[:DIM]).unsqueeze(0)
        eng = make_engine("absolute", derived, 0)
        for _ in range(steps):
            eng.process(x)
        st = torch.stack([c.hidden.squeeze(0) for c in eng.cells]).mean(dim=0)
        ab = offline(st.detach().numpy(), a, b)
        if ab is None:
            continue          # keep real/ablated aligned — the test is paired
        real.append(r)
        ablated.append(ab)
        for c in WORDS:
            if c not in (a, b):
                cc = offline(S[c], a, b)
                if cc is not None:
                    ctrl.append(cc)

    real, ablated, ctrl = map(np.array, (real, ablated, ctrl))
    print(f"  {'':<28} {'mean':>9} {'n':>5}")
    print(f"  {'combination A+B':<28} {real.mean() * 100:>8.1f}% {len(real):>5}")
    print(f"  {'same, junction ablated':<28} {ablated.mean() * 100:>8.1f}% {len(ablated):>5}")
    print(f"  {'unrelated word C (control)':<28} {ctrl.mean() * 100:>8.1f}% {len(ctrl):>5}")

    # PAIRED test. Each pair contributes both a real and an ablated measurement
    # of the SAME combination, so the per-pair difference is the statistic. An
    # unpaired standard error throws that pairing away and understates the
    # effect — the first run of this bench did exactly that and read z = +1.74
    # where the paired test on the same design reads far higher.
    diff = real - ablated
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    z = float(diff.mean() / se) if se > 0 else float("nan")
    print(f"\n  junction contributes: {diff.mean() * 100:+.2f} ± {se * 100:.2f} "
          f"percentage points  ·  paired z = {z:+.2f}")
    print(f"  pairs moving in the predicted direction: "
          f"{int((diff > 0).sum())}/{len(diff)}  (chance would be half)")
    print("  (ablation is the test — if deleting the one novel token changes")
    print("   nothing, the junction was never carried.)")
    return float(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=ENGINE_STEPS)
    ap.add_argument("--no-engine", action="store_true")
    args = ap.parse_args()

    print("\n  창발 후보 — 전체에만 있고 부품에는 없는 조각")
    err, novel = step1_residual_is_the_junction()
    step2_how_much_does_it_move()
    z = None if args.no_engine else step3_does_it_survive(args.steps)

    print("\n  ── reading " + "─" * 58)
    print(f"  residual matches the algebra exactly: {'yes' if err < 1e-9 else 'no'} "
          f"(worst {err:.1e})")
    print(f"  junction absent from both parts     : {'yes' if novel else 'no'}")
    if z is not None:
        print(f"  survives to the engine state        : "
              f"{'yes' if abs(z) > 2 else 'no'} (z = {z:+.2f})")
    print()


if __name__ == "__main__":
    main()
