#!/usr/bin/env python3
"""bench_concept_combination.py — is a combination more than its parts?

The sharp form of "memorisation or not": if the system merely stores pieces,
then `서예만다라` is *predictable* from `서예` and `만다라`. If something
emerges, part of the combination is explainable by neither.

Two nulls, one per layer, both stated before measuring:

  string layer   qualia_sense's features are ratios over characters, so
                 concatenation should give a length-weighted mean of the two
                 ratio vectors, with `length` adding. Anything left over is
                 what concatenation CREATES — and the junction between the two
                 words is the only place it can come from.

  engine layer   if the combination is a mix of the parts, its settled state
                 lies on the segment between the two parts' states. The
                 component perpendicular to that segment is the part no mix of
                 A and B can produce. Normalised by |A−B| so it is scale-free.

Order is also a probe: a bag of parts cannot tell `AB` from `BA`.

Requires torch for the engine layer — run under the repo .venv.
"""

import argparse
import itertools
import math
import numpy as np

from qualia_sense import sense, FEATURE_NAMES

# `length` adds; every other feature is a ratio over characters and is
# predicted by a length-weighted mean. Index 0 is `length` — see FEATURE_NAMES.
LENGTH_IDX = FEATURE_NAMES.index("length")
LENGTH_CAP = 16.0                      # the min(n/16, 1) ceiling in qualia_sense

WORDS = ("서예", "만다라", "검은사각형", "빅뱅", "공", "용", "단맛", "빨강")
ENGINE_STEPS = 300


def predict_concat(a, b):
    """The bag-of-parts null: what concatenation would give with no emergence."""
    va, vb = np.array(sense(a).vector()), np.array(sense(b).vector())
    na, nb = len(a), len(b)
    pred = (na * va + nb * vb) / (na + nb)          # ratio features
    pred[LENGTH_IDX] = min((na + nb) / LENGTH_CAP, 1.0)   # length adds
    return pred


def string_layer():
    print("\n  ── string layer · what does joining two words create? " + "─" * 14)
    print(f"  {'pair':<22} {'|actual|':>9} {'|residual|':>11} {'residual %':>11} "
          f"{'AB vs BA':>9}")

    residuals, order_gaps, per_feature = [], [], []
    for a, b in itertools.combinations(WORDS, 2):
        actual = np.array(sense(a + b).vector())
        pred = predict_concat(a, b)
        resid = actual - pred
        rev = np.array(sense(b + a).vector())
        order = float(np.linalg.norm(actual - rev))

        residuals.append(float(np.linalg.norm(resid)))
        order_gaps.append(order)
        per_feature.append(np.abs(resid))
        pct = residuals[-1] / np.linalg.norm(actual) * 100
        print(f"  {a + '+' + b:<22} {np.linalg.norm(actual):>9.4f} "
              f"{residuals[-1]:>11.4f} {pct:>10.1f}% {order:>9.4f}")

    pf = np.mean(per_feature, axis=0)
    print(f"\n  mean |residual| {np.mean(residuals):.4f}  ·  "
          f"mean AB↔BA gap {np.mean(order_gaps):.4f}")
    print("\n  where the residual lives — per measurement:")
    for name, v in sorted(zip(FEATURE_NAMES, pf), key=lambda t: -t[1]):
        blocks = "█" * int(v * 40)
        print(f"    {name:<18} {v:>7.4f} {blocks}")
    return float(np.mean(residuals)), float(np.mean(order_gaps))


def engine_layer(steps):
    try:
        import torch
        from bench_mitosis_calibration import calibrate, make_engine
        from bench_psi_mitosis import stimulus_draw, DIM
    except ImportError as exc:
        print(f"\n  (engine layer skipped — {exc})")
        return None

    derived, _ = calibrate(stimulus_draw())

    def settle(text):
        v = np.array(sense(text).vector(), dtype=np.float32)
        reps = int(math.ceil(DIM / len(FEATURE_NAMES)))
        x = torch.tensor(np.tile(v, reps)[:DIM]).unsqueeze(0)
        eng = make_engine("absolute", derived, 0)
        for _ in range(steps):
            eng.process(x)
        st = torch.stack([c.hidden.squeeze(0) for c in eng.cells]).mean(dim=0)
        return st.detach().numpy()

    print(f"\n  ── engine layer · does the combination leave the A–B line? " + "─" * 8)
    print(f"  {'pair':<22} {'|A-B|':>9} {'off-line':>10} {'off-line %':>11} "
          f"{'mix t':>7}")

    cache, offs = {}, []
    for w in WORDS:
        cache[w] = settle(w)
    for a, b in itertools.combinations(WORDS, 2):
        sa, sb, sab = cache[a], cache[b], settle(a + b)
        d = sb - sa
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            continue
        t = float(np.dot(sab - sa, d) / dn ** 2)        # where along A→B it sits
        onto = sa + t * d
        off = float(np.linalg.norm(sab - onto))
        offs.append(off / dn)
        print(f"  {a + '+' + b:<22} {dn:>9.4f} {off:>10.4f} "
              f"{off / dn * 100:>10.1f}% {t:>7.3f}")

    print(f"\n  mean off-line component: {np.mean(offs) * 100:.1f}% of |A−B|")
    print("  0% = the combination is exactly a mix of its parts (no emergence)")
    return float(np.mean(offs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=ENGINE_STEPS)
    ap.add_argument("--no-engine", action="store_true")
    args = ap.parse_args()

    print("\n  개념 결합 — is a combination more than its parts?")
    resid, order = string_layer()
    off = None if args.no_engine else engine_layer(args.steps)

    print("\n  ── reading " + "─" * 58)
    print(f"  string residual beyond the bag-of-parts null : {resid:.4f}")
    print(f"  order sensitivity (AB vs BA)                 : {order:.4f}")
    if off is not None:
        print(f"  engine off-line component                    : {off * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()
