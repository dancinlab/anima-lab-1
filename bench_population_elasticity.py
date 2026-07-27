#!/usr/bin/env python3
"""bench_population_elasticity.py — the a/b band metric measured the wrong thing.

`bench_population_feedback.py` scored arm B by BAND WIDTH: how many calibration
quantiles land the population strictly between the floor and the ceiling. By
that score k=0.25 won 6/8 against k=1.0's 4/8, and the write-up called it the
better arm.

The theory says that is backwards. With the bar scaled as `bar0·(n/2)^k`, the
population settles where the bar meets the typical tension:

    bar0 · (n*/2)^k = T        ⇒        n* = 2·(T/bar0)^(1/k)

So **1/k is the elasticity of the population to the bar**. A small k does not
make the engine better regulated — it makes n* swing harder for the same change
in calibration. k=0.25 moved 25.3 → 3.3 across the quantile sweep while k=1.0
moved 4.0 → 2.3. Width in quantile-space is sensitivity, and sensitivity to a
knob is fragility, not a band.

Two things are measured here, because a prediction that is only algebra is not
a result:

  1. ELASTICITY   d(log n*)/d(log bar), which should come out near 1/k
  2. INVARIANCE   n* should not depend on max_cells, since the ceiling is not
                  in the equation. If n* tracks the ceiling instead, the
                  settling point is an artifact and neither k is regulated.

    .venv/bin/python bench_population_elasticity.py
"""

import argparse
import math

import numpy as np
import torch

from bench_population_feedback import NormalisedTriggerEngine, DIM, HIDDEN

STEPS = 400
SEEDS = 3
TAIL = 100
QUANTILES = (0.5, 0.6, 0.7, 0.75, 0.8)
EXPONENTS = (1.0, 0.5, 0.25)
CEILINGS = (16, 32, 64)


def settle(exponent, quantile, seed, max_cells, steps, noise=0.05):
    torch.manual_seed(seed)
    eng = NormalisedTriggerEngine(input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
                                  initial_cells=2, max_cells=max_cells,
                                  noise_scale=noise, exponent=exponent)
    xs = [torch.randn(1, DIM) for _ in range(8)]
    eng.calibrate_split_threshold(xs, quantile=quantile)
    bar = eng.split_threshold
    counts = []
    for i in range(steps):
        eng.process(xs[i % len(xs)])
        if i >= steps - TAIL:
            counts.append(len(eng.cells))
    return bar, float(np.mean(counts))


def elasticity(exponent, steps, seeds, max_cells=32):
    """Slope of log(n*) against log(bar). Predicted: −1/k."""
    xs_, ys_ = [], []
    for q in QUANTILES:
        bars, ns = [], []
        for s in range(seeds):
            b, n = settle(exponent, q, s, max_cells, steps)
            bars.append(b)
            ns.append(n)
        b, n = float(np.mean(bars)), float(np.mean(ns))
        if b > 0 and n > 0:
            xs_.append(math.log(b))
            ys_.append(math.log(n))
    if len(xs_) < 2:
        return float("nan"), float("nan")
    slope = float(np.polyfit(xs_, ys_, 1)[0])
    swing = max(math.exp(y) for y in ys_) / max(min(math.exp(y) for y in ys_), 1e-9)
    return slope, swing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    print(f"\n  탄력도 — n* = 2·(T/bar)^(1/k) 예측 검증 · "
          f"{args.seeds} seeds × {args.steps} steps")
    print("  민감할수록 보정 손잡이를 조금만 돌려도 집단이 크게 흔들린다\n")
    print(f"  {'k':>6} {'예측 기울기 −1/k':>16} {'측정 기울기':>12} "
          f"{'집단 진폭(최대/최소)':>20}")
    for k in EXPONENTS:
        slope, swing = elasticity(k, args.steps, args.seeds)
        print(f"  {k:>6} {-1.0 / k:>16.2f} {slope:>12.2f} {swing:>20.1f}×")

    print(f"\n  천장 불변성 — n* 는 max_cells 에 의존하면 안 된다 (식에 없으므로)")
    print(f"  {'k':>6} " + " ".join(f"{'max=' + str(c):>10}" for c in CEILINGS)
          + f" {'천장 추종?':>12}")
    for k in EXPONENTS:
        row = []
        for c in CEILINGS:
            ns = [settle(k, 0.7, s, c, args.steps)[1] for s in range(args.seeds)]
            row.append(float(np.mean(ns)))
        tracks = (max(row) / max(min(row), 1e-9)) > 1.6
        print(f"  {k:>6} " + " ".join(f"{v:>10.1f}" for v in row)
              + f" {'yes ⚠' if tracks else 'no':>12}")

    print("\n  기울기가 −1/k 에 가까우면 식이 맞는 것이고,")
    print("  그러면 k 를 줄이는 것은 규제를 개선하는 게 아니라 민감도를 키우는 것.")
    print()


if __name__ == "__main__":
    main()
