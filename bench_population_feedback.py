#!/usr/bin/env python3
"""bench_population_feedback.py — two ways to tell the engine it has enough cells.

After 661a083 every mechanical defect in division is fixed: the child no longer
inflates (weight norm +0.0% over five generations, was +103.6%), tension across
3→32 cells rises 1.2× rather than 12×, and splits over 400 steps fell 984 → 46.

The population still saturates. Nothing about population size feeds back into
tension, so the engine never learns it has enough cells.

(An earlier note called this "a cliff, no band". That was measured BEFORE
661a083. With the child no longer inflating, the control already softened into
a slope — 32, 32, 31.9, 31.9, 22.0, 13.3, 2.0, 2.0 across the quantile sweep, so
2 of 8 settings land interior. The saturation is real; the word "cliff" was
stale.)

Two candidates, and they are genuinely different things:

  A  load relief    division physically splits the response. Parent and child
                    weights are scaled by 1/√2, so their COMBINED output matches
                    the pre-split parent and each one's tension halves. The
                    state changes; the criterion does not.

  B  normalised     tension is untouched; the bar scales with the population:
     trigger        split when mean(recent) > bar · n_cells / min_cells. A
                    bigger population has to work harder to justify growing.
                    The criterion changes; the state does not.

Neither touches `mitosis.py`. Both are subclasses here, as in QD-6 — only a
winner earns an engine change.

What counts as working: a BAND. A range of calibration quantiles that lands
strictly between the 2 floor and the 32 ceiling, and holds there rather than
drifting to an edge. A single lucky setting is not a band.

    .venv/bin/python bench_population_feedback.py
"""

import argparse
import math

import numpy as np
import torch

from mitosis import MitosisEngine

DIM, HIDDEN, MAX_CELLS = 32, 64, 32
STEPS = 400
SEEDS = 3
QUANTILES = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
TAIL = 100                  # cell count is sampled over the last this-many steps


class LoadReliefEngine(MitosisEngine):
    """Arm A — splitting divides the response between parent and child."""

    def split_cell(self, cell):
        event = super().split_cell(cell)
        if event is None:
            return None
        child = self.cells[-1]
        with torch.no_grad():
            for c in (cell, child):
                for p in c.mind.parameters():
                    p.mul_(1.0 / math.sqrt(2.0))
        return event


class NormalisedTriggerEngine(MitosisEngine):
    """Arm B — the bar scales with the population.

    `exponent` sets how hard the brake is. 1.0 is linear: going from 2 cells to
    4 demands twice the tension, which is aggressive and holds the population
    near the floor. 0.5 scales with the square root, so the same growth demands
    only 1.41x. The exponent is the knob this arm introduces, so it is measured
    rather than assumed.
    """

    def __init__(self, *a, exponent=1.0, **kw):
        super().__init__(*a, **kw)
        self.pop_exponent = exponent

    def _check_splits(self):
        base = self.split_threshold
        ratio = len(self.cells) / max(self.min_cells, 1)
        self.split_threshold = base * (ratio ** self.pop_exponent)
        try:
            return super()._check_splits()
        finally:
            self.split_threshold = base


def _b(exp):
    def make(**kw):
        return NormalisedTriggerEngine(exponent=exp, **kw)
    return make


ARMS = (("none (control)", MitosisEngine),
        ("A load relief", LoadReliefEngine),
        ("B normalised n^1.0", _b(1.0)),
        ("B normalised n^0.5", _b(0.5)),
        ("B normalised n^0.25", _b(0.25)))


def run(cls, quantile, seed, steps, noise=0.05):
    torch.manual_seed(seed)
    eng = cls(input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
              initial_cells=2, max_cells=MAX_CELLS, noise_scale=noise)
    xs = [torch.randn(1, DIM) for _ in range(8)]
    eng.calibrate_split_threshold(xs, quantile=quantile)
    counts = []
    for i in range(steps):
        eng.process(xs[i % len(xs)])
        if i >= steps - TAIL:
            counts.append(len(eng.cells))
    counts = np.array(counts)
    splits = sum(1 for e in eng.event_log if e.get("type") == "split")
    return counts.mean(), counts.std(), splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    print(f"\n  집단 되먹임 — floor {2} · ceiling {MAX_CELLS} · "
          f"{args.seeds} seeds × {args.steps} steps")
    print("  working = lands strictly between the floor and the ceiling, "
          "across a RANGE of settings\n")

    results = {}
    for name, cls in ARMS:
        print(f"  {name}")
        print(f"    {'quantile':>9} {'cells (mean)':>13} {'sd over last 100':>18} "
              f"{'splits':>8}")
        band = 0
        for q in QUANTILES:
            ms, sds, sps = [], [], []
            for s in range(args.seeds):
                m, sd, sp = run(cls, q, s, args.steps)
                ms.append(m)
                sds.append(sd)
                sps.append(sp)
            m, sd, sp = np.mean(ms), np.mean(sds), np.mean(sps)
            interior = 2.5 < m < MAX_CELLS - 0.5
            band += int(interior)
            mark = "  ← interior" if interior else ""
            print(f"    {q:>9} {m:>13.1f} {sd:>18.2f} {sp:>8.0f}{mark}")
        results[name] = band
        print(f"    band width: {band}/{len(QUANTILES)} settings land interior\n")

    print("  ── reading " + "─" * 56)
    for name, band in results.items():
        verdict = "band" if band >= 2 else ("one lucky setting" if band == 1 else "cliff")
        print(f"    {name:<18} {band}/{len(QUANTILES)} interior → {verdict}")
    print()


if __name__ == "__main__":
    main()
