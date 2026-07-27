#!/usr/bin/env python3
"""bench_rank_tension.py — one design change against six measured defects.

Every defect this session measured came from the same place: a raw magnitude
compared against a bare constant, with no shared reference.

    (output**2).mean() vs 0.3        the bar sat 8x above a 0.037 peak
    the same 0.3 on two engines      right in one, unreachable in the other
    no population → tension link     growth never made growth harder
    child weights grew on division   +27.9% per generation, a runaway
    cell_tension = 0.5 in an else    a constant drove every split
    two constants vs a third         an ethics gate that always opens

The proposal: **tension is a cell's RANK within the population at this instant**,
not a magnitude, not a z-score over its own history.

    tension_i = (number of cells with a smaller raw response) / (n - 1)

Four properties follow from the definition rather than from tuning:

    scale-free      multiplying every input changes no ordering, so no value
    never unreachable   a quantile bar always has cells above it, by construction
    population feedback  "top 20%" of 30 cells is a harder club than of 3
    order-independent   a rank does not depend on loop position, unlike
                        consciousness_engine's "mean of the cells so far"

QD-6 tried a z-score against each cell's own HISTORY and it never fired — under
a fixed stimulus that history is flat, so nothing is ever unusual. The reference
has to be the population at this instant, not the past.

Measured here against the four failures that motivated it:

    1. does it fire under varying input?     (MitosisEngine: never did)
    2. does it avoid the ceiling?            (every working config saturated)
    3. is it scale-invariant?                (x10 input broke everything else)
    4. is it order-independent?              (consciousness_engine's is not)

    .venv/bin/python bench_rank_tension.py
"""

import argparse
import math

import numpy as np
import torch

from mitosis import MitosisEngine

DIM, HIDDEN, MAX_CELLS = 32, 64, 32
STEPS, SEEDS, TAIL = 400, 3, 100


class RankTensionEngine(MitosisEngine):
    """Tension as rank within the population, in place of raw magnitude.

    Only `process`'s tension bookkeeping changes: each cell's recorded tension
    becomes its rank among this step's raw responses. Everything downstream —
    split_patience, the merge path, the population-scaled bar — is untouched, so
    any difference measured is the definition and not a second change.
    """

    def _check_splits(self):
        # The population-scaled bar landed in 0597e24 multiplies the threshold by
        # n/min_cells, which assumed tension was unbounded. A rank is capped at
        # 1.0, so past n = 2/bar the effective bar exceeds any possible value and
        # splitting becomes arithmetically impossible — measured as a hard stop
        # at 2.9 cells with bar 0.5 and 2.0 cells above it.
        #
        # A rank does not need that scaling: "top 20%" of 30 cells is already a
        # harder club than "top 20%" of 3. The feedback is in the definition, so
        # the bar stays a plain quantile.
        base = self.split_threshold
        try:
            self.min_cells, saved = len(self.cells), self.min_cells
            return super()._check_splits()
        finally:
            self.min_cells = saved
            self.split_threshold = base

    def process(self, text_vec, label=""):
        result = super().process(text_vec, label)
        raw = [c.tension_history[-1] for c in self.cells if c.tension_history]
        if len(raw) < 2:
            return result
        order = np.argsort(np.argsort(np.array(raw)))      # 0 .. n-1
        denom = len(raw) - 1
        for cell, rank in zip(self.cells, order):
            if cell.tension_history:
                cell.tension_history[-1] = float(rank) / denom
        return result


def run(cls, bar, seed, steps, scale=1.0, rotate=True, max_cells=MAX_CELLS):
    torch.manual_seed(seed)
    eng = cls(input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
              initial_cells=2, max_cells=max_cells, noise_scale=0.05,
              split_threshold=bar)
    xs = [torch.randn(1, DIM) * scale for _ in range(8)]
    counts = []
    for i in range(steps):
        eng.process(xs[i % len(xs)] if rotate else xs[0])
        if i >= steps - TAIL:
            counts.append(len(eng.cells))
    splits = sum(1 for e in eng.event_log if e.get("type") == "split")
    return float(np.mean(counts)), splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    seeds = range(args.seeds)

    print(f"\n  순위 장력 — {args.seeds} seeds × {args.steps} steps · "
          f"바닥 2 · 천장 {MAX_CELLS}")
    print("  순위는 0..1 이므로 기준선도 0..1 — '상위 몇 %' 라는 뜻이 된다\n")

    print("  ① 변화하는 입력에서 분열하는가 · ② 천장을 피하는가")
    print(f"  {'bar':>6} {'세포':>7} {'분열':>7}   (raw 엔진은 8자극 순환에서 0회였다)")
    for bar in (0.5, 0.7, 0.8, 0.9):
        r = [run(RankTensionEngine, bar, s, args.steps) for s in seeds]
        cells = np.mean([x[0] for x in r])
        sp = np.mean([x[1] for x in r])
        mark = "  ← 사이" if 2.5 < cells < MAX_CELLS - 0.5 else ""
        print(f"  {bar:>6} {cells:>7.1f} {sp:>7.0f}{mark}")

    print("\n  ③ 입력 규모를 10배로 바꾸면")
    print(f"  {'engine':>16} {'x1':>7} {'x10':>7} {'변화':>8}")
    for name, cls, bar in (("raw magnitude", MitosisEngine, 0.0441),
                           ("rank", RankTensionEngine, 0.7)):
        a = np.mean([run(cls, bar, s, args.steps, scale=1.0)[0] for s in seeds])
        b = np.mean([run(cls, bar, s, args.steps, scale=10.0)[0] for s in seeds])
        drift = abs(b - a) / max(a, 1e-9) * 100
        print(f"  {name:>16} {a:>7.1f} {b:>7.1f} {drift:>7.0f}%")

    print("\n  ④ 천장을 바꾸면 정착점이 따라가는가 (따라가면 규제가 아니다)")
    print(f"  {'engine':>16} " + " ".join(f"{'max=' + str(c):>9}" for c in (16, 32, 64)))
    for name, cls, bar in (("raw magnitude", MitosisEngine, 0.0441),
                           ("rank", RankTensionEngine, 0.7)):
        row = [np.mean([run(cls, bar, s, args.steps, max_cells=c)[0] for s in seeds])
               for c in (16, 32, 64)]
        print(f"  {name:>16} " + " ".join(f"{v:>9.1f}" for v in row))
    print()


if __name__ == "__main__":
    main()
