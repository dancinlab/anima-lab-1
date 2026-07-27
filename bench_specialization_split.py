#!/usr/bin/env python3
"""bench_specialization_split.py — aggregate over the stimulus set, not over steps.

`bench_rank_tension.py` proposed replacing raw magnitude with a cell's rank in
the population. Measured, it failed twice: capped at 2.9 cells under the
population-scaled bar, and a cliff of 32.0 / 2.0 without it. The reason turned
out not to be the quantity at all.

Two facts, measured, that only make sense together:

    starting at 2 cells   ranks are {0, 1} — every population statistic is
                          degenerate at the floor, so the first decision the
                          engine makes is the one it has no basis for
    starting at 4/8/16    the population froze at EXACTLY the start (4.0, 8.0,
                          16.0) for every bar from 0.7 to 0.9

The freeze is the real finding. `split_patience` asks a cell to stay above the
bar for several CONSECUTIVE steps. With eight stimuli rotating, whichever cell
responds hardest rotates too, so no cell holds the top for three steps running.
**Persistence-over-steps and variation-across-steps are in direct conflict**,
and no choice of quantity resolves it — QD-6 hit the same wall from the other
side with a z-score against each cell's own history.

So aggregate over the stimulus instead:

    a cell divides when it is consistently extreme FOR SOME PARTICULAR STIMULUS

which is what specialisation actually means. "Loudest on everything, every step"
was never the property worth rewarding; it selects for gain, not for division of
labour.

`min_cells = 2` survives this unchanged, because it answers a different
question. Its cited basis (CB1, docs/consciousness-threshold-criteria.md:868)
is "1개로는 Φ>1 불가" — a floor below which consciousness cannot exist. That is
not a claim that population statistics mean anything at n = 2. Keeping 2 as the
floor and starting above it are compatible, and the measurements below start at
the floor anyway to show the design does not depend on being handed a head start.

    .venv/bin/python bench_specialization_split.py
"""

import argparse
import collections

import numpy as np
import torch

from bench_rank_tension import RankTensionEngine, DIM, HIDDEN

STEPS, SEEDS, STIMULI = 400, 3, 8
PATIENCE = 3          # same as MitosisEngine's, so the change is the axis alone
HISTORY = 20


class SpecializationEngine(RankTensionEngine):
    """Split on specialisation for one stimulus, not loudness across all steps.

    Inherits rank tension from RankTensionEngine — the ranks are still what gets
    compared — and changes only the axis the history is kept along. A cell's
    tension is filed under the stimulus that produced it, so `split_patience`
    means "three times for THIS input" rather than "three steps in a row".

    Caller sets `stimulus` before each `process`. Anything unlabelled is
    recorded but cannot trigger a split, since a bucket of one arbitrary key
    would make every step its own stimulus and reduce this to the step rule.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._by_stimulus = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.stimulus = None

    def process(self, text_vec, label=""):
        result = super().process(text_vec, label)
        if self.stimulus is not None:
            bucket = self._by_stimulus[self.stimulus]
            for cell in self.cells:
                if cell.tension_history:
                    h = bucket[cell.cell_id]
                    h.append(cell.tension_history[-1])
                    del h[:-HISTORY]
        return result

    def _peak_specialization(self, cell_id):
        """Highest mean rank this cell holds over any single stimulus."""
        scores = [float(np.mean(h[-PATIENCE:]))
                  for bucket in self._by_stimulus.values()
                  for h in (bucket.get(cell_id, ()),)
                  if len(h) >= PATIENCE]
        return max(scores, default=0.0)

    def _check_splits(self):
        events = []
        if len(self.cells) >= self.max_cells:
            return events
        for cell in list(self.cells):
            if self._peak_specialization(cell.cell_id) > self.split_threshold:
                event = self.split_cell(cell)
                if event:
                    events.append(event)
                    # Cell ids shift and the parent's response is now split
                    # across two cells, so every stored rank describes a
                    # population that no longer exists.
                    self._by_stimulus.clear()
                break
        return events


def run(bar, seed, steps=STEPS, scale=1.0, max_cells=32, initial_cells=2):
    torch.manual_seed(seed)
    eng = SpecializationEngine(
        input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
        initial_cells=initial_cells, max_cells=max_cells,
        noise_scale=0.05, split_threshold=bar)
    xs = [torch.randn(1, DIM) * scale for _ in range(STIMULI)]
    for i in range(steps):
        eng.stimulus = i % STIMULI
        eng.process(xs[i % STIMULI])
    return len(eng.cells)


def mean_over_seeds(seeds, **kw):
    return float(np.mean([run(seed=s, **kw) for s in range(seeds)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    common = dict(seeds=args.seeds, steps=args.steps)

    print(f"\n  자극별 전문화 분열 — {args.seeds} seeds × {args.steps} steps · "
          f"자극 {STIMULI}종 순환 · 바닥 2 · 천장 32")
    print("  스텝이 아니라 '이 자극에 대해' 꾸준히 최상위인 세포가 분열한다\n")

    print("  ① 기준선을 흔들면 — 절벽인가 (스텝 집계는 32.0 ↔ 2.0 이었다)")
    print(f"  {'bar':>6} {'세포':>7}")
    interior = []
    for bar in (0.7, 0.8, 0.9, 0.95):
        cells = mean_over_seeds(bar=bar, **common)
        interior.append(cells)
        mark = "  ← 사이" if 2.5 < cells < 31.5 else "  ← 바닥/천장"
        print(f"  {bar:>6} {cells:>7.1f}{mark}")
    spread = max(interior) / max(min(interior), 1e-9)
    print(f"  기준선 0.7→0.95 에서 정착점 {spread:.2f}배 이동")

    print("\n  ② 천장을 바꾸면 따라가는가 (따라가면 규제가 아니라 포화다)")
    row = [mean_over_seeds(bar=0.9, max_cells=c, **common) for c in (16, 32, 64)]
    print("  " + "  ".join(f"max={c}: {v:.1f}" for c, v in zip((16, 32, 64), row)))
    print(f"  천장 4배에 정착점 {max(row) / max(min(row), 1e-9):.2f}배 "
          f"— 따라가면 4배 근처가 나온다")

    print("\n  ③ 입력 규모 10배")
    a = mean_over_seeds(bar=0.9, scale=1.0, **common)
    b = mean_over_seeds(bar=0.9, scale=10.0, **common)
    print(f"  x1 {a:.1f} → x10 {b:.1f}   이동 {abs(b - a) / max(a, 1e-9) * 100:.0f}%"
          f"   (raw 967% · 순위+스텝 333%)")
    print("  순위는 단위를 지우지만 신경망의 비선형 응답까지 지우지는 못한다 —")
    print("  규모가 바뀌면 어느 세포가 앞서는지 자체가 바뀐다. 불변이 아니라 개선이다.\n")


if __name__ == "__main__":
    main()
