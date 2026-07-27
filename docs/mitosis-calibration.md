# MitosisEngine — why mitosis never fired, and what changed

Threshold-driven cell division has never occurred on any path in this
repository. `mitosis.demo()` prints one `MITOSIS` line and it comes from a
`--- Forced Mitosis Demo ---` block calling `split_cell()` directly; across its
30 threshold-driven steps, zero splits happen (QD-5, QD-6).

Two independent causes, both measured. One is fixed here; the other is
characterised and left for the owner.

## Cause 1 — an absolute bar on a scale-free quantity (fixed)

`tension = (output ** 2).mean()` (mitosis.py:60) is an absolute magnitude, so
it tracks the caller's input size directly, while `split_threshold` is an
absolute constant. Whether a cell ever divides is therefore decided by how
large the caller's vectors happen to be:

| input | norm | peak tension | bar | cells |
|---|---|---|---|---|
| `demo()`'s `text_to_vector` | 0.051 | 0.01 | 1.5 | 2 |
| `qualia_sense` | 3.66 | 0.029 | 0.3 | 2 |
| `torch.randn` — the engine's own default | 6.09 | 0.083 | 0.3 | 2 |
| `randn` × 5 | 32.6 | 2.38 | 0.3 | **32** |

Every scale the repo actually uses falls short — its own default by 3.6×, its
own demo by 150×.

**What changed.** The constant is untouched — replacing 0.3 with another guessed
number is the manipulation CLAUDE.md #2 forbids and would need re-guessing per
input scale. Instead:

- `_check_threshold_reachable()` — after 200 steps, if peak tension never came
  within half the bar, it says so once, with the numbers. A silent failure is
  now audible: *"split_threshold=0.3 is unreachable: peak tension over 200 steps
  was 0.0370 (8x below). Mitosis cannot fire."*
- `calibrate_split_threshold(sample_inputs, quantile=0.9)` — derives the bar
  from the tension this engine actually produces, as a quantile of it. That
  states something stable: *split when tension is in this engine's top decile*,
  which survives a change of input scale.

A quantile rather than mean + k·sd, deliberately. `median + 2·sd` measured
0.0676 against a peak of 0.0702 — a bar only the maximum reaches is never held
for `split_patience` consecutive steps, so it fires exactly as never as the 0.3
it replaced. A quantile fixes the *fraction* of steps above the bar regardless
of distribution shape.

Calibrate on the inputs the engine will really see; a bar derived from a
different distribution does not transfer.

## Cause 2 — the persistence rule forbids division under varying input (open)

`_check_splits` requires `all(t > split_threshold for t in recent)` over
`split_patience` consecutive steps. Measured with a correctly derived bar:

| input | derived bar | cells after 400 steps | splits |
|---|---|---|---|
| one fixed vector | 0.0370 | **31** | 1051 |
| 8 vectors in rotation | 0.0458 | **2** | **0** |

Under rotation, tension exceeds the bar on 50% of steps and the longest
consecutive run is **2**, against the 3 required. That is not chance — a random
50% sequence would produce runs of 3 constantly over 200 steps. Tension follows
the input, so with varying input it oscillates and can never persist.

**The engine can only divide when fed the same thing repeatedly** — the opposite
of novelty-driven division. And once it does fire it saturates: 1051 splits to
31 cells against a 32 ceiling, the narrow band QD-6 measured.

This is left unchanged. Altering the division criterion changes what the engine
means by "sustained tension" and is an owner decision, not the side effect of an
audit — the same line taken with `corpus_v2.txt`. It is on the board.

## Worth knowing before acting on any of this

QD-6 measured that fixing the calibration *does* form a population — 3.1 cells
with a derived bar — and that the population does not improve what the QD series
was chasing: stimulus retention was 0.409 at 2 cells, 0.377 at 3.1, and 0.403 at
31.8. Population size does not move it. These fixes make the engine behave as
designed; they do not make it do more.

---

# Owner said go — the criterion was changed, and the failure moved

`_check_splits` now compares the **mean** of the recent window against the bar
instead of requiring every step of it to clear (`sum(recent)/len(recent)` vs
`all(t > bar)`). That keeps "sustained" and drops "uninterrupted", and matches
`Cell.avg_tension`, which already averages its recent window.

`MitosisC.__init__`'s forced clone growth is also gone. It manufactured cells
the merge logic deletes within ten steps, so `max_cells` only looked like a
starting size; construction now honestly reports 2.

## Division is possible under varying input now — and lands on the ceiling

8 stimuli in rotation, 400 steps:

| quantile | bar | cells | splits |
|---|---|---|---|
| 0.75 | 0.0462 | **31** | 984 |
| 0.80 | 0.0594 | **2** | **0** |
| 0.85 | 0.0603 | 2 | 0 |
| 0.90 | 0.0673 | 2 | 0 |

Before this change every one of these was 2 cells and 0 splits. So the
impossibility is gone. But there is **no band** — between bars of 0.0462 and
0.0594 the population flips from ceiling to floor with nothing in between.

QD-6 pre-registered that "pinning to `max_cells` is the same failure as pinning
to `min_cells`". By that standard **this moved the failure rather than fixing
it**, and saying otherwise would be dishonest.

## Why it always runs to the ceiling — measured

At the shipped `noise_scale=0.014`, **97% of splits are undone**: 984 splits
against 955 merges. Children differ from their parent by too little for their
inter-cell tension to clear `merge_threshold`, so the engine deletes what it
just made. Raising the noise stops the churn and changes nothing about the
outcome:

| noise_scale | cells | splits | merges | undone | mean inter-tension |
|---|---|---|---|---|---|
| 0.014 (shipped) | 31 | 984 | 955 | **97%** | 0.094 |
| 0.05 | 32 | 30 | 0 | 0% | 0.53 |
| 0.15 | 32 | 30 | 0 | 0% | 51.0 |
| 0.40 | 32 | 30 | 0 | 0% | 2373 |

Either the children are erased or they survive and the population saturates
immediately. The deeper reason is visible in the tension as the population
grows:

| step | cells | mean tension per cell |
|---|---|---|
| 0 | 3 | 0.0315 |
| 5 | 9 | 0.1947 |
| 15 | 32 | **0.3881** |
| 150 | 32 | 0.4611 |

**Dividing does not relieve the pressure that caused it — it raises it, 12×
from 3 cells to 32.** Each generation inherits its parent's weights plus noise,
so outputs grow, so `tension = (output**2).mean()` grows, so the split
condition holds harder the more it has already fired. The loop is positively
self-reinforcing and the ceiling is the only thing that stops it.

## The remaining decision

Nothing about population size feeds back into tension. A working population
needs that negative feedback — division has to lower per-cell load, or the
trigger has to normalise by population — and either is a change to what tension
*means*, which is a larger decision than changing when a comparison fires.
That one is recorded, not taken.
