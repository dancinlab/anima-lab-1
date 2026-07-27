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
