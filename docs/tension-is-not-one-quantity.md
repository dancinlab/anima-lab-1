# `split_threshold = 0.3` means two different things

Two engines in this repo take a `split_threshold` and both are given **0.3**.
They are not measuring the same thing, and the number is right in one and 8×
too high in the other.

| | quantity | measured peak | bar | fires? |
|---|---|---|---|---|
| `MitosisEngine` (mitosis.py:60) | `(output**2).mean()` — the cell's output magnitude | **0.037** | 0.3 | never |
| `ConsciousnessEngine` (consciousness_engine.py:330) | `((output − population mean)**2).mean()` — the cell's deviation from its peers | **0.500** | 0.3 | yes, grows to 25 cells |

`anima_unified.py:330` constructs the second with `split_threshold=0.3` and it
works. `mitosis.demo()` and every other caller construct the first with the same
0.3 and it has never fired on any path (QD-5, QD-6).

**The constant is shared; the quantity is not.** A reader who sees `0.3` in both
places will assume they mean the same thing. They differ by more than 13× in
their typical range.

## One difference is real, one guess was wrong

`ConsciousnessEngine`'s tension is **population-referenced**: it is the
deviation of a cell's output from the mean over cells. As the population grows
more alike that deviation falls on its own, which is the negative feedback
`MitosisEngine` lacks and which had to be added to its *bar* instead
(0597e24, `split_threshold · n_cells/min_cells`).

A first draft of this note also claimed the population reference makes it
**scale-invariant**, unlike the absolute form. That is false and the check says
so — multiplying the inputs moves both identically:

| input scale | `MitosisEngine` | `ConsciousnessEngine` |
|---|---|---|
| ×1 | 0.0931 | 0.0822 |
| ×3 | 0.8377 | 0.7401 |
| ×10 | 9.3075 | 8.2233 |

Both track input magnitude. `ConsciousnessEngine` is not protected from the
defect that bit `MitosisEngine`; its output magnitudes simply happen to land
near the bar it was given. Change its `cell_dim`, `hidden_dim` or input scale
and the same silent failure is available.

The reachability guard added in `mitosis.py` (`_check_threshold_reachable`)
covers only that engine. `ConsciousnessEngine` has no equivalent.

## What the sweep found, and did not

The defect class this session kept hitting is *values that look like
measurements but are not* — a hash read as a feature, a rule computed and
discarded, a gate that changes nothing, a bar on the wrong scale. All six were
found by accident, so the live code was swept for more:

- **hash → named measurement**: no further instances outside the one fixed.
- **computed-then-discarded**: none beyond `best_rule` and `g`, both fixed.
- **thresholds on incomparable quantities**: this one — the only new finding.

`split_threshold=999.0` and `merge_threshold=0.0` elsewhere are explicit
"disable this" settings with comments saying so, not defects.
