# `split_threshold = 0.3` means two different things — and neither engine reaches it

Two engines take a `split_threshold` and both are given **0.3**. Neither one's
measured tension ever gets there. One of them divides anyway, for a reason that
is not the threshold.

> **This page was published wrong and is corrected here.** The first version
> reported `ConsciousnessEngine` peaking at 0.500 against its 0.3 bar and
> concluded it worked. 0.500 is not a measurement — it is a hardcoded constant
> assigned to one cell every step. Its actual computed tension peaks at 0.037,
> the same 8× shortfall `MitosisEngine` has. The error was reading a peak
> without checking what produced it.

| | quantity | **computed** peak | bar | divides? |
|---|---|---|---|---|
| `MitosisEngine` (mitosis.py:60) | `(output**2).mean()` — output magnitude | 0.037 | 0.3 | never |
| `ConsciousnessEngine` (consciousness_engine.py:364) | deviation from the mean of cells *so far* | **0.037** | 0.3 | yes — 148 times, none of them from tension |

## Where `ConsciousnessEngine`'s divisions actually come from

```python
if outputs:                    # every cell EXCEPT the first of the loop
    cell_tension = ((output - out_stack.mean(dim=0)) ** 2).mean()
else:                          # the FIRST cell — every single step
    cell_tension = 0.5
```

`outputs` is empty only when processing the loop's first cell, so **cell 0
receives the constant 0.5 on every step**. That is not a rare fallback. 0.5 sits
above the 0.3 bar, so cell 0 satisfies `all(t > bar)` after `split_patience`
steps and **divides on a timer**, its history resets, and it repeats.

Measured over 300 steps: 148 splits, and of the tension values that cleared the
bar, **100% were the constant** — 740 of 14,635 recorded values are exactly
0.5000 and nothing computed ever reached 0.3.

```
  cell 0:  0.5000  0.5000  0.5000  0.5000     ← the constant, every step
  cell 1:  0.0237  0.0213  0.0376  0.0297     ← computed
  cell 2:  0.0089  0.0097  0.0170  0.0156
  cell 4:  0.0070  0.0075  0.0067  0.0082
```

**A hardcoded constant named like a measurement, and it is the thing actually
driving the behaviour.** The same defect class as the sha256 features, the
discarded `best_rule`, and the inert gate `g` — this one in the engine
`anima_unified.py` actually runs.

## A second defect in the same expression

The computed branch measures deviation from the mean of **the cells processed so
far in this loop**, not from the population. So tension falls with loop
position — 0.024 at cell 1, 0.007 at cell 4 — because later cells are compared
against more peers. The same cell in a different position gets a different
"tension". It is not a property of the cell.

## Not changed

Removing the `0.5` stops division entirely, exactly as `MitosisEngine` stopped;
making the deviation population-wide changes what tension means. Both are design
decisions, and this is an audit. Annotated at the site so the next reader does
not mistake the constant for a reading — which is what this page did.

## What the sweep found

The defect class this session kept hitting is *values that look like
measurements but are not*. All earlier instances were found by accident, so the
live code was swept deliberately:

- **hash → named measurement**: no further instances.
- **computed-then-discarded**: none beyond `best_rule` and `g`, both fixed.
- **constants standing in for measurements**: this one — `cell_tension = 0.5`.
- **thresholds on incomparable quantities**: both engines, above.

`split_threshold=999.0` and `merge_threshold=0.0` elsewhere are explicit
"disable this" settings with comments saying so, not defects.
