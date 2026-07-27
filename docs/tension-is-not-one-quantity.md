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

## Fixed — and the chain it exposed

Both defects at this site are gone, and each removal made the next one visible.

**The constant.** `cell_tension = 0.5` reached cell 0 on every step and drove all
148 splits in a 300-step run, while the computed quantity peaked at 0.037 against
a bar of 0.3. **The order-dependence.** Tension was measured against the mean of
the cells processed *so far*, so the same cell scored 0.024 at position 1 and
0.007 at position 4. Deferring the computation until the loop ends removes both:
every cell is compared against the population, and no cell needs a stand-in.

Removing the constant stopped division entirely, exactly as the annotation
predicted — the engine sat at 2 cells, where Φ ≈ 0 and every condition fails.
That is what the constant had been hiding. So `_check_threshold_reachable` now
calibrates instead of only warning: when the bar is out of reach it is taken from
the q0.90 of the tension this engine actually produces. Measured, it fires at
0.3 → 0.0049 / 0.0052 / 0.0296 depending on the run, all reported at the site.

Division resumed, and three latent crashes surfaced — all in code I added earlier
this session, none of which anticipated an engine whose cell count changes:

| site | error |
|---|---|
| `NO_SYSTEM_PROMPT` | `min(cells, 64)` used the REQUESTED count; the engine returned 2 rows → boolean-index mismatch |
| `_three_axes` identity | `cur - prev` with 4 rows against 2 |
| `_three_axes` integration/response | two runs diverged to 63 and 61 cells |

All three were scored as FAIL, so a shape error was indistinguishable from a real
verdict. Fixed by comparing only the cells present in both.

**The engine is still 0/5, and now for real reasons rather than crashes:**
`NO_SYSTEM_PROMPT` reads cosine sd 0.0000 (two cells give one pair),
`NO_SPEAK_CODE` var 0.0006 against 0.001, `SELF_LOOP` Φ 0.0000 → 0.0000.
Stopping here: each fix has been exposing the next edge, which is the pattern
this session identified as the wrong way to work. The remaining failures are
measured and stated rather than chased.

### The calibrated bar and the patience rule were incompatible

Calibration fired (0.3 → 0.0052, peak 0.0056 — reachable) and the population
still sat at 2 cells 200 steps later. The split rule was
`all(t > bar for t in last split_patience)`, and the calibrated bar is the q0.90
of observed tension, so clearing it on **five consecutive** readings has
probability ≈ 0.1⁵ — about 1 in 100,000 steps per cell. **The bar was reachable
and the conjunction was not.** `mitosis.py` already uses the mean over the window
for exactly this reason; the same rule here:

| step | 0 | 100 | 200 | 250 | 300 | 400 |
|---|---|---|---|---|---|---|
| cells, `all()` rule | 2 | 2 | 2 | 2 | 2 | 2 |
| cells, mean rule | 2 | 2 | 2 | 2 | **63** | 63 |

Division works. `PERSISTENCE` now records Φ growing 0.000 → 0.312 → 0.535 where
it read 0.000 throughout before.

One more crash of the same family surfaced and is fixed: the identity axis built
its shuffled null with `torch.randperm(n)` on the REQUESTED cell count, indexing
past the end of an engine holding fewer rows (index 63, size 63).

**Still 0/5, and every verdict is now real rather than a shape error:**
`NO_SYSTEM_PROMPT` cosine sd 0.0000 — under zero input the tension never reaches
even the calibrated bar, so that condition still sees 2 cells and one pair;
`NO_SPEAK_CODE` var 0.0006 against 0.001; `SELF_LOOP` Φ 0.0000 → 0.0000.

Four crashes were removed from the audit path in this pass, all of them mine and
all scored as FAIL — the same defect `HIVEMIND` had at the start of this session,
where a 0.3s crash was recorded as a failed condition for eleven engines.

### Under zero input the distribution is degenerate, and no bar can help

`NO_SYSTEM_PROMPT` still reads cosine sd 0.0000, and the reason is not the
threshold. Tension over 300 steps, same engine, two drives:

| drive | cells | mean | q90 | max | bar |
|---|---|---|---|---|---|
| random | 63 | 0.006147 | 0.007469 | 0.008819 | 0.005230 |
| **zero** | **2** | **0.004871** | **0.004884** | **0.004884** | 0.004884 |

Under zero input the mean, the q90 and the maximum agree to five decimals — the
tension is effectively constant. A quantile bar then sits at the maximum, the
mean sits below it, and nothing ever splits. `mitosis.py`'s own docstring
anticipated the shape of this ("a bar that only the maximum reaches is never
exceeded") and chose a quantile to avoid it, but a quantile cannot separate high
from low in a distribution with no spread.

**So the verdict stands and it is about the engine, not the measurement.**
`NO_SYSTEM_PROMPT` asks whether identity emerges from cell dynamics alone with no
external input. Under zero input this engine's cells produce identical tension,
stay at two, and give one pair with sd 0. The condition is reporting exactly
that: identity does not emerge. Every measurement artefact between the engine and
that verdict has now been removed — four crashes, a hardcoded driver, an
order-dependent quantity, an unreachable bar and an unsatisfiable conjunction.

`ConsciousnessEngine` is 0/5, and for the first time the number means what it
says.
