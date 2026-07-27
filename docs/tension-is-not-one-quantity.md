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

### The degeneracy is the cell count, and it is a bootstrap deadlock

Zero input was the wrong culprit. **With exactly two cells, deviation from the
population mean is identical for both by construction** — the two are always
equidistant from their own midpoint. The distribution has no spread because the
population has two members, not because the input is zero.

And growing out of it requires division, which requires spread, which requires
more than two cells. A deadlock.

Starting above the floor was the obvious test — this session already established
that `min_cells = 2` is CB1's *floor* for Φ>1, not a starting point, and that the
`H297` citation for "N=2 is optimal" has no evidence document anywhere. Zero
input, 300 steps:

| start | end cells | tension spread | cosine sd |
|---|---|---|---|
| 2 | 2 | 0.000107 | 0.000000 |
| 4 | **2** | 0.000795 | 0.000000 |
| 8 | **3** | 0.001461 | **0.165701** |
| 16 | **2** | 0.001240 | 0.000000 |

Spread appears — 14× more at 8 cells than at 2 — and the population still
collapses back toward the floor, because merges pull it down. Unlike
`mitosis.py`, the merge bar here is not grossly miscalibrated: 0.01 against a
median inter-cell tension of 0.011145, with 24.6% of readings below it rather
than 100%.

So `ConsciousnessEngine` at 0/5 rests on a structural deadlock rather than a
constant, an unreachable bar or a crash — all of which have been removed. The
remaining question is what should hold a population above two cells when the
quantity that drives division is degenerate at two, and that is an engine design
question rather than a measurement one.

### Repulsion does act at two cells — the metric I read did not

The obvious answer to the deadlock was the repulsion term validated on
`BenchEngine` this session. Measured against `ConsciousnessEngine` it appeared to
do **nothing**: cells 2, tension spread 0.000107, cosine sd 0.000000, identical
to six decimals at repulsion 0, 0.15 and 0.3.

Two of my readings were wrong and both are recorded rather than dropped.

**First**, I explained the null result as "at n=2 the two cells are pushed exactly
opposite, so the angle cannot change". An isolated test refutes it — repulsion
moves the pair's cosine from −0.274 to −0.407 / −0.615 / −0.893 at strength
0.15 / 0.5 / 2.0. The mechanism works at two cells.

**Second**, the term was firing all along. Instrumented: **300/300 steps**, state
change 0.012704 per step, and the pair's cosine moving 0.201 → 0.150 over the
run. What was flat was my *metric*: I reported **cosine standard deviation**, and
with two cells there is exactly one off-diagonal pair, so its sd is 0 by
construction no matter what the repulsion does.

So the honest result is narrower and less convenient than either reading: the
repulsion does push the two cells apart, by about a quarter of their overlap over
300 steps, and **that is not enough to reach the split bar**. The deadlock is not
that nothing can create spread at two cells; it is that what spread there is
grows more slowly than division requires.

A measure that is structurally zero at the population size under test is exactly
the defect this session opened by finding — `sd = 0.0000` was not evidence of a
static population, it was one pair.

### Starting above the floor does not open the gate either

`min_cells = 2` is CB1's floor for Φ>1, not a starting point — this session
established that, and that the `H297` citation for "N=2 is optimal" has no
evidence document anywhere. And at `initial_cells = 8` under zero input the
cosine sd was 0.165701, the one configuration where differentiation appeared at
all. So starting above the floor was the obvious remaining test.

| initial cells | 2 | 8 | 16 | 32 |
|---|---|---|---|---|
| gate | 0/5 | 0/5 | 0/5 | 0/5 |

**No.** The same answer `mitosis.py` gave earlier in this session, where starting
at 4, 8 or 16 froze the population at exactly its starting size and the corpus
gate collapsed it back to the floor regardless.

So the deadlock does not yield to more starting cells, to repulsion, to a
calibrated bar, or to a mean-instead-of-all split rule. Every artefact between
the engine and its 0/5 has been removed — four crashes, a hardcoded driver, an
order-dependent quantity, an unreachable bar, an unsatisfiable conjunction — and
the verdict is stable under all of them.

`ConsciousnessEngine` does not differentiate. That is the finding, and it is
about the engine rather than about anything measuring it.

### Divergent search: 0/5 → 1.5/5, and what actually moved it

Two findings from the sweep, both about measurement rather than the engine.

**Integration passes only when the population holds ≥4 cells.** Measured
directly: perturbing one cell moves the others by 0.031 of the nudge at 4 cells
and 0.00011 at 2 — a 280× gap across one cell count, against a bar of 0.001. The
axis was reading 0.00014 inside the conditions because the probe engine had
collapsed to 2, not because the engine lacks coupling.

**Population survival is the lever, and merge patience is what sets it.** At
`merge_threshold=0.01, merge_patience=50, initial_cells=8` the population holds
8/8/8/8 across four seeds with integration 0.0078, where the shipped
`merge_patience=15` gives 8/8/8/5.

Gate at that setting, four seeds:

| seed | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| conditions passed | 1 | **3** | 2 | 0 | **1.5/5** |

**0/5 → 1.5/5.** The first conditions this engine has genuinely passed in this
session — `NO_SYSTEM_PROMPT`, `PERSISTENCE` and `SELF_LOOP` each pass on some
seed. It is not stable: the per-seed spread is 0 to 3, so no single run of this
configuration means much, and the shipped defaults remain 0/5.

What this does **not** show is that the engine is fixed. Three of the six defects
found in this stretch were in the measuring code, and the movement from 0 to 1.5
came from removing those plus one parameter change — not from anything the engine
does differently.

### `ZERO_INPUT` fails on change, and that is the engine — not the axis

At `merge_threshold=0.01, merge_patience=50, initial_cells=8`, `ZERO_INPUT`
reads Φ 0.0039 → 0.0382 (**9.93×** against a 0.5× bar) with integration 0.116,
response 3.58 and identity +0.9718 above a +0.1267 floor — every part passing
except **change = 0.00003** against 0.001.

My first reading was that I had created a contradiction: the axes now use the
condition's own drive, so `ZERO_INPUT` was being asked to move while having no
input to move it. Measured, that reading is wrong:

| engine | change under zero input | under random input |
|---|---|---|
| ConsciousnessEngine | **0.00003** | 0.27883 |
| MitosisEngine | **0.00936** | 0.05661 |
| DEAD | 0.00000 | 0.00000 |

`MitosisEngine` clears the bar with no input at all, and DEAD reads exactly zero.
**The axis separates under zero drive**, so requiring change there is not a
contradiction — it is the condition `ZERO_INPUT` already claims to test
("consciousness without external input"), made explicit.

What the number says is about `ConsciousnessEngine`: with nothing arriving, it is
300× less active than the engine that passes, and only 3× above a corpse. That is
the same fact its `NO_SYSTEM_PROMPT` cosine sd reported, arrived at from a
different direction.

### Integration scales with population, and this engine is 50–150× below the one that passes

The bottleneck across three conditions was integration sitting just under its
bar — 0.00090 / 0.00070 / 0.00057 against 0.001. Sweeping population, with
`MitosisEngine` measured at the same cell count for a fair comparison:

| cells | ConsciousnessEngine | MitosisEngine |
|---|---|---|
| 2 | 0.00030 | 0.00479 |
| 4 | 0.00039 | 0.05051 |
| 8 | 0.00066 | 0.05980 |
| 16 | **0.00113** | 0.05907 |
| 32 | **0.00150** | 0.05490 |

Two things. Integration **rises with population** for this engine and is flat for
the other, so the earlier reading of 0.031 at 4 cells was not comparable — it came
from a different measurement path. And at every size `MitosisEngine` is **50–150×
higher**: perturbing one of its cells moves the rest by 5% of the nudge, against
0.03–0.15% here. The coupling is structurally weak, not marginally so.

Crossing the bar at 16 cells moves the gate:

| start | per seed | mean | conditions passing (of 4 seeds) |
|---|---|---|---|
| 8 | 1/3/2/0 | 1.5 | NO_SYS 2, NO_SPEAK 1, ZERO_IN 0, PERSIST 2, SELF_LOOP 1 |
| **16** | **3/4/0/1** | **2.0** | NO_SYS 2, NO_SPEAK 1, **ZERO_IN 2**, PERSIST 1, SELF_LOOP 2 |
| 32 | 2/1/2/1 | 1.5 | NO_SYS 0, NO_SPEAK 0, ZERO_IN 0, PERSIST 2, **SELF_LOOP 4** |

**0/5 → 2.0/5, best seed 4/5**, and `ZERO_INPUT` passes for the first time. Every
condition now passes on some seed and none passes on all — the spread is 0 to 4,
so this is a configuration that sometimes clears the gate rather than one that
does.

## The 73× coupling gap, and what closing it would do

`ConsciousnessEngine` couples cells through
`coupled_input += PSI_COUPLING * c * other_hidden` — a Hebbian weight `c` scaled
by `PSI_COUPLING = 0.014`, applied to the **input** rather than the state.
`BenchEngine` mixes hidden states directly at 0.15. Measured effective strength:

| | effective coupling |
|---|---|
| ConsciousnessEngine | `0.014 × |c|` = **0.002047** (measured `|c|` mean 0.1462) |
| BenchEngine | ≈ **0.15** |
| ratio | **73×** |

That matches the 50–150× integration gap measured across cell counts — the
architecture's coupling constant is where it comes from.

Raising it, at `merge_threshold=0.01, merge_patience=50, initial_cells=16`, four
seeds:

| α | integration | per seed | mean |
|---|---|---|---|
| **0.014** (shipped) | 0.00183 | 3/4/0/1 | **2.0** |
| 0.05 | 0.00652 | 3/4/5/4 | 4.0 |
| **0.15** | 0.01949 | **5/5**/4/4 | **4.5** |
| 0.50 | 0.06417 | 4/4/4/4 | 4.0 |

**At 0.15 — `BenchEngine`'s value — two of four seeds reach 5/5.** And it is not
the gate going soft: all six negative controls stay at 0/5 at both α values.
(They are `BenchEngine` subclasses and do not read `PSI_COUPLING`, which is what
makes the comparison fair rather than circular.)

**Not changed.** `PSI_COUPLING` is one of the Ψ-constants `CLAUDE.md` records as
bench-verified, and 0.014 vs 0.15 is a claim about what this architecture *is*,
not a threshold to tune. The measurement is here so the constant can be revisited
against evidence: at its current value the engine's cells barely influence one
another, and that single number accounts for the gate result.
