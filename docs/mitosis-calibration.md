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

---

# The positive feedback had a mechanical cause, and it is fixed

The 12× tension rise was not about what tension means. `_create_cell` added
noise to a copy of the parent's weights, and adding noise **grows the norm** —
the noise is orthogonal on average, so the result is `sqrt(a² + b²) > a`:

| generation | weight norm | vs parent |
|---|---|---|
| parent | 14.3419 | — |
| 1 | 18.3368 | **+27.9%** |
| 3 | 24.4163 | +70.2% |
| 5 | 29.2001 | **+103.6%** |

Bigger weights make bigger outputs, and `tension = (output ** 2).mean()`, so
every division raised the quantity that triggers division. That is the whole
mechanism of the runaway.

**Fix:** rescale each of the child's weight tensors back to its pre-noise norm.
The direction still changes, so cells still differentiate — that is what the
noise was for — and the amplification nobody asked for is gone.

| | before | after |
|---|---|---|
| weight norm after 5 generations | +103.6% | **+0.0%** |
| mean tension, 3 cells → 32 cells | 0.0315 → 0.3881 (**12×**) | 0.0408 → 0.049 (**1.2×**) |
| splits over 400 steps | 984 | **46** |

Twenty times less churn for the same endpoint, and tension is now stable across
population sizes — which is what makes any threshold on it mean the same thing
at 2 cells and at 32.

Regression checked: `mitosis.demo()` completes and
`bench_mitosis_calibration.py` reproduces its arms (control 2.0 cells, absolute
5.1 at the derived bar).

## What is still not fixed

The population still saturates. With the runaway gone it climbs slowly instead
of instantly, but it climbs: quantile 0.75 and below reach 32, 0.80 and above
stay at 2. Still a cliff, no band.

The reason is unchanged and is the one thing here that genuinely requires
redefining something: **tension is a per-cell property with no dependence on
population size**, so nothing ever tells the engine it has enough cells. Making
division relieve per-cell load, or normalising the trigger by population, are
both changes to what tension *means*.

That one is recorded, not taken. Everything above it was a defect with a
mechanical cause and has been fixed.

---

# a / b — the experiment

Both candidates for the missing negative feedback, measured against the control.
Neither is landed in `mitosis.py`; both are subclasses in
`bench_population_feedback.py`, as in QD-6.

**A — load relief.** Division physically splits the response: parent and child
weights are each scaled by 1/√2, so their combined output matches the pre-split
parent and each one's tension halves. The state changes, the criterion does not.

**B — normalised trigger.** Tension is untouched; the bar scales with the
population, `bar · (n_cells / min_cells)^k`. A bigger population has to work
harder to justify growing. The criterion changes, the state does not.

Working means a **band**: a range of calibration quantiles landing strictly
between the 2 floor and the 32 ceiling, and holding there. 3 seeds × 400 steps.

| arm | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | band |
|---|---|---|---|---|---|---|---|---|---|
| control | 32.0 | 32.0 | 31.9 | 31.9 | 22.0 | 13.3 | 2.0 | 2.0 | 2/8 |
| **A load relief** | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | **0/8** |
| B, k=1.0 | 4.0 | 4.0 | 3.7 | 3.3 | 2.3 | 2.3 | 2.0 | 2.0 | 4/8 |
| B, k=0.5 | 5.7 | 4.7 | 4.7 | 4.3 | 3.0 | 2.3 | 2.0 | 2.0 | 5/8 |
| **B, k=0.25** | **25.3** | **15.0** | **12.7** | **11.0** | **3.3** | 2.7 | 2.0 | 2.0 | **6/8** |

Every interior figure is stable — cell-count sd over the last 100 steps is 0.00
in all of them.

## A fails, and the reason is instructive

Halving both cells' weights drops their tension immediately below any bar that
had just been exceeded, so division stops after one or two events and the
population never leaves the floor — 2.0 cells at every setting. **The relief is
applied at full strength to the very cells that had earned the split**, which
removes the condition before it can do anything. It is a correct idea about
load and a wrong magnitude, and the magnitude is not obviously tunable: anything
less than 1/√2 is a fresh constant with no principle behind it.

## B works — and its exponent is a new knob

The mechanism does what was missing: growth raises the bar for further growth,
so the population settles instead of running to the ceiling. The band widens
from the control's 2/8 to 6/8, and the interior populations are usable rather
than nominal.

**But k is a guessed constant, and the sweep is why that matters.** k = 1.0 is
the only value with a stated reading — each cell must justify its share of a
linear load — and it holds the population at 3–4 cells. The values that give
larger, more useful populations, 0.5 and 0.25, have no principle behind them;
0.25 is simply what scored best on this metric, on this stimulus set, at this
`max_cells`. Landing it as a default would be exactly the kind of number this
whole audit has been removing.

## Not landed

`mitosis.py` is unchanged. B is the mechanism that answers the gap and the
measurements are here; choosing k — or deciding that k = 1.0's small
populations are the honest price of having a principle — is the owner's call,
with the sweep in hand rather than in the abstract.

## Correction

An earlier section here called the post-661a083 control "a cliff, no band".
That was measured before the child-inflation fix. With it, the control is
already a slope (32 → 22.0 → 13.3 → 2.0) and 2 of 8 settings land interior.
The saturation is real; "cliff" was stale and is corrected above.
