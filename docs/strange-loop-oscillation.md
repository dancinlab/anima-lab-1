# The split/merge excursion in ConsciousnessEngine

Measurement of the cell-count oscillation in `consciousness_engine.ConsciousnessEngine`.
Configuration throughout: `cell_dim=32, hidden_dim=128, initial_cells=2, max_cells=32,
n_factions=12`, defaults otherwise (`split_threshold=0.3`, `split_patience=5`,
`merge_threshold=0.01`, `merge_patience=15`, `min_cells=2`).

No engine code was changed. Instrumentation wraps `_check_splits` / `_check_merges`
from outside and only reads state before delegating to the original method.

Headline: **it is not an oscillation.** It is a single transient, triggered at exactly
the calibration step, whose occurrence is decided by the sign of a drift of order
10⁻¹⁰ per step. Once it collapses, the n=2 state is absorbing — verified over 19,431
consecutive quiet steps.

---

## 0. Preconditions: is the thing being measured bounded and coupled?

Checked first, because a decoupled or diverged population would produce these same
plots while measuring nothing.

**Bounded.** 5 seeds × 3000 steps × 2 input regimes: 0 NaN/inf steps, `max|h| ≤ 0.922`,
`max|output| ≤ 1.013`. Nothing runs away numerically.

**Coupled.** Fork the engine at step S, perturb ONE cell's hidden by ε=0.05, replay the
identical input sequence in both forks, and measure the other cells. All cells share the
same input, so a non-perturbed cell can only move via the coupling term. Control: an
identical fork with `_coupling` zeroed every step.

| fork at | n_cells | non-perturbed cell \|Δh\| (coupled) | same cell, coupling zeroed | ratio |
|---|---|---|---|---|
| step 250 | 2  | 1.01e-4 | 4.0e-6  | 25× |
| step 600 | 31 | 1.55e-2 | 1.10e-4 | 141× |
| step 900 | 32 | 1.62e-2 | 2.2e-5  | 736× |

Coupling is real, and its strength scales with population:

| n_cells | mean \|c\| off-diagonal | coupling-term norm per cell | vs shared input \|x\|=5.16 |
|---|---|---|---|
| 2  | 0.184 | 0.0024 | 0.05% |
| 31 | 0.637 | 0.258  | 5.0% |
| 32 | 0.807 | 0.349  | 6.8% |

---

## 1. The regime the gate actually runs in

`bench_v2._verify_no_system_prompt` and its siblings drive the engine with
`x_zero = torch.zeros(1, dim)` on every step. This matters more than any parameter:

| input | outcome over 3000 steps (5 seeds) |
|---|---|
| zeros (gate) | one excursion, then n=2 permanently; splits == merges exactly |
| N(0,1) random | runaway to the ceiling, pinned at 25–32, never returns to 2 |

Everything below is the zero-input regime unless marked otherwise, because that is the
condition the gate and the deployed runtime measure under.

---

## 2. Period and amplitude — over 5 seeds

There is no period. Every seed that moves at all produces exactly ONE excursion, and
every one of them begins at **step 200** — the step `_check_threshold_reachable` fires.

| seed | excursion | duration | peak n | splits | merges | final n |
|---|---|---|---|---|---|---|
| 42   | none         | —   | 2  | 0   | 0   | 2 |
| 7    | step 200→569 | 370 | 32 | 688 | 688 | 2 |
| 1234 | step 200→294 |  95 | 32 | 145 | 145 | 2 |
| 2026 | step 200→306 | 107 | 32 | 188 | 188 | 2 |
| 99   | step 200→…   |   — | 32 | 303 | 303 | 2 |

Splits and merges are **exactly equal in every seed**. Every cell ever created is
destroyed. Amplitude 2 → 32 → 2 (16× and back), rise time 17 steps.

```
seed 7, zero input, n_cells (max per 10-step bucket)

 32 |      ######### ###### ######### ### # #          |
 28 |      ################ ######### ########         |
 24 |      ################ ######### ########         |
 20 |      ###################################         |
 16 |      ###################################         |
 12 |      ###################################         |
  8 |     ####################################         |
  4 |     #####################################        |
  2 |##################################################|
    +--------------------------------------------------+
   150            step                             650
     ^calibration at 200          collapse at 553^
```

**Is it a limit cycle?** No — checked directly with a 20,000-step run:

| seed | last step with n>2 | consecutive steps at n=2 after it | splits after step 1000 | merges |
|---|---|---|---|---|
| 7    | 569 | 19,431 | 0 | 0 |
| 1234 | 294 | 19,706 | 0 | 0 |

A one-off excursion into an absorbing state. The sampled trace that started this
investigation (2, 2, 2, 31, 2, 2 at steps 100/200/300/500/700/1000) was not
undersampling a cycle — the per-step count confirms there is only ever one peak.

---

## 3. Driver 1 — the split bar is calibrated into numerical noise

`_check_threshold_reachable` fires once at step 200, sets
`split_threshold := q0.90` of all tension observed so far, and latches
`_calibrated = True` permanently.

Under zero input the engine reaches a fixed point by step ~100. The calibration sample
is therefore very nearly a constant — and **q0.90 of a constant is the constant.** The
bar is placed exactly at the operating point it is supposed to discriminate against.

| seed | calib sample std/mean | bar (q0.90) | split window mean @200 | (window/bar − 1) | drift/step | n @260 |
|---|---|---|---|---|---|---|
| 42   | 9.6e-3 | 0.0019636 | 0.0019635 | **−4.2e-5** | −4.7e-10 | 2 |
| 8    | 1.8e-2 | 0.0025277 | 0.0025276 | **−1.2e-5** | −2.0e-10 | 2 |
| 13   | 1.9e-2 | 0.0019483 | 0.0019482 | **−4.1e-5** | −5.3e-10 | 2 |
| 1234 | 1.3e-2 | 0.0024346 | 0.0024346 | **+1.8e-6** | +3.7e-10 | 30 |
| 2026 | 2.8e-2 | 0.0023651 | 0.0023651 | **+4.1e-6** | +5.9e-10 | 31 |
| 5    | 1.9e-2 | 0.0022227 | 0.0022227 | **+5.1e-6** | +6.7e-10 | 27 |
| 99   | 1.5e-2 | 0.0018845 | 0.0018845 | **+6.4e-6** | +7.1e-10 | 7 |
| 7    | 2.0e-2 | 0.0014565 | 0.0014565 | **+9.4e-6** | +8.0e-10 | 31 |

The window mean lands within ±4.2×10⁻⁵ *relative* of the bar. The sign of a residual
drift of ~10⁻¹⁰/step decides between **no growth ever** and a **16× excursion**:
3 of 8 seeds drift down and sit at 2 cells forever; 5 of 8 drift up and hit 27–31 cells
within 60 steps. The engine's entire population behaviour is decided in the 7th
significant figure.

For contrast, the same calibration under random input has a real margin — the sample is
genuinely dispersed (std/mean ≈ 0.25–0.33), so q0.90 sits 18–47% above the window mean.
The pathology is specific to the fixed-point regime the gate measures in.

---

## 4. Driver 2 — the split statistic grows with n, so splitting amplifies itself

`step()` computes each cell's tension as `mean((output_i − population_mean)²)`. For a
population with per-dimension dispersion σ², the expectation of that statistic is

```
    E[cell_tension] = σ² · (1 − 1/n)
```

Verified against measurement (σ² estimated independently as `inter_mean/2`, since
inter-cell tension is `mean((o_i − o_j)²)` with expectation 2σ²):

| regime | n | measured cell tension | σ²(1−1/n) predicted | ratio |
|---|---|---|---|---|
| zero   | 2  | 0.000266 | 0.000266 | 1.0003 |
| zero   | 32 | 0.002905 | 0.002908 | 0.9992 |
| random | 2  | 0.006069 | 0.006069 | 1.0000 |
| random | 32 | 0.022149 | 0.022153 | 0.9998 |

Ratio is 0.999–1.002 across all 17 seed × regime × n combinations. The identity is exact.

The bar is calibrated at n=2, where the statistic reads **0.5σ²**. At n=32 the *same*
dispersion reads **0.969σ²** — 1.94× larger. So the first split mechanically raises the
measured tension of every cell, pushing more cells over a bar that never moves:

```
seed 7, during the runaway — % of cells whose 5-step window is above the bar

  n= 3  ██████████████████████████                        66.7%
  n= 5  ████████████████████████████████                  80.0%
  n= 7  ██████████████████████████████████                85.7%
  n=11  ███████████████████████████████████               88.9%
  n=19  █████████████████████████████████████             94.1%
  n=32  ██████████████████████████████████████            96.9%
```

2 → 32 cells in 17 steps (steps 200→217). Note this is only half the growth: total
measured tension rises 10.9× from trough to peak, of which 1.94× is the estimator's
n-dependence and 5.6× is genuine added dispersion (each split deep-copies the parent
and adds N(0, PSI_COUPLING=0.0153) weight noise).

---

## 5. Driver 3 — the merge bar sits above the bulk of the distribution, and `min_cells` is all that holds the trough

`merge_threshold = 0.01` against the inter-cell tension actually produced under zero input:

| state | median pair tension | % of individual pair readings below 0.01 |
|---|---|---|
| n=2 (trough)  | 0.00107 – 0.00785 | **100%** |
| n=32 (peak)   | 0.00344 – 0.00660 | **76 – 100%** |

Merging is effectively unconditional. Of the pairs that survive long enough to
accumulate `merge_patience=15` samples, **60–100% qualify on every step**.

At n=2 the single surviving pair is merge-eligible 100% of the time (inter-tension
0.00583 vs bar 0.01) and is prevented from merging by exactly one line:

```python
if self.n_cells <= self.min_cells:      # _check_merges, line 608
    return events
```

So the trough is not a state where merging has stopped — it is a state where merging is
continuously qualified and blocked by the floor. The instant a split lifts n to 3, the
block lifts and every pair is already over the line.

What rate-limits merging is bookkeeping, not tension: `_remove_cell` deletes every
history key containing the removed cell, so churn keeps resetting the 15-step counters.
This is why the collapse arrives as a burst rather than a decay:

```
 step   n  splits  merges
  552  31    1       1
  553   7    1      25   ← 25 merges in ONE step
  554   4    3       6
  ...
  570   2    0       1    last event; silent for the next 19,431 steps
```

**Refutation checked.** I first stated the merge bar sat above the *entire*
distribution. That is false and the check caught it: peak pair tension reaches
0.020–0.031 under zero input (259 of 3000 steps above 0.01 for seed 7) and 0.23–0.33
under random input. The defensible claim is the bulk one — 76–100% — which is what makes
merging effectively unconditional.

---

## 6. Driver 4 — merging is irreversible homogenisation, which is why there is no second cycle

Merge averages both the parameters and the hidden state of the two cells. After the
excursion, the surviving pair is far more similar than the pair that started it — and
`split_threshold` is frozen at its pre-excursion value:

| seed | trough tension before | trough tension after | frozen bar | after/bar | after/before |
|---|---|---|---|---|---|
| 7    | 0.001456 | 0.000266 | 0.001456 | **0.183** | 0.183 |
| 1234 | 0.002435 | 0.000486 | 0.002435 | **0.200** | 0.200 |
| 2026 | 0.002365 | 0.000256 | 0.002365 | **0.108** | 0.108 |
| 99   | 0.001884 | 0.000787 | 0.001884 | **0.418** | 0.418 |

The post-collapse operating point sits **2.4–9.3× below its own split bar**, so it can
never split again. Simultaneously its inter-cell tension (0.0008–0.0011) sits **9–12×
below the merge bar**, so it permanently wants to merge and is held only by `min_cells`.

The n=2 end state is doubly pinned. That is the whole reason the trace reads
"oscillation": the system has exactly one excursion available to it, and spends the rest
of its life in a state it cannot leave in either direction.

---

## 7. Consequence for the gate

`_verify_no_system_prompt` runs 300 zero-input steps, then requires
`0.01 < mean_cos < 0.99 AND std_cos > 0.001` over the off-diagonal pairwise cosines.

Measured directly:

| seed | n @300 | n @1000 | mean_cos | std_cos | passes |
|---|---|---|---|---|---|
| 42   | 2  | 2 | −0.0609 | **0.000e+00** | no |
| 1234 | 2  | 2 | 0.8667  | **0.000e+00** | no |
| 5    | 2  | 2 | 0.6687  | **0.000e+00** | no |
| 8    | 2  | 2 | −0.0905 | **0.000e+00** | no |
| 13   | 2  | 2 | −0.1647 | **0.000e+00** | no |
| 7    | 26 | 2 | 0.5115  | 2.87e-01 | **yes** |
| 2026 | 7  | 2 | 0.6089  | 2.31e-01 | **yes** |
| 99   | 32 | 2 | 0.4721  | 3.05e-01 | **yes** |

At n=2 the mask over a 2×2 leaves one distinct value appearing twice, so `std_cos` is
identically **0.000e+00** — not small, zero. The criterion cannot be met at 2 cells by
any dynamics whatsoever.

One correction to the framing that prompted this work: the step-300 measurement is **not**
unpassable by construction — it passes in 3 of 8 seeds, whenever the excursion happens
to still be in flight at step 300. What decides it is the drift sign from §3, not
anything about the cells' behaviour. By step 1000 all 8 seeds are back at n=2, so any
criterion evaluated there fails for every seed. The gate is measuring a coin flip at
step 300 and an absorbing state at step 1000.

---

## 8. What a fix has to change

Not a threshold retune — three of these are structural, and the numbers say which.

**(a) The split statistic must not depend on n.** `mean((o_i − pop_mean)²)` has
expectation σ²(1−1/n), verified to ratio 0.999–1.002. Comparing it against a bar
calibrated at a different n is comparing two different quantities. Dividing by (1−1/n)
— equivalently using the unbiased variance, or comparing pairwise differences directly —
removes a 1.94× spurious inflation between n=2 and n=32 and with it the positive feedback
that takes 2→32 in 17 steps.

**(b) Calibration must not place the bar inside the sample's own noise.** Under the
gate's drive the calibration sample has std/mean ≈ 0.01–0.03 and the resulting bar lands
within 4.2e-5 relative of the operating point, leaving a 10⁻¹⁰/step drift to decide the
outcome. A bar taken from a distribution this degenerate needs either a multiplicative
margin above the observed maximum, or a refusal to calibrate at all when the sample's
dispersion is below some floor — the current code has neither, and `_calibrated = True`
latches the result permanently.

**(c) The merge bar has the same disease as the original `split_threshold=0.3`, inverted.**
0.3 was unreachable (153–206× above the observed peak). 0.01 is *always* reachable: it
sits above 100% of pair readings at n=2 and 76–100% at n=32. A bar that everything clears
is not a bar. To be selective in the trough it would need to be ~1e-4 — two orders of
magnitude below its current value. It should be calibrated from the observed inter-cell
tension distribution the same way `split_threshold` now is, with the margin guard from
(b) attached.

**(d) A merge that averages parameters is not reversible, and nothing accounts for that.**
It costs 0.108–0.418× of the population's tension permanently, which is what makes the
n=2 state absorbing against a frozen bar. Either the bar has to track the population it
is measuring (re-calibrate rather than latch), or merging has to preserve dispersion.

**Diagnostic that would refute any proposed fix:** run zero input for 3000 steps across
≥5 seeds and require (i) the split/merge counts not to be exactly equal, (ii) the
population at step 1000 to be above `min_cells` for every seed, and (iii) the outcome not
to flip on the sign of a 10⁻¹⁰/step drift — i.e. the same seeds succeed when the input is
perturbed by 1e-6. The current engine fails all three.

---

## Reproduction

Instrumentation and analysis scripts (read-only wrappers, no engine changes):

```
scratchpad/instrument.py      per-step trace, --mode {random,zero}, N seeds
scratchpad/coupling_check.py  perturbation test vs zeroed-coupling control
scratchpad/pairstats.py       pair-level merge eligibility + calibration drift
scratchpad/calib.py           calibration razor width, 8 seeds × 2 regimes
scratchpad/refute.py          checks C1–C4, including the one that failed
scratchpad/gatecheck.py       the actual NO_SYSTEM_PROMPT values
```
