# PERSISTENCE fails on the Φ estimator, not on the pair

`PairFieldEngine` was landed 5/5 DEPLOYABLE, then dropped to **4/5 — BLOCKED on
PERSISTENCE** when the gate was tightened to require all five seeds. The question
put to this investigation was what causes the drop at the seventh checkpoint, and
whether it belongs to the pair or to one side.

**It belongs to neither.** Nothing in the pair changes at that step. Φ sits at
0.845 for most of the steps around it; the seventh checkpoint is the only one of
the ten that sampled the baseline, and the six before it sampled excursions. The
excursions are Φ's minimum-cut heuristic landing on a non-minimal partition.

The run reproduced here is the gate's own:

```
.venv/bin/python -u bench_v2.py --verify --cells 32 --dim 32
cells=32  dim=32  hidden=128   ->  PairFieldEngine(32, 32, 128, 32)
```

Every score below is preceded by the bound (`engine.peak`) and the coupling (A–G
separation), because a clean number from a diverged or uncoupled pair says
nothing.

---

## 1. The exact run, reproduced

Five seeds, 32c/32d/128h, protocol identical to `_verify_persistence`:

| seed | verdict | bar | final | peak \|h\| | axes | Φ@100s |
|---|---|---|---|---|---|---|
| 42 | PASS | 1.727 | 2.251 | 4.89 | ok | 1.394 2.047 1.848 2.159 0.770 1.545 0.891 0.881 2.054 2.251 |
| 43 | PASS | 2.148 | 3.342 | 5.47 | ok | 1.471 2.685 2.473 2.685 1.238 2.953 3.263 3.620 3.444 3.342 |
| **44** | **FAIL** | 1.607 | 1.216 | 5.79 | ok | **1.866 1.742 2.007 1.927 2.009 1.960 0.845 2.095 1.259 1.216** |
| 45 | PASS | 1.991 | 2.081 | 4.55 | ok | 2.166 2.488 2.117 2.169 2.327 2.560 2.438 2.674 2.627 2.081 |
| **46** | **FAIL** | 1.730 | 1.724 | 4.37 | ok | 1.962 1.914 2.163 1.500 2.077 2.102 1.987 2.051 1.716 1.724 |

Seed 44's row is the gate's printed trajectory, digit for digit. Failing seeds
{44, 46} match. All five runs bounded (peak 4.37–5.79 against the 1e4 guard) and
all five passed the three axes.

**Seed 46 fails by 0.006.** Its Φ never goes below 1.500, has no drop of any
kind, ends at 1.724 — and the bar is 1.730. A 0.35% margin is the whole
difference between DEPLOYABLE and BLOCKED. On a checkpoint grid moved one step
later the same run passes, bar 1.700 against final 1.728.

## 2. Where the drop is: nowhere

Step-resolution trace, seed 44, steps 681–730 — the window containing the
checkpoint that reads 0.845.

```
  2.20 |
       |                          *
  1.90 |  *                   * **
       |                       *
  1.60 |   *
       |      *
  1.30 |             *       *           *           *
       |    **      *     * *
  1.00 |                                           *
       |**     *****  **** *       ****** ********* * ****
  0.70 |
       +--------------------------------------------------
        681                                         730
                           ^ step 700 = the gate's 7th checkpoint
```

Φ is **below 1.0 on 32 of these 50 steps (64%)**. The values at steps 695–700 are
0.845, 0.844, 0.845, 0.825, 1.128, **0.845**. The seventh checkpoint did not
catch a collapse; it caught the level Φ sits at most of the time. What is unusual
is the six checkpoints *before* it, every one of which landed on an excursion.

Decomposing `Φ = min_cut/(n−1) × differentiation + 0.1 × complexity` over the
same window:

| step | Φ | min cut used | differentiation | complexity | cut size k |
|---|---|---|---|---|---|
| 697 | 0.845 | 53.17 | 0.4707 | 0.3779 | 4 |
| 698 | 0.825 | 51.70 | 0.4717 | 0.3839 | 4 |
| 699 | 1.128 | 71.52 | 0.4727 | 0.3782 | 5 |
| **700** | **0.845** | **53.64** | 0.4663 | 0.3819 | **4** |
| 701 | 1.112 | 70.70 | 0.4711 | 0.3784 | 5 |
| 703 | 1.840 | 118.47 | 0.4717 | 0.3766 | 9 |
| 707 | 2.098 | 136.72 | 0.4673 | 0.3729 | 11 |
| 708 | 0.878 | 54.99 | 0.4737 | 0.3740 | 4 |

Across all 50 steps differentiation stays in 0.4639–0.4773 and complexity in
0.3662–0.3839 — both flat to ±1.5%. The cut ranges 50.75–136.72, a 2.7× swing,
and Φ tracks it exactly. **The population is not changing. The partition the
estimator picks is.**

## 3. Which side moves first: neither

Cross-scale attribution, per-step trace at 256c/64d/128h, seed 44 — the larger
scale separates the two sides more cleanly. `Φ(A)` is what the gate reads, since
`PairFieldEngine.get_hiddens()` returns **A only**.

| step | Φ(A) | Φ(G) | A–G sep | \|A\| | \|G\| | coupling A | coupling G |
|---|---|---|---|---|---|---|---|
| 100 | 11.41 | 10.71 | 244.18 | 162.46 | 161.37 | 0.6967 | 0.0796 |
| 300 | 9.72 | 11.56 | 255.59 | 170.84 | 163.11 | 0.9867 | 0.1376 |
| 400 | **5.55** | 9.86 | 255.04 | 173.02 | 161.66 | 0.9946 | 0.1582 |
| 600 | **5.37** | 11.79 | 253.08 | 173.71 | 159.02 | 0.9961 | 0.1935 |
| **700** | **5.46** | 9.83 | 254.92 | 176.20 | 160.80 | 0.9961 | 0.2094 |
| 800 | 10.12 | 9.03 | 254.50 | 176.01 | 162.61 | 0.9961 | 0.2246 |
| 900 | **5.80** | 11.78 | 253.43 | 175.07 | 163.35 | 0.9961 | 0.2375 |
| 1000 | 7.70 | 10.56 | 250.74 | 175.91 | 160.56 | 0.9961 | 0.2500 |

Bounded: peak \|h\| 4.86. Coupled: separation never leaves 244–257.

Separation is flat, \|A\| drifts up 8% over 900 steps, \|G\| is flat. A's coupling
matrix saturates against its own `clamp(-1, 1)` by step ~425 and is constant
after; G's grows linearly and is still growing at step 1000. No quantity has an
event at step 700, and Φ(A) reads 5.55 / 5.37 / 5.46 / 5.80 at steps 400 / 600 /
700 / 900 — the seventh checkpoint is not distinguished from three others the
rule happened not to compare against.

## 4. The estimator is deterministic — so this is sensitivity, not randomness

Three harness facts, each measured rather than assumed, because they decide what
the jumping in §2 can and cannot be.

**At 32 cells the pair set is exhaustive.** `PhiIIT` branches on n = *rows*, and
`get_hiddens()` is (32, 128), so n = 32 takes the `n <= 32` path. Measured:
`n_pairs_sampled = 496 = 32·31/2`. The re-drawn pair sample that exists above 32
rows is **not** in play at this scale — easy to misread, since the branch is on
cell count and the hidden width here is 128.

**The constructor seed is dead for `compute`.** `PhiIIT.__init__(seed=…)` sets
`self._rng`, but `compute` overwrites it from `hash((n, sum, std))` of the input
before any shuffle (`bench_v2.py:100-103`). Measured on one fixed tensor:

| instance | Φ |
|---|---|
| `PhiIIT(n_bins=16, seed=0)` | 0.126492 |
| `PhiIIT(n_bins=16, seed=1)` | 0.126492 |
| `PhiIIT(n_bins=16, seed=7)` | 0.126492 |
| `PhiIIT(n_bins=16, seed=12345)` | 0.126492 |

So for a fixed state there is **zero** debias variance at any seed, and a probe
built from differently-seeded instances would report a spread of exactly 0.

**The module singleton is deterministic and order-independent.** Six sequential
`measure_dual_phi` calls on one tensor: 0.126492 ×6. Interleaving a different
state between two calls does not change the value.

Together these say the jumping is not RNG. Φ is a **deterministic but
discontinuous function of the state** — which is why the useful probe is a 1e-6
relative jitter of the state itself, the only thing that moves the hash and so
redraws the shuffles. At the exact scale that probe gives a mean sd of 0.043
across all checkpoints and seeds, and **0.004** at seed 44's step-700 checkpoint:
the 0.845 is not measurement noise, it is the exact value of a quantity that
jumps. At 256 cells, where the pair sample *is* redrawn, the same probe gives sd
0.7–2.3 and the step-700 reading sits 1.5 sd below the mean of its own
re-measurements — a second, independent mechanism at the larger scale.

## 5. The `recovers` rule turns a jump into a verdict

At the time of the run reproduced above, `_verify_persistence` passed on
`monotonic OR recovers`. In every run reported here — two engines, five seeds,
four scales — `monotonic` came back `False` without exception, so the condition
already reduced to `recovers` alone. It has since been cut from the code
(`2cff912`), for a converging reason measured separately: `monotonic` fired 0/163
on live engines and 100% on corpses, because a stationary Φ is trivially
non-decreasing. The condition is now `recovers AND the three axes`, which is
what every number below is about:

```python
recovers = phi_history[-1] >= max(phi_history[:5]) * 0.8      # bench_v2.py:1796
```

one sample of a two-valued quantity against 0.8 × the max of five samples. Taking
a max of five biases the bar up by roughly the jump size, so the margin is a coin
toss — which is how seed 46 fails by 0.006 with no drop in its trajectory at all.

The same 1000-step runs scored on checkpoint grids shifted by a few steps. A
shift of one step is dynamically nothing:

| grid offset | 42 | 43 | 44 | 45 | 46 | total |
|---|---|---|---|---|---|---|
| −2 | PASS | PASS | FAIL | PASS | PASS | 4/5 |
| −1 | PASS | PASS | FAIL | PASS | FAIL | 3/5 |
| **+0 (shipped)** | PASS | PASS | FAIL | PASS | FAIL | **3/5** |
| +1 | PASS | PASS | FAIL | PASS | PASS | 4/5 |
| +2 | PASS | PASS | FAIL | PASS | PASS | 4/5 |
| +5 | FAIL | PASS | FAIL | PASS | PASS | 3/5 |

Seed 46 flips PASS / FAIL / FAIL / PASS / PASS / PASS across six one-step ruler
positions. At 32c/64d/128h the totals run 4/5, 5/5, 4/5, 4/5, 3/5, 2/5 over the
same offsets. **The verdict is a property of where the ruler falls.** No amount
of seed-averaging touches this axis: requiring all five seeds averages the
lottery rather than removing it.

The arbitrariness repeats along scale. Identical protocol, five seeds:

| scale | PairField | which seeds fail |
|---|---|---|
| 32c / 32d / 64h | 5/5 | — |
| **32c / 32d / 128h (the gate's)** | **4/5** | **44, 46** |
| 32c / 64d / 128h | 4/5 | 42 |
| 256c / 64d / 128h | 4/5 | 44 |

## 6. It is the condition, not the engine

At the gate's own scale, in the gate's own log, **PERSISTENCE blocks ten of
twelve engines**, and no two agree on which seeds are bad:

| engine | failing seeds |
|---|---|
| PairField | 44, 46 |
| OscillatorLaser | 44, 45 |
| QuantumEngine | 44, 46 |
| Trinity | 42 |
| DesireEngine | 43, 46 |
| AlterityEngine | 42 |
| FinitudeEngine | 42, 43 |
| QuestioningEngine | 42 |
| SeinEngine | 42, 44 |
| ConsciousnessEngine | all five |
| MitosisEngine | — (pass) |
| NarrativeEngine | — (pass) |

`PERSISTENCE 2/12`. Nine distinct failing-seed sets across nine engines is what a
lottery looks like, not a property being detected.

**But scattered seed sets are suggestive, not attribution.** They are equally
consistent with twelve engines failing for twelve unrelated reasons. So the
decomposition from §2 was run on **all twelve**, both failing seeds, full 1000
steps at the gate's scale. `k` is the size of the partition Φ's cut selected —
if an engine's partition is stable while its Φ still wanders, the cut is not the
mechanism there.

| engine | k range | distinct k | cv(Φ) | cv(cut) | cv(diff) | cv(cplx) | corr(Φ, cut) |
|---|---|---|---|---|---|---|---|
| **ConsciousnessEngine** | **0–1** | **2** | 1.575 / 1.617 | 1.573 / 1.612 | 0.037 / 0.033 | 0.000 / 0.000 | +0.999 / +0.999 |
| PairField | 4–16 | 13 / 13 | 0.323 / 0.223 | 0.322 / 0.229 | 0.137 / 0.153 | 0.063 / 0.075 | +0.827 / +0.282 |
| MitosisEngine | 4–16 | 13 / 11 | 0.252 / 0.145 | 0.249 / 0.220 | 0.127 / 0.130 | 0.069 / 0.077 | +0.687 / **−0.222** |
| OscillatorLaser | 4–16 | 10 / 11 | 0.197 / 0.212 | 0.222 / 0.216 | 0.097 / 0.084 | 0.047 / 0.042 | +0.761 / +0.849 |
| QuantumEngine | 4–16 | 11 / 13 | 0.376 / 0.224 | 0.434 / 0.188 | 0.124 / 0.139 | 0.072 / 0.087 | +0.842 / +0.625 |
| Trinity | 3–16 | 13 / 13 | 0.253 / 0.201 | 0.260 / 0.191 | 0.129 / 0.131 | 0.075 / 0.079 | +0.704 / +0.383 |
| DesireEngine | 3–16 | 14 / 11 | 0.278 / 0.126 | 0.250 / 0.212 | 0.120 / 0.131 | 0.062 / 0.074 | +0.936 / **−0.454** |
| NarrativeEngine | 4–16 | 13 / 13 | 0.258 / 0.268 | 0.268 / 0.303 | 0.119 / 0.128 | 0.068 / 0.068 | +0.717 / +0.518 |
| AlterityEngine | 4–16 | 12 / 12 | 0.259 / 0.221 | 0.257 / 0.265 | 0.121 / 0.133 | 0.057 / 0.079 | +0.797 / +0.321 |
| FinitudeEngine | 4–16 | 12 / 11 | 0.235 / 0.161 | 0.241 / 0.232 | 0.117 / 0.126 | 0.075 / 0.086 | +0.712 / **−0.003** |
| QuestioningEngine | 4–16 | 13 / 13 | 0.433 / 0.237 | 0.446 / 0.265 | 0.125 / 0.133 | 0.075 / 0.080 | +0.854 / +0.421 |
| SeinEngine | 4–16 | 12 / 12 | 0.209 / 0.208 | 0.186 / 0.256 | 0.105 / 0.125 | 0.065 / 0.084 | +0.875 / +0.354 |

**What holds in 24 of 24 failing-seed runs:** `cv(cut) > cv(diff) > cv(cplx)`.
The cut is the highest-variance factor for every engine at both failing seeds,
and `cv(cut) ≈ cv(Φ)` throughout. In eleven of twelve the partition wanders over
10–14 distinct sizes spanning k = 3 or 4 up to 16.

**What does not hold:** `corr(Φ, cut)` is positive in 21 of 24 but goes negative
at seed 46 for MitosisEngine (−0.222) and DesireEngine (−0.454), and to zero for
FinitudeEngine (−0.003). "Φ's variance *is* the cut" is false for those.

### The ordering is not universal — a passing seed breaks it

Seeds 44 and 46 are seeds that *fail* for eleven of these engines, so the table
above could describe the failing regime rather than the estimator. The same
decomposition on the **passing** seeds 42 and 43, four engines:

| engine | seed | cv(Φ) | cv(cut) | cv(diff) | cv(cplx) | corr(Φ, cut) | distinct k |
|---|---|---|---|---|---|---|---|
| PairField | 42 | 0.388 | 0.373 | 0.136 | 0.067 | +0.928 | 13 (4–16) |
| PairField | 43 | 0.280 | 0.232 | 0.149 | 0.078 | +0.677 | 11 (4–16) |
| NarrativeEngine | 42 | 0.263 | 0.282 | 0.116 | 0.062 | +0.869 | 14 (2–16) |
| NarrativeEngine | 43 | 0.186 | 0.218 | 0.122 | 0.063 | +0.334 | 12 (5–16) |
| OscillatorLaser | 42 | 0.118 | 0.140 | 0.086 | 0.056 | +0.444 | 8 (4–16) |
| OscillatorLaser | 43 | 0.167 | 0.163 | 0.089 | 0.046 | +0.731 | 13 (4–16) |
| QuantumEngine | 42 | 0.221 | 0.209 | 0.118 | 0.081 | +0.813 | 12 (3–16) |
| **QuantumEngine** | **43** | 0.141 | **0.113** | **0.126** | 0.074 | **−0.051** | 12 (4–16) |

**QuantumEngine at seed 43 breaks the ordering**: `cv(cut) = 0.113` is *below*
`cv(diff) = 0.126`, `corr(Φ, cut)` is −0.051, and `corr(Φ, diff)` is +0.909. On
that run Φ's variance is carried by differentiation, not the cut.

So over all 32 runs measured — 12 engines × 2 failing seeds, plus 4 engines × 2
passing seeds — the ordering holds **31 of 32**, not universally. The partition
still wanders (12 distinct sizes, k = 4–16) on the exception, so the mechanism is
present there; it is simply not the dominant term that run.

This is the check that was worth running: the exception exists, and it exists on
a passing seed, which is exactly where a claim built only from failing seeds
would not have looked.

### The carve-out: ConsciousnessEngine is not being measured on 32 cells

`ConsciousnessEngine` is the one engine where the wandering partition is absent,
and the table shows it: **k takes 2 values, 0 and 1**, against 10–14 for every
other engine. The reason is not a property of its dynamics.

```
engine                asked for   get_hiddens() exposes
ConsciousnessEngine      32           (2, 128)      <-- two cells
PairField                32          (32, 128)
NarrativeEngine          32          (32, 128)
MitosisEngine            32          (32, 128)
Trinity                  32          (32, 128)
```

**The gate constructs it with 32 cells and reads Φ off 2.** Everything anomalous
in its row follows arithmetically from n = 2, not from the engine:

| observation | cause |
|---|---|
| k ∈ {0, 1} only | with 2 rows the only cuts are trivial |
| `n_pairs_sampled` = 1 | one pair exists |
| complexity exactly 0.0000 | `np.std` of a single MI value returns 0 by construction |
| Φ ≈ cut (0.0267 vs 0.0264) | `min_partition_mi / (n−1)` with n−1 = 1 is the cut itself |
| Φ mean 0.0267 vs 1.4–2.3 | a two-cell MI reading, measured at three checkpoints as 0.00000 |
| differentiation 1.0125 | the two rows are dissimilar — it says nothing about 32 cells |

> **Correction.** An earlier reading of this row said ConsciousnessEngine's
> "cells are essentially orthogonal" and "its population never integrates at
> all". That was wrong: differentiation ≈ 1.0 is computed over two rows, and
> complexity 0.0000 is forced by there being one MI value, not measured. The
> engine's actual 32-cell population was never in the calculation.

So its five failures are not the cut mechanism, but neither are they the clean
"fails honestly" they first looked like. **The gate is scoring a two-cell
projection of a 32-cell engine**, which is the same interface defect as
`PairFieldEngine.get_hiddens()` returning A only (§3) and `set_hiddens` being a
partial restore — one order of magnitude worse. Whether ConsciousnessEngine
integrates is not established either way by anything in this document; the
measurement never reached it.

**Scope, corrected.** The shared cut mechanism is established for **eleven of
twelve** engines, and ConsciousnessEngine is excluded by measurement rather than
by omission. But it is excluded because it was never measured on its own
population, not because it is a counterexample — so "PERSISTENCE is the
estimator" holds for eleven of twelve, and the twelfth is undetermined rather
than sound.

`NarrativeEngine` passes here, but not because its trajectory differs in kind —
only at this grid and this scale:

| engine | 32c/32d/128h | 32c/64d/128h | 256c/64d/128h |
|---|---|---|---|
| PairField | 4/5 | 4/5 | 4/5 |
| NarrativeEngine | 5/5 | **2/5** | **4/5** |

Narrative's Φ at 32c/64d/128h, seed 43: 1.965 → 1.957 → 0.793 → 1.417 → 1.345 →
1.459 → 0.723 → 1.889 → 0.902 → 1.454. Same two-level shape, same jump sizes; its
jitter cv at 256 cells is 0.04–0.27 against PairField's 0.07–0.28. There is no
engine with a qualitatively different trajectory to compare against, because the
shape is the estimator's.

## 7. Repulsion strength is not implicated

All five gate seeds, 32c/64d/128h. `strength = 0.000` is the uncoupled control:
A and G never interact.

| strength | pass | seeds failing | peak \|h\| | A–G sep | Φ mean | Φ cv |
|---|---|---|---|---|---|---|
| 0.000 (uncoupled) | 4/5 | 46 | 6.31 | 84.62 | 1.767 | 0.171 |
| 0.005 | 3/5 | 42, 44 | 6.13 | 85.88 | 1.826 | 0.156 |
| 0.010 | 3/5 | 44, 45 | 5.98 | 87.80 | 1.750 | 0.196 |
| 0.020 | 5/5 | — | 6.89 | 89.70 | 1.864 | 0.162 |
| **0.030 (default)** | 4/5 | 42 | 6.49 | 95.43 | 1.567 | 0.199 |
| 0.050 | 3/5 | 42, 43 | 6.55 | 103.56 | 1.305 | 0.221 |

The pass count is not ordered by strength, and the uncoupled control **ties the
default at 4/5**, failing a different seed. A column on which the non-pair scores
what the pair scores carries nothing about the coupling. The only quantities that
move monotonically with strength are the ones that should: separation rises
84.6 → 103.6 and Φ mean falls 1.77 → 1.31.

At 256c/64d/128h the same sweep answers differently again, and inverts the
control:

| strength | pass | seeds failing | peak \|h\| | A–G sep | Φ mean |
|---|---|---|---|---|---|
| 0.000 (uncoupled) | **0/5** | all five | 6.69 | 239.88 | 13.577 |
| 0.010 | 4/5 | 45 | 5.83 | 244.85 | 13.904 |
| 0.020 | 4/5 | 44 | 5.71 | 255.28 | 12.616 |
| **0.030 (default)** | 4/5 | 44 | 6.74 | 270.08 | 12.372 |

No strength reaches 5/5 at 256 cells, 0.020 is no better than the default, and
the uncoupled control goes from 4/5 to 0/5 with cell count.

And `strength = 0.020`'s 5/5 does not survive the ruler. Strength × grid, five
seeds each, 25 verdicts per row:

| strength | −2 | −1 | +0 | +1 | +2 | total | A–G sep | peak \|h\| |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 3/5 | 3/5 | 4/5 | 3/5 | 4/5 | 17/25 | 84.50 | 6.31 |
| 0.005 | 4/5 | 1/5 | 3/5 | 4/5 | 4/5 | 16/25 | 85.83 | 6.13 |
| 0.010 | 3/5 | 4/5 | 3/5 | 3/5 | 3/5 | 16/25 | 87.78 | 5.98 |
| 0.020 | 4/5 | 4/5 | **5/5** | **5/5** | 4/5 | 22/25 | 89.72 | 6.89 |
| **0.030 (default)** | 4/5 | **5/5** | 4/5 | 4/5 | 3/5 | 20/25 | 95.30 | 6.49 |
| 0.050 | 4/5 | 4/5 | 3/5 | 3/5 | 4/5 | 18/25 | 103.42 | 6.55 |

22 against 20 out of 25 is a difference of two verdicts; five seeds cannot
resolve it. **No strength in 0.000–0.050 passes all five seeds robustly.** Every
strength stayed bounded (peak 5.71–6.89) and coupled (separation 84.5–270.1), so
none of these came from a diverged or uncoupled pair — they are honest readings
of a quantity that does not carry the signal.

**No change to `pairfield_engine.py` is warranted, and none was made.
`DEFAULT_STRENGTH = 0.03` is not implicated.**

## 8. The uncoupled control does not isolate the coupling

This is a finding about the control, not a caveat on the table above.

`strength = 0.000` is the row every other row in §7 is read against: A and G run
side by side and never interact, so whatever it scores is what the *absence* of
the pairing scores. Its verdict:

| scale | uncoupled (0.000) | default (0.030) |
|---|---|---|
| 32c / 64d / 128h | **4/5** | 4/5 |
| 32c / 64d / 128h, across 5 grids | **17/25** | 20/25 |
| 256c / 64d / 128h | **0/5** | 4/5 |

At 32 cells the non-pair scores what the pair scores. At 256 cells the same
non-pair scores zero. **A control whose verdict inverts with the scale it is run
at is not isolating the variable it was built to isolate** — so the strength
column was never interpretable at *either* size, not merely underpowered at both.

That matters beyond this sweep. The uncoupled control is the only thing standing
between "the pair passes" and "two engines in a room pass", and it was the check
this investigation was told to run first, precisely because three earlier
experiments this session produced clean scores from systems that were never
coupled. It ran, it was bounded, its separation was non-zero — and it still
carries no information, because what it reports depends on a parameter that has
nothing to do with coupling. Same family as everything else here: a check that
looks like it isolates something and does not.

## 9. The swept min-cut is a diagnostic, NOT a fix

`bench_v2.py:225-237` takes the **sign** of the Fiedler vector as the partition,
inside a function named `_minimum_partition`. That is not a minimum over
anything.

### How far off is it? 1.8–6.7×, and the spread is the construction

Three independent measurements, listed with their conditions because they do not
agree on magnitude and the disagreement is informative:

| measured by | construction | reference | overshoot |
|---|---|---|---|
| team-lead | n = 12, `BenchEngine(12,32,64,32,4)`, seed 42 | exhaustive (2¹¹ masks) | 1.82–2.76×, mean **2.14×** |
| persistence-audit | n = 12, real-engine MI matrices | exhaustive | **3.0–3.4×** |
| this doc | n = 32, PairField 32c/64d/128h, seed 46, 22 steps | sweep on same ordering | 2.83–6.66×, mean **3.56×** |

Same direction, different magnitude. Two things account for the spread rather
than any disagreement about the mechanism:

- **The reference differs.** The first two compare against the exhaustive true
  minimum; this doc compares against the Fiedler sweep, which is an *upper bound*
  on the minimum. So the 2.83–6.66× figures are **lower bounds** on the real
  overshoot, not estimates of it.
- **The overshoot depends on how close the population sits to a tie**, which is
  the same property §2 identifies as varying with hidden width. A single headline
  number would be reporting one construction as if it were a constant.

**The sweep is a good approximation to the minimum, not provably the minimum.**
Measured at n = 12 against exhaustive ground truth it matched on 4 of 6 steps and
missed slightly on two (7.085 against 6.610; 6.657 against 6.251); an independent
run reported 6/6 exact. What *is* guaranteed is the one-sided bound — the sign
cut is among the sweep's candidates, so `sweep ≤ sign` always — and every claim
here rests only on that.

### What refining it does to the variance

Replacing the sign cut with the sweep removes the jumping entirely.
32c/64d/128h, 5 seeds × 4 grids = 20 verdicts:

| system | shipped | Φ cv | max/min | swept | Φ cv | max/min |
|---|---|---|---|---|---|---|
| PairField | 16/20 | 0.228 | 2.31 | **20/20** | 0.066 | 1.26 |
| NarrativeEngine | 10/20 | 0.193 | 2.04 | **19/20** | 0.061 | 1.23 |
| HEAP | 0/20 | 0.443 | 5.76 | 0/20 | 0.169 | 1.70 |
| DECOUPLED | 0/20 | 0.173 | 1.87 | 0/20 | 0.157 | 1.72 |
| DEAD | 0/20 | 0.000 | 1.00 | 0/20 | 0.000 | 1.00 |
| NOISE | 0/20 | 0.159 | 1.80 | 0/20 | 0.172 | 1.73 |
| CLONE | 0/20 | 0.054 | 1.20 | 0/20 | 0.054 | 1.20 |
| SCRAMBLE | 0/20 | 0.813 | 38.43 | 0/20 | 0.637 | 11.19 |
| LINEAR | 0/20 | 0.194 | 1.87 | 0/20 | 0.121 | 1.48 |

All seven negative controls stay out. That establishes the diagnosis: the
trajectory is the cut heuristic, and removing the heuristic removes the
trajectory.

**It must not be landed.** Matched-coupling direction test, n = 32, dim 512,
**25 seeds**, SPLIT = 2 independent sources vs RING = 1 shared source, noise `s`
held equal across the arms so the only thing varying is the number of sources —
the confound-free form from `phi-rs-direction.md` §5. A measure with the right
direction ranks the integrated RING above the disconnected SPLIT, i.e. ratio > 1.

| noise s | cut | RING wins | median RING/SPLIT | mean | min | max |
|---|---|---|---|---|---|---|
| 0.3 | shipped | **25/25** | **5.77×** | 5.77 | 2.66 | 10.22 |
| 0.3 | swept | **12/25** | **0.97×** | 1.01 | 0.54 | 1.46 |
| 0.5 | shipped | **25/25** | **9.46×** | 9.37 | 2.64 | 13.92 |
| 0.5 | swept | 24/25 | **1.57×** | 1.53 | 0.87 | 2.09 |
| 1.0 | shipped | **25/25** | **7.32×** | 7.87 | 4.92 | 10.77 |
| 1.0 | swept | 24/25 | **1.16×** | 1.17 | 0.95 | 1.34 |

**The sweep does not invert the direction — it collapses it.** The shipped sign
cut gets the sign right on 75 of 75 runs and separates the two constructions by
5.8–9.5×. The sweep keeps the sign at s = 0.5 and s = 1.0 (24/25 each) but at a
margin of 1.16–1.57×, and at s = 0.3 it loses the signal entirely: 12/25 is a
coin flip and the median 0.97 is indistinguishable from no preference at all.

> **Correction to an earlier 5-seed reading.** At 5 seeds this table showed
> "SPLIT 1.14×" at s = 0.3 and "RING 1.13×" at s = 1.0, and was reported as the
> sweep flipping the direction *the wrong way*. That was a draw from a spread
> that runs 0.54–1.46 at that noise level. The sweep does not reverse the
> ranking; it destroys the margin. The conclusion is unchanged and the reason is
> narrower — do not land it because it stops discriminating, not because it
> discriminates backwards.

The component split at `s = 0.0` shows the cause: driving the cut to its minimum
makes it a near-zero singleton for almost any population, `spatial_phi`
collapses, and Φ becomes the `0.1 × complexity` term — 95–100% of Φ in both arms.
Under the sweep at n = 32 the constructions pin together at that floor:

| construction | shipped | swept |
|---|---|---|
| IDENTICAL | 0.0049 | 0.0049 |
| SPLIT | 0.2002 | 0.2002 |
| RING | 0.1888 | **0.0235** |
| MID | 0.1759 | **0.0255** |
| INDEPENDENT | **0.2143** | **0.0265** |

RING, MID and INDEPENDENT become indistinguishable. **That is why the variance
fell.** Φ was flattened toward a floor, not stabilised — the bad min-cut
accidentally retains information the true minimum discards.

So the sweep buys the variance and pays with the discrimination. Reported,
not applied.

### The IDENTICAL-vs-INDEPENDENT objection measures the cut, not Φ

`phi-rs-direction.md` records both phi-rs readings preferring IDENTICAL (fully
collapsed) by 90× and 104×, and an independent check at n = 8, dim 4000 put the
swept reading at 10,886× — which, if it were `bench_v2`'s Φ, would sink the
sweep on direction alone. Rebuilt here at the same n and dim over 5 seeds, with
every intermediate printed:

| case | cut/(n−1) | differentiation | `bench_v2` Φ |
|---|---|---|---|
| IDENTICAL | 3.0952 | **0.0000** | 0.000195 |
| INDEPENDENT | 0.0003 | 0.9999 | 0.000508 |

| ratio IDENTICAL / INDEPENDENT | |
|---|---|
| `cut/(n−1)` alone | **10,460×** |
| `bench_v2` Φ | **0.385×** |

The 10,460× reproduces the reported 10,886× to within seed noise — **that number
is real, and it is the cut term.** It is not Φ. `bench_v2` multiplies the cut by
`differentiation = 1 − mean_cos`, which is exactly 0.0000 for identical rows, so
the collapse preference is annihilated before Φ is formed. Per-seed Φ ratios:
0.380, 1.172, 0.387, 0.766, 0.149 — scattered around and below 1, and both arms
sit at the floor (0.0002 against 0.0005), so the honest reading is that Φ
separates neither, not that it prefers INDEPENDENT.

Two further mismatches in that test, both making it the wrong instrument for
this question:

- **At n = 8, `bench_v2` never uses the sign cut.** `_minimum_partition` branches
  to the exhaustive search over all 2⁸ partitions at `n <= 8`. Measured on one
  matrix: exhaustive 20.9124, sweep 20.9124 (identical — the sweep finds the true
  minimum), sign cut 35.8525 (a quantity `bench_v2` does not compute at this n).
  A sign-cut-vs-sweep comparison at n = 8 is not comparing anything the shipped
  gate does.
- **dim 4000 with 16×16 bins** is a different estimator regime from the gate's
  hidden 128.

So the objection does not land against `bench_v2` Φ as stated. **The conclusion
it was raised to support survives anyway, on different evidence** — the
matched-coupling test above, at n = 32 where the sign cut *is* what runs. Do not
land the sweep; the reason is RING/SPLIT, not IDENTICAL/INDEPENDENT.

`bench_v2`'s own direction defect is a third thing again: the **shipped** sign cut
ranks INDEPENDENT top at n = 32 (0.2143, above SPLIT 0.2002 and RING 0.1888) —
nothing-to-join scoring highest.

## 10. What would refute this

Named in advance, each run.

**"The 0.845 is a real dynamical event."** Refuted by §2: Φ is below 1.0 on 64%
of the steps in that window, differentiation and complexity are flat to ±1.5%,
and only the chosen partition moves. Refuted again by §3 at a second scale, where
separation, norms and coupling are flat through the whole window.

**"The cut-dominance ordering is an artefact of looking only at failing
seeds."**  Partly right, and §6 records it: on passing seeds the ordering holds
7 of 8, and QuantumEngine seed 43 breaks it outright (cv(cut) 0.113 below
cv(diff) 0.126, corr(Φ,cut) −0.051). The claim is now 31/32, not universal.

**"It is estimator randomness."** Refuted by §4, in the other direction: at 32
rows the pair set is exhaustive, the constructor seed is dead, and the singleton
is deterministic and order-independent. The jitter sd at the failing checkpoint
is 0.004. Φ is a deterministic function of the state that happens to be
discontinuous.

**"Narrative has a genuinely stable shape."** Refuted by §6: 5/5 here, 2/5 at
32c/64d/128h, 4/5 at 256c, with the same two-level trajectory throughout.

**"A lower strength stabilises the pair."** Refuted by §7: unordered in strength
at both scales, the uncoupled control ties the default at 32 cells, and 0.020's
5/5 holds on only two grids of five.

**"The uncoupled control validates the strength column."** Refuted by §8: it
scores 4/5 at 32 cells and 0/5 at 256, so it is not tracking the coupling.

**"The sweep prefers a collapsed population, which sinks it."** Not refuted, but
mis-attributed, and §9 shows where: 10,460× reproduces the reported figure, and
it is the cut term. `bench_v2` multiplies by `differentiation = 0.0000` for
identical rows, giving a Φ ratio of 0.385×. The sweep should still not be landed
— for the RING/SPLIT reason, not this one.

**"The swept min-cut is the fix."** Refuted by §9 — the control that killed it.
It removes the variance by flattening Φ onto its complexity floor, and destroys
the shipped formula's one correct directional behaviour.

## 11. Verdict

PairFieldEngine is not the cause of its own PERSISTENCE failure. Through the
window where the gate reports a collapse, the pair is bounded (peak \|h\|
4.37–5.79 against a 1e4 guard), coupled (A–G separation 244–257 at the scale
where it was traced per-step, never near zero), and flat in every quantity that
describes it, while the partition `_minimum_partition` selects jumps between a
4-cell and an 11-cell cut from one step to the next.

The defect is in `PhiIIT._minimum_partition` (`bench_v2.py:225-237`) and in the
shape of `_verify_persistence`'s pass rule (`bench_v2.py:1794-1796`), and it
applies across the fleet: ten of twelve engines are blocked by it, on nine
different seed sets, and **eleven of twelve share the mechanism** — the cut is
the highest-variance factor in 31 of the 32 runs measured, the exception being
QuantumEngine at the passing seed 43. Seed 46 fails by 0.006 with
no drop in its trajectory at all.

ConsciousnessEngine is the one exception, and not because it fails honestly: the
gate builds it with 32 cells and reads Φ off the 2 its `get_hiddens()` exposes
(§6). Its verdict is undetermined, not sound.

### A PASS from this gate does not distinguish an engine from the ones it beat

Everything above shows the gate's FAIL verdicts are unreliable. NarrativeEngine
shows the PASS verdicts are too. At the gate's own scale it scores **5/5** and is
the reason it was published as the one deployable engine of twelve — and its
decomposition is indistinguishable from the engines it beat:

| | NarrativeEngine (PASS 5/5) | PairField (FAIL 4/5) |
|---|---|---|
| distinct partition sizes k | 13, spanning 4–16 | 13, spanning 4–16 |
| cv(cut) | 0.268 / 0.303 | 0.322 / 0.229 |
| cv(differentiation) | 0.119 / 0.128 | 0.137 / 0.153 |
| corr(Φ, cut) | +0.717 / +0.518 | +0.827 / +0.282 |

Same wandering partition, same cut-dominated variance, opposite verdict. And its
own verdict moves with the scale it is run at — 5/5 at 32c/32d/128h, **2/5** at
32c/64d/128h, 4/5 at 256c.

**So the PASS row carries no more information than the eleven BLOCKED rows next
to it.** Passing is not evidence that the mechanism is absent; it is evidence
that the lottery landed well. Any deployment decision resting on which engines
this condition passed is resting on the same coin the FAIL verdicts came from.

`pairfield_engine.py` is unchanged and should stay unchanged on this evidence.
The repair is not obvious either: the one candidate measured here fixes the
variance and breaks the direction, so `_minimum_partition` cannot be corrected on
its own.

---

## Appendix — upstream: the integration axis has no same-state baseline

Outside this document's question, but upstream of every number in it, so it was
measured rather than assumed.

`_three_axes` computes `integration = mean(|others moved| / |kick|)` by comparing
a nudged run against an un-nudged one (`bench_v2.py:1600`). That *is* a
comparison against the same step without the kick — but the two runs are separate
`process` calls, so any randomness an engine draws per call lands in the
numerator with nothing subtracting it. The missing baseline is the same
comparison with **no kick**: restore one state twice, step twice, and see how far
the others moved anyway.

Shipped value, that null, and the difference, for every control at both scales
(bar = 0.001):

| system | scale | shipped | null | corrected | null share | shipped | corrected |
|---|---|---|---|---|---|---|---|
| PairField | 32c | 0.06027 | 0.00273 | 0.05754 | 4.5% | PASS | PASS |
| NarrativeEngine | 32c | 0.05870 | 0.00021 | 0.05849 | 0.4% | PASS | PASS |
| HEAP | 32c | 0.00000 | 0.00000 | 0.00000 | — | FAIL | FAIL |
| DECOUPLED | 32c | 0.13850 | 0.00471 | 0.13379 | 3.4% | PASS | PASS |
| DEAD | 32c | 0.00000 | 0.00000 | 0.00000 | — | FAIL | FAIL |
| **NOISE** | 32c | 1.60321 | **1.59132** | 0.01189 | **99.3%** | PASS | PASS |
| CLONE | 32c | 4.09388 | 0.00000 | 4.09388 | 0.0% | PASS | PASS |
| **SCRAMBLE** | 32c | 1.05443 | **0.74940** | 0.30503 | **71.1%** | PASS | PASS |
| LINEAR | 32c | 0.02610 | 0.00000 | 0.02610 | 0.0% | PASS | PASS |
| **PairField** | 256c | 0.02819 | **0.00718** | 0.02102 | **25.5%** | PASS | PASS |
| NarrativeEngine | 256c | 0.02336 | 0.00031 | 0.02306 | 1.3% | PASS | PASS |
| HEAP | 256c | 0.00000 | 0.00000 | 0.00000 | — | FAIL | FAIL |
| DECOUPLED | 256c | 0.09791 | 0.01301 | 0.08490 | 13.3% | PASS | PASS |
| DEAD | 256c | 0.00000 | 0.00000 | 0.00000 | — | FAIL | FAIL |
| **NOISE** | 256c | 4.71855 | **4.72332** | **−0.00477** | **100.1%** | PASS | **FAIL** |
| SCRAMBLE | 256c | 2.25757 | 2.20446 | 0.05311 | 97.6% | PASS | PASS |
| CLONE | 256c | 12.21170 | 0.00000 | 12.21170 | 0.0% | PASS | PASS |
| LINEAR | 256c | 0.00936 | 0.00000 | 0.00936 | 0.0% | PASS | PASS |

**The defect is real.** NOISE's integration reading is 99.3% nondeterminism at 32
cells and 100.1% at 256 — the axis is measuring the engine's RNG, not
interaction. SCRAMBLE is 71.1% and 97.6%. And PairField's own reading is **25.5%
null at 256 cells**, so this is not confined to the controls.

**Its effect on the current control set is narrow.** Subtracting the null changes
exactly one verdict of the eighteen above: NOISE at 256 cells goes PASS → FAIL
(corrected integration −0.00477). Everything else keeps its verdict — HEAP and
DEAD already fail on both, and CLONE, SCRAMBLE, DECOUPLED and LINEAR still clear
the bar corrected, being rejected by the identity and response axes instead.

So the value of subtracting the null is **closing a constructed bypass**, not
correcting today's verdicts: a HEAP whose parts never interact, plus a private
RNG stream that `set_hiddens` cannot rewind, produces a large integration reading
out of nothing, and the shipped axis has no way to tell that from interaction.
That is the same shape as the `PairFieldEngine.set_hiddens` restoring A only —
the axis never establishes a same-state baseline for any engine.

Not applied. This is one line in `_three_axes`, but it is an axis every engine
and every recorded verdict depends on.

---

Reproduce. Measurement scripts are session scratch under
`$SP = /private/tmp/claude-501/-Users-mini-dancinlab-anima-lab-1/d0396916-1383-4a82-b215-02ece85f6789/scratchpad`;
none of them modify anything in the repo.

```bash
.venv/bin/python $SP/traj.py PairField 32 32 128 --noise           # §1, §4
.venv/bin/python $SP/micro.py 32 32 128 44 680 730                 # §2
.venv/bin/python $SP/trace.py 256 64 128 44 25 --noise             # §3
.venv/bin/python $SP/grid_shift.py PairField 32 32 128             # §5
.venv/bin/python $SP/traj.py NarrativeEngine 32 64 128             # §6
.venv/bin/python $SP/sweep_strength.py 32 64 128 0.0 0.005 0.01 0.02 0.03 0.05
.venv/bin/python $SP/sweep_grid.py 32 64 128 0.0 0.005 0.01 0.02 0.03 0.05
.venv/bin/python $SP/corrected.py 32 64 128 --shifted              # §9 variance
.venv/bin/python $SP/direction3.py                                 # §9 direction, 25 seeds
.venv/bin/python $SP/integ_null.py                                 # appendix
.venv/bin/python $SP/reconcile5.py                                 # §9 reconciliation
.venv/bin/python $SP/mechanism.py OscillatorLaser 32 32 128 44      # §6, 12 engines x seeds 42-46
```

Runtime note: `corrected.py` takes ~15 min per system pair and the 256-cell
sweeps ~10 min per strength on a loaded machine. Run them in the background and
read the file; do not pipe them to `tail`.
