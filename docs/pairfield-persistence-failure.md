# PERSISTENCE fails on the Φ estimator, not on the pair

`PairFieldEngine` was landed 5/5 DEPLOYABLE, then dropped to **4/5 — BLOCKED on
PERSISTENCE** when the gate was tightened to require all five seeds. The reported
trajectory holds near 2.0 for six checkpoints, drops to 0.845, recovers to 2.095,
and decays to 1.216. The question put to this investigation was what causes the
drop at the seventh checkpoint, and whether it belongs to the pair or to one side.

**It belongs to neither.** Nothing in the pair changes at that step. The drop is
the Φ estimator's minimum-cut heuristic flickering between two answers at a
per-step timescale, and the `recovers` rule reading one flicker against the max
of five.

Everything below was run with `.venv/bin/python`. Every score is preceded by the
bound (`engine.peak`) and the coupling (A–G separation), because a clean number
from a diverged or uncoupled pair says nothing.

---

## 1. Nothing turns at step 700

Per-step trace, seed 44, 256 cells / dim 64 / hidden 128, Φ every 25 steps.
`phi_A` is what the gate reads — `PairFieldEngine.get_hiddens()` returns **A
only**, so the condition never sees G directly.

| step | Φ(A) | Φ(G) | A–G sep | \|A\| | \|G\| | coupling A | coupling G |
|---|---|---|---|---|---|---|---|
| 100 | 11.41 | 10.71 | 244.18 | 162.46 | 161.37 | 0.6967 | 0.0796 |
| 200 | 7.59 | 10.87 | 256.93 | 172.38 | 164.25 | 0.9454 | 0.1123 |
| 300 | 9.72 | 11.56 | 255.59 | 170.84 | 163.11 | 0.9867 | 0.1376 |
| 400 | **5.55** | 9.86 | 255.04 | 173.02 | 161.66 | 0.9946 | 0.1582 |
| 500 | 8.54 | 11.19 | 256.10 | 173.84 | 160.26 | 0.9961 | 0.1766 |
| 600 | **5.37** | 11.79 | 253.08 | 173.71 | 159.02 | 0.9961 | 0.1935 |
| **700** | **5.46** | 9.83 | 254.92 | 176.20 | 160.80 | 0.9961 | 0.2094 |
| 800 | 10.12 | 9.03 | 254.50 | 176.01 | 162.61 | 0.9961 | 0.2246 |
| 900 | **5.80** | 11.78 | 253.43 | 175.07 | 163.35 | 0.9961 | 0.2375 |
| 1000 | 7.70 | 10.56 | 250.74 | 175.91 | 160.56 | 0.9961 | 0.2500 |

Bounded: peak \|h\| = 4.86 over the run, against the 1e4 guard. Coupled: A–G
separation never leaves 244–257.

Read down the columns. Separation is flat. \|A\| drifts up 8% over 900 steps and
\|G\| is flat. A's coupling matrix saturates against its own `clamp(-1, 1)` by
step ~425 and is constant from there; G's grows linearly and is still growing at
step 1000. **No quantity has an event at step 700.** Φ(A) reads 5.55 / 5.37 /
5.46 / 5.80 at steps 400 / 600 / 700 / 900 — the seventh checkpoint is not
distinguished from three others that the rule happened not to compare against.

All 40 sampled points, Φ(A), seed 44, 256c — the gate reads only every fourth
column of this:

```
   12.5 |
        |   *                *     *     *     *
   10.8 |         *                    *      *
        |      *   **      *            * **
    9.1 |    *   *     *      **
        |  *  *      **     *    **  *
    7.4 |       *                     *      *  *
        |                **
    5.7 |**             *       *   *       *
        |
    4.0 |
        +----------------------------------------
         step 25                        step 1000
                                    ^ step 700
```

There is no trajectory here. There is a stationary series with a spread, sampled
ten times.

## 2. The estimator's own spread covers the drop

Φ was recomputed 20× on each **stored checkpoint state** under a 1e-6 relative
jitter. That perturbation is dynamically nothing, but it re-draws the debias
shuffles and (above n = 32) the sampled pair set — so the spread is the
estimator's contribution, with the state held fixed.

| step | Φ as measured | mean of 20 | sd | min | max | cv |
|---|---|---|---|---|---|---|
| 100 | 11.414 | 10.297 | 0.825 | 8.815 | 11.414 | 0.080 |
| 400 | 5.548 | 5.654 | 1.048 | 2.797 | 7.512 | 0.185 |
| 600 | 5.367 | 8.526 | 1.875 | 5.367 | 12.003 | 0.220 |
| **700** | **5.462** | **8.389** | **1.936** | 5.777 | 13.395 | 0.231 |
| 900 | 5.796 | 8.449 | 2.348 | 4.156 | 12.888 | 0.278 |
| 1000 | 7.700 | 9.201 | 1.561 | 6.637 | 11.594 | 0.170 |

The step-700 reading of 5.462 sits **1.5 sd below the mean of its own
re-measurements**. Mean sd over the ten checkpoints is 1.479. The same probe on
`NarrativeEngine` at the same scale, over five seeds, gives sd 0.66–3.19 and cv
0.05–0.27, mean sd **1.535** — indistinguishable. The noise is a property of Φ at
this scale, not of PairField.

## 3. What the estimator is actually doing

At 32 cells the pair set is exhaustive and the jitter probe is small (mean sd
0.046), yet Φ still swings 0.55 ↔ 1.64 between checkpoints. So the swings are not
the debias shuffles. Step-resolution microscope, seed 46, 32c/64d/128h, steps
791–812, decomposing `Φ = min_cut/(n−1) × differentiation + 0.1 × complexity`:

`best cut` below is the minimum over the threshold sweep along the *same* Fiedler
ordering. The sign cut is one of that sweep's candidates, so `best ≤ sign` holds
by construction, and since the true minimum is in turn ≤ `best`, every gap in
these columns is a lower bound on how far the sign cut is from the minimum.

| step | Φ | Fiedler sign cut (used) | best cut on same ordering | differentiation | complexity | cut size k |
|---|---|---|---|---|---|---|
| 795 | 0.597 | 36.44 | 12.71 | 0.4715 | 0.4221 | 4 |
| **796** | **1.398** | **89.69** | 14.04 | 0.4689 | 0.4145 | **9** |
| 797 | 0.577 | 34.68 | 12.27 | 0.4781 | 0.4247 | 4 |
| 799 | 0.564 | 33.72 | 11.52 | 0.4787 | 0.4285 | 4 |
| **800** | **1.343** | **84.56** | 12.70 | 0.4768 | 0.4237 | **9** |
| 801 | 0.604 | 36.56 | 12.26 | 0.4761 | 0.4258 | 4 |
| 805 | 0.658 | 40.56 | 13.04 | 0.4706 | 0.4191 | 4 |
| **806** | **1.317** | **84.45** | 13.69 | 0.4681 | 0.4171 | **8** |
| 807 | 0.640 | 38.80 | 12.46 | 0.4779 | 0.4188 | 4 |

Across the full 40-step window: differentiation stays in 0.4681–0.4837 and
complexity in 0.4064–0.4285 — both flat to ±2%. The best cut on the ordering
stays in 11.5–14.0 and isolates a single cell at every one of the 22 steps.
Everything that moves in Φ is the cut the estimator *used*.

`bench_v2.py:225-237` takes the **sign** of the Fiedler vector as the partition.
That is not the minimum cut. It exceeds the best cut on its own ordering by 3–7×,
and it is bistable: it settles on a 4-cell isolation (cut ≈ 35) or an 8–9-cell
one (cut ≈ 85), with the occasional intermediate (step 793, k = 6, cut 58), and
it flips between them from one step to the next.

```
   95.0 |                                              * = Fiedler sign cut
        |          *       *           *               = = best cut, same ordering
        |
   71.2 |
        |
        |    *
   47.5 |
        |  *   * *     *     * * * * *   * * * * * *
        |*           *   *
   23.8 |
        |= = = = = = = =   = = = = = = = = = = = = =
        |                =
    0.0 |
        +--------------------------------------------
         791                                    812
```

**So the spikes are the artefact and the low readings are the baseline.** The
"drop to 0.845" is the estimator briefly getting closer to the right answer.

## 4. The `recovers` rule turns a flicker into a verdict

`_verify_persistence` passes on `monotonic OR recovers`. In every run reported
here — PairField and NarrativeEngine, five seeds, three scales — `monotonic` came
back `False` without exception, so the condition reduces to `recovers` alone:

```python
recovers = phi_history[-1] >= max(phi_history[:5]) * 0.8
```

one draw of a bistable quantity against 0.8 × the max of five draws. Taking a max
of five biases the bar upward by roughly the spread, so the margin is a coin
toss.

The same 1000-step run scored on checkpoint grids shifted by a few steps — a
shift of one step is dynamically nothing. PairField, 32c/64d/128h:

| grid offset | 42 | 43 | 44 | 45 | 46 | total |
|---|---|---|---|---|---|---|
| −2 | PASS | PASS | FAIL | PASS | PASS | 4/5 |
| −1 | PASS | PASS | PASS | PASS | PASS | **5/5** |
| **+0 (shipped)** | FAIL | PASS | PASS | PASS | PASS | 4/5 |
| +1 | FAIL | PASS | PASS | PASS | PASS | 4/5 |
| +2 | FAIL | PASS | PASS | FAIL | PASS | 3/5 |
| +5 | FAIL | PASS | FAIL | FAIL | PASS | 2/5 |

Seed 42 passes at −1 and fails at 0, on bar 1.770 vs final 2.049 against bar
1.714 vs final 1.690. Seed 44 fails at −2, passes at −1 through +2, fails at +5.
**The verdict is a property of where the ruler falls, not of the run.**

The same arbitrariness shows up along the scale axis. Identical protocol, five
seeds:

| scale | PairField | which seeds fail |
|---|---|---|
| 32c / 32d / 64h | 5/5 | — |
| 32c / 64d / 128h | 4/5 | 42 |
| 256c / 64d / 128h | 4/5 | 44 |
| reported by the gate | 4/5 | 44, 46 |

Four runs of one condition on one engine, four different answers about which
seeds are the bad ones.

## 5. The alternative hypothesis, tested: it is the condition, not the engine

`NarrativeEngine` was named as the only engine that passes PERSISTENCE on all
five seeds. It does not.

| engine | 32c/64d/128h | 256c/64d/128h |
|---|---|---|
| PairField | 4/5 (fails 42) | 4/5 (fails 44) |
| NarrativeEngine | **2/5** (fails 43, 44, 45) | **4/5** (fails 45) |

Narrative's Φ series at 32c, seed 43: 1.965 → 1.957 → 0.793 → 1.417 → 1.345 →
1.459 → 0.723 → 1.889 → 0.902 → 1.454. Same shape, same bistable jumps, same
lottery. Its jitter cv at 256 cells is 0.04–0.27, matching PairField's 0.07–0.28.

There is no engine with a qualitatively different trajectory shape to compare
against, because the shape is the estimator's.

## 6. Repulsion strength is not implicated

The sweep was run on all five gate seeds, 32c/64d/128h. `strength = 0.000` is the
uncoupled control: A and G never interact.

| strength | pass | seeds failing | peak \|h\| | A–G sep | Φ mean | Φ cv |
|---|---|---|---|---|---|---|
| 0.000 | 4/5 | 46 | 6.31 | 84.62 | 1.767 | 0.171 |
| 0.005 | 3/5 | 42, 44 | 6.13 | 85.88 | 1.826 | 0.156 |
| 0.010 | 3/5 | 44, 45 | 5.98 | 87.80 | 1.750 | 0.196 |
| 0.020 | 5/5 | — | 6.89 | 89.70 | 1.864 | 0.162 |
| **0.030 (default)** | 4/5 | 42 | 6.49 | 95.43 | 1.567 | 0.199 |
| 0.050 | 3/5 | 42, 43 | 6.55 | 103.56 | 1.305 | 0.221 |

The pass count is not ordered by strength, and **the uncoupled control at
strength 0.000 scores 4/5 — better than the default.** Two engines that never
touch are not a pair; a strength read off this column would be read off the
lottery. The only quantities that do move monotonically with strength are the
ones that should: separation rises 84.6 → 103.6 and Φ mean falls 1.77 → 1.31.

`strength = 0.020` scoring 5/5 is the same coincidence as `−1` scoring 5/5 in the
grid table. See §8 for whether it survives moving the ruler.

**No change to `pairfield_engine.py` is warranted. `DEFAULT_STRENGTH = 0.03` is
not implicated in this failure.**

## 7. What a correct minimum cut does — measured, not applied

Only `PhiIIT._minimum_partition` was changed, and only to sweep the threshold
along the same Fiedler ordering instead of taking its sign. That refinement is
strictly no worse than the sign cut, costs O(n²), and is what "minimum
information partition" means. Nothing else was touched.

32c/64d/128h, 5 seeds × 4 grid offsets = 20 verdicts per system:

| system | shipped: pass | Φ cv | max/min | swept: pass | Φ cv | max/min |
|---|---|---|---|---|---|---|
| PairField | 16/20 | 0.228 | 2.31 | **20/20** | 0.066 | 1.26 |
| NarrativeEngine | 10/20 | 0.193 | 2.04 | **19/20** | 0.061 | 1.23 |
| HEAP | 0/20 | 0.443 | 5.76 | **0/20** | 0.169 | 1.70 |
| DECOUPLED | 0/20 | 0.173 | 1.87 | **0/20** | 0.157 | 1.72 |
| DEAD | 0/20 | 0.000 | 1.00 | **0/20** | 0.000 | 1.00 |
| NOISE | 0/20 | 0.159 | 1.80 | **0/20** | 0.172 | 1.73 |
| CLONE | 0/20 | 0.054 | 1.20 | **0/20** | 0.054 | 1.20 |
| SCRAMBLE | 0/20 | 0.813 | 38.43 | **0/20** | 0.637 | 11.19 |
| LINEAR | 0/20 | 0.194 | 1.87 | **0/20** | 0.121 | 1.48 |

**All six negative controls stay out**, at 0/20 before and after. Φ's
checkpoint-to-checkpoint spread on the real engines falls by 3.5× and the verdict
stops depending on the grid.

This is a measurement, not a recommendation to land. Changing
`_minimum_partition` changes every Φ number the repo has recorded, and that is a
decision about the gate, not a fix to PairField.

## 8. What would refute this

Three things were named in advance as able to break the account, and each was
run.

**"The 0.845 is a real dynamical event that the jitter probe is too small to
see."** Refuted by §1: the trace at 25-step resolution shows Φ(A) hitting 5.37 at
step 600 and 5.80 at step 900 with separation, norms and coupling flat through
the whole window. There is no event to see.

**"Narrative has a genuinely different, stable shape."** Refuted by §5: Narrative
is 2/5 at 32c and 4/5 at 256c, with the same jump sizes and the same jitter cv.

**"A lower strength genuinely stabilises the pair."** Refuted by §6: the pass
count is not ordered by strength and the uncoupled control beats the default.

The last of those deserves its own table, because "0.020 scores 5/5" is exactly
the kind of result this session has been burned by. Strength × checkpoint grid,
five seeds each, 25 verdicts per strength:

| strength | −2 | −1 | +0 | +1 | +2 | total | A–G sep | peak \|h\| | Φ mean |
|---|---|---|---|---|---|---|---|---|
| 0.000 (uncoupled) | 3/5 | 3/5 | 4/5 | 3/5 | 4/5 | 17/25 | 84.50 | 6.31 | 1.767 |
| 0.005 | 4/5 | 1/5 | 3/5 | 4/5 | 4/5 | 16/25 | 85.83 | 6.13 | 1.826 |
| 0.010 | 3/5 | 4/5 | 3/5 | 3/5 | 3/5 | 16/25 | 87.78 | 5.98 | 1.750 |
| 0.020 | 4/5 | 4/5 | **5/5** | **5/5** | 4/5 | 22/25 | 89.72 | 6.89 | 1.864 |
| **0.030 (default)** | 4/5 | **5/5** | 4/5 | 4/5 | 3/5 | 20/25 | 95.30 | 6.49 | 1.567 |
| 0.050 | 4/5 | 4/5 | 3/5 | 3/5 | 4/5 | 18/25 | 103.42 | 6.55 | 1.305 |

`strength = 0.020` reaches 5/5 on two grids out of five and 4/5 on the other
three. The default reaches 5/5 on the grid at −1. Neither is a stable all-seed
pass; both are the same coin landing differently. Across all 25 verdicts the
default's 20 against 0.020's 22 is a difference of two, which five seeds cannot
resolve.

**So the answer to "does a lower strength pass all five seeds" is no.** No
strength in 0.000–0.050 passes all five seeds robustly, and the apparent winner
changes with where the checkpoints fall. Every strength stayed bounded (peak
\|h\| 5.98–6.89, against the 1e4 guard) and coupled (separation 84.5–103.4), so
none of these numbers came from a diverged or uncoupled pair — they are honest
readings of a quantity that does not carry the signal.



## 9. Verdict

PairFieldEngine is not the cause of its own PERSISTENCE failure. Through the
window where the gate reports a collapse and a partial recovery, the pair is
bounded (peak \|h\| 4.86 against a 1e4 guard), coupled (separation 244–257,
never near zero), and stationary in every quantity that describes it. What moves
is the Fiedler sign-cut inside Φ, which overshoots the true minimum cut by 3–7×
and flips between two overshoots from step to step; the `recovers` rule then
compares one flip against the max of five.

The defect is in `PhiIIT._minimum_partition` (`bench_v2.py:225-237`) and in the
shape of `_verify_persistence`'s pass rule (`bench_v2.py:1794-1796`), and it
applies to every engine the gate scores. `pairfield_engine.py` is unchanged, and
should stay unchanged, on this evidence.

---

Reproduce. The measurement scripts are session scratch, under
`$SP = /private/tmp/claude-501/-Users-mini-dancinlab-anima-lab-1/d0396916-1383-4a82-b215-02ece85f6789/scratchpad`;
none of them modify anything in the repo.

```bash
.venv/bin/python $SP/trace.py 256 64 128 44 25 --noise   # §1, §2
.venv/bin/python $SP/sweepcut.py                         # §3
.venv/bin/python $SP/grid_shift.py PairField 32 64 128   # §4
.venv/bin/python $SP/traj.py NarrativeEngine 32 64 128   # §5
.venv/bin/python $SP/sweep_strength.py 32 64 128 0.0 0.005 0.01 0.02 0.03 0.05  # §6
.venv/bin/python $SP/sweep_grid.py 32 64 128 0.0 0.005 0.01 0.02 0.03 0.05      # §8
.venv/bin/python $SP/corrected.py 32 64 128 --shifted    # §7
```
