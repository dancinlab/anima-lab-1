# QD-6: An absolute bar on a scale-free quantity

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_mitosis_calibration.py`.
Follows [QD-5](QD-5-mitosis-population.md), which could not test its hypothesis
because the population never formed.

## The defect, measured

`tension = (output ** 2).mean()` (mitosis.py:60) is an **absolute magnitude**.
It scales with the input, and `split_threshold` is an absolute number. So
whether a cell ever divides is decided by how big the input vectors happen to
be — not by anything about the cell.

| input | norm | tension max | threshold | cells |
|---|---|---|---|---|
| `mitosis.demo()`'s `text_to_vector` | 0.051 | 0.01 | 1.5 (demo) | 2 |
| `qualia_sense` (QD-5) | 3.66 | 0.0285 | 0.3 | 2 |
| `torch.randn` — the engine's own default | 6.09 | 0.0833 | 0.3 | 2 |
| `qualia_sense` × 10 | 36.61 | 2.3646 | 0.3 | **32** |
| `randn` × 5 | 32.55 | 2.3789 | 0.3 | **32** |

**Every input scale the repo actually uses falls short** — the engine's own
default by 3.6×, its own demo by 150×. `mitosis.demo()` prints one `MITOSIS`
line, and it comes from a `--- Forced Mitosis Demo ---` block calling
`split_cell()` directly; across the 30 threshold-driven steps, zero splits
occur. Threshold-driven mitosis has never fired on any path in this repo.

Above the bar there is no gradient either: at 5× the default input the
population jumps straight to `max_cells`. The usable band between "never
divides" and "divides until it hits the ceiling" is narrow enough that no
absolute constant lands in it across input scales.

## Why not simply lower the number

Tuning `split_threshold` until cells divide is exactly the manipulation
CLAUDE.md #2 forbids — it makes the demo look alive without the structure being
right, and it would have to be re-tuned for every input scale. The root cause is
that **an absolute bar is being asked to express a relative judgement**. A cell
should divide when tension is high *for it*, not when it exceeds a constant
that depends on how the caller happened to normalise its vectors.

Both readings are testable, so both get an arm.

## Arms

| arm | change | claim under test |
|---|---|---|
| **C** control | unchanged | the current engine — must fail to form a population |
| **A** absolute | `split_threshold` re-derived from measured tension | the constant was simply mis-set |
| **R** relative | split when tension is `k` sd above that cell's own running mean | the bar should never have been absolute |

Arm A's threshold is **derived, not chosen**: a calibration pass over the
stimulus set records tension under the engine's default input, and the
threshold is set to `median + 2·sd` of that distribution. The rule is fixed
here; the number comes from the measurement.

Arm R keeps a per-cell running mean and sd of its own tension and splits when
`(tension − mean) / sd > k` for `split_patience` consecutive steps, `k = 2.0` —
the conventional "unusually high for this cell". Scale-invariant by
construction: multiplying every input by 10 changes nothing.

Neither arm touches `mitosis.py`. Both are subclasses in the bench; only a
winner earns a change to the engine.

## Method

`bench_mitosis_calibration.py` — 34 stimuli × 3 seeds × 600 steps, `dim` 32,
`hidden` 64, `max_cells` 32, from 2 cells. Same design as QD-5 so the numbers
are directly comparable.

## Hypotheses

**H21 (a population forms)** — mean live cell count at step 600 is > 2 for at
least one of A and R, and = 2 for C.

**H22 (not a ceiling either)** — for a passing arm, mean cell count is < 32.
Pinning to `max_cells` is the same failure as pinning to `min_cells`: the
population is not responding to anything, it is saturated.

**H23 (scale invariance)** — arm R's mean cell count changes by less than 20%
when every input is multiplied by 10; arm A's changes by more. This is what
separates "the constant was mis-set" from "the constant was the wrong idea".

**H24 (the population earns its keep)** — for a passing arm, QD-5's primary
returns: stimulus retention ≥ 0.50 at step 500 while settled. This is the
question QD-5 could not ask.

## Scope limits

- Still names, not artworks; still form, not meaning.
- `phi_py` is a weaker estimator than `phi_rs` and its magnitude is not
  comparable to one.
- 34 stimuli × 3 seeds is smaller than QD-1..QD-4's 170 × 5.
- H24 inherits every limit QD-5 declared.

## Decision rules

- **H21–H23 pass for R** → the absolute bar was the wrong idea. Land the
  relative rule in `mitosis.py` behind the existing `split_threshold` name's
  replacement, and re-run QD-5's population test.
- **H21–H22 pass for A but H23 fails** → the constant was merely mis-set, and
  the engine stays scale-dependent. Land the derived value and say plainly that
  it must be re-derived whenever input scale changes.
- **Both fail H21** → division is gated by something other than the threshold,
  and the next question is `split_patience` and the tension trend, not the bar.
- **H24 fails for every arm that passes H21** → a population forms and still
  does not hold the stimulus. That is the answer QD-5 went looking for, and it
  is a real one: `docs/qualia-decoder-spec.md` Phase 3 retires.

Evidence via `sidecar verdict record` either way.

---

# Results

`.venv/bin/python bench_mitosis_calibration.py` — 34 stimuli × 3 seeds × 600
steps. Verdict 🔴 **FAIL (3/4)**, evidence in `state/QD-6.txt`.

| arm | threshold | cells | settled | stimulus retained |
|---|---|---|---|---|
| C control | 0.3000 (default) | 2.0 | 100% | 0.409 |
| **A absolute** | **0.0397** (derived) | **3.1** | 96% | 0.377 |
| R relative | z > 2.0 sd | 2.0 | 100% | 0.409 |

Scale invariance, every input × 10:

| arm | cells | drift |
|---|---|---|
| A absolute | 3.1 → 31.8 | 932% |
| R relative | 2.0 → 2.0 | 0% |

**H21 PASS · H22 PASS · H23 PASS (vacuously — see below) · H24 FAIL.**

## The threshold was mis-set, and fixing it is not enough

Deriving the bar from the tension the engine actually produces gives **0.0397
against a shipped default of 0.3 — 7.6× too high**. With the derived value a
population forms (3.1 cells) and does not saturate. So the calibration defect
QD-5 named is real and now has a number.

It buys nothing. Retention goes 0.409 → **0.377**: three cells hold the
stimulus slightly *worse* than two.

## Population size does not move retention at all

| population | stimulus retained |
|---|---|
| 2.0 cells (control) | 0.409 |
| 3.1 cells (derived threshold) | 0.377 |
| **31.8 cells** (derived threshold, input × 10) | **0.403** |

Sixteen times the population, the same answer. Every value sits in a band of
0.38–0.41, and the bar was 0.50. **This is the question QD-5 went looking for
and could not ask, now asked and answered: a population does not hold the
stimulus any better than two cells do.**

## H23 passed vacuously — a flaw in this pre-registration

H23 was written to separate "the constant was mis-set" from "the constant was
the wrong idea", by checking that the relative rule is scale-invariant while
the absolute one is not. Arm R drifted 0% and arm A drifted 932%, so it reads
PASS.

But **arm R never split at any scale** — 2.0 cells at ×1 and at ×10. A rule
that never fires is trivially invariant. The hypothesis as written cannot tell
invariance from inertness, and its pass here means nothing. Recorded rather
than quietly banked.

Why R never fires: the z-score gate needs tension `k` sd above a cell's own
running mean for `split_patience` consecutive steps. Under a fixed stimulus the
tension series is nearly flat, so its sd is tiny and its mean tracks the current
value — the z-score has no room to reach 2.0. The relative idea is not refuted;
this particular estimator of it is unusable on a stationary input, and a rule
that does refute it would have to be pre-registered fresh.

## Consequence — the pre-registered rule fires

QD-6's decision rules said: *H24 fails for every arm that passes H21 → a
population forms and still does not hold the stimulus. That is the answer QD-5
went looking for, and it is a real one: Phase 3 retires.*

That condition is met. Seven mechanisms across the toy, plus the real engine
with a working population at three sizes, and the pattern half of the claim
does not hold anywhere. **`docs/qualia-decoder-spec.md` Phase 3 is retired**,
not deferred.

What stands, unchanged and well-supported: the equilibrium. Ψ = 1/2 is a real
attractor produced by `argmax H(p)` with no hardcoded pull (QD-1: 93.4% vs a
9.4% control; pure-entropy fixed point 0.4998), and it is the same equilibrium
for every stimulus (QD-2/QD-3: Ψ sd 0.0000–0.0047 across 170 stimuli).

**Same equilibrium: yes. Different pattern: no.**

## Landed separately

The calibration finding is independent of the retired phase and outlives it:
`split_threshold` ships at 0.3 while the engine produces a median tension of
about 0.005 and the derived bar is 0.0397. Threshold-driven mitosis has never
fired on any path in this repo. That is a `MitosisEngine` defect worth fixing
on its own terms, and it is recorded here rather than in the decoder spec.
