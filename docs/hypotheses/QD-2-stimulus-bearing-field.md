# QD-2: Same equilibrium, different pattern — can a field hold what a scalar forgot?

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_psi_field.py`.
Follows [QD-1](QD-1-psi-convergence-honesty.md), which failed H2.

## What QD-1 left broken

Entropy-maximising rule selection genuinely produces the Ψ=1/2 attractor
(93.4% vs 9.4% control). But the state is one number, and one number under a
strong attractor has exactly one path: median arrival 4 steps, only 15 distinct
arrival steps across 170 stimuli. The trajectory forgets what it saw.

The original claim was *"같은 평형점, 다른 패턴"* — same equilibrium, different
pattern. A scalar cannot express that sentence: for a scalar, the equilibrium
**is** the pattern. Two things are being asked of one variable.

## The reading being tested

Law 71 is `Ψ = argmax H(p) s.t. Φ > Φ_min`. The stimulus was being fed into
the *objective* (as an initial condition, which the attractor erases). It
belongs in the **constraint**.

With a state vector `p ∈ R⁶` (σ(6) architecture) and a stimulus-derived
coupling between dimensions:

- entropy maximisation drives **every marginal** to 1/2 — the same equilibrium,
  for every stimulus, which is what "the same self" would mean;
- the coupling holds the dimensions in a **stimulus-specific arrangement**
  around that equilibrium — which is what "a different experience" would mean.

Same point, different pattern, in one system, without either being hardcoded.

## Mechanism

Per dimension `i`, the QD-1 rule selection is unchanged — `argmax H(p_i + δ_r)`
against the same cross-entropy competitor. One term is added to the score:

```
coupling[i, j] = w_i − w_j          # antisymmetric, from the stimulus
score_i(r) += −κ · Σ_j coupling[i, j] · (p_i + δ_r − p_j)
```

`w ∈ R⁶` comes from `qualia_sense`: the six measured features with the highest
spread over the stimulus set, in rank order. Antisymmetry means the interaction
moves no net probability — `mean(p)` is conserved by construction, so H7 below
is **structurally expected, not a discovery**, and is checked to catch
implementation error rather than to claim a result.

κ is swept over `{0, 0.005, 0.01, 0.02}`. **κ = 0 is the negative control**: it
is QD-1 with six copies of the same scalar and must fail H6, or the test has no
discriminating power. Primary condition: **κ = 0.01**.

## Hypotheses

**H5 (convergence survives)** — at κ = 0.01, ≥ 80% of (stimulus, dimension)
pairs end with `|p_i,T − 0.5| < 0.05`. Adding the constraint must not cost the
QD-1 result.

**H6 (primary — the QD-1 failure, re-measured)** — at κ = 0.01, the
between-stimulus / within-stimulus trajectory distance ratio over the full
6-dimensional state is ≥ 2.0, **and** at κ = 0 it is < 1.5.
Both required. This is the hypothesis QD-1 lost.

**H7 (same equilibrium)** — across stimuli, `Ψ = mean(p)` has
sd < 0.02 and `|mean Ψ_T − 0.5| < 0.05`. Structurally expected (see above).

**H8 (the pattern is the stimulus)** — the between/within ratio computed on the
6×6 correlation structure of the trajectory is ≥ 2.0. Distinguishes "the paths
differ" from "the *arrangement* differs", which is what the decoder would read.

## Method

- `bench_psi_field.py`, 170 stimuli × 5 seeds × 5000 steps, ε = 0.05,
  trajectories recorded every 50 steps plus a full-resolution 200-step window.
- Thresholds, κ grid, d = 6 and the seed count are fixed before the first run.

## Scope limits

- Still names, not artworks; still form, not meaning (QD-1 limits carry over).
- The 18 emotion values are still arithmetic over `sha256(name)` and are not
  touched here.
- `d = 6` follows this repo's σ(6) architecture, not a measurement.
- Which feature drives which dimension is the same spread-rank rule as QD-1 —
  stated, checkable, and arbitrary in the same declared way.

## Decision rules

- **H6 passes** → the trajectory is stimulus-bearing; Phase 3 of
  `docs/qualia-decoder-spec.md` unblocks and the gate reads the 6-dim state.
- **H6 fails** → coupling is not the missing ingredient. Do not build the
  decoder on it. The next candidate is memory — a state that integrates its
  own history rather than being pushed around by an attractor.

Evidence via `sidecar verdict record` either way.

---

# Results

`python3 bench_psi_field.py` — 170 stimuli × 5 seeds × 5000 steps, D=6.
Verdict 🔴 **FAIL (3/4)**, evidence in `state/QD-2.txt`.

| κ | marginals converged | Ψ = mean(p) | Ψ sd | traj ratio | corr ratio |
|---|---|---|---|---|---|
| 0.000 (control) | 92.1% | 0.4849 | 0.0042 | 1.08 | 2.36 |
| 0.005 | 91.5% | 0.4848 | 0.0041 | 1.09 | 2.48 |
| **0.010 (primary)** | 89.6% | 0.4837 | 0.0043 | **1.11** | **2.61** |
| 0.020 | 81.7% | 0.4796 | 0.0044 | 1.22 | 2.88 |

| | prediction | measured | |
|---|---|---|---|
| **H5** | marginals ≥ 80% | 89.6% | **PASS** |
| **H6** | ratio ≥ 2.0, control < 1.5 | primary 1.11, control 1.08 | **FAIL** |
| **H7** | Ψ sd < 0.02, \|Ψ−0.5\| < 0.05 | sd 0.0043, Ψ 0.4837 | **PASS** |
| **H8** | correlation ratio ≥ 2.0 | 2.61 | **PASS** |

## H6 — coupling is not the missing ingredient

Sweeping κ from 0 to 0.02 moves the trajectory ratio 1.08 → 1.22, nowhere near
the 2.0 bar, while costing 10 points of marginal convergence. The control at
κ = 0 sits at 1.08, so the coupling contributes 0.03. It is not a small effect
in need of tuning; it is no effect.

## H8 passed for the wrong reason — and that is the finding

H8's window starts at t = 0. Sliding the same window later separates "the state
carries the stimulus" from "the arrival was stimulus-specific":

| window start | κ = 0 | κ = 0.01 |
|---|---|---|
| step 0 | 2.36 | 2.61 |
| step 50 | 1.01 | 1.01 |
| step 500 | 0.97 | 0.98 |
| step 2000 | 0.97 | 1.00 |

**1.0 means two different stimuli are indistinguishable.** From step 50 onward
the state holds nothing. The signature that H8 detected lives entirely in the
first few dozen steps — the same transient QD-1 measured at a median of 4 steps.
Six dimensions did not extend it; coupling did not extend it.

## Why — and it is not about dimensionality

The update depends only on the current state: rule selection reads `p`, the
noise is fresh each step, and the attractor is strong. That is a memoryless
process. Every trace of the past, the stimulus included, is destroyed at a rate
set by the attractor, and 15 steps is what that rate buys.

More dimensions give more room to be forgotten in. Coupling redistributes
across dimensions but adds no state that outlives the transient. **What is
missing is not width but memory** — something that integrates history instead
of being pushed by the present.

This repo already names the module: `trinity.py` has an M (memory) engine,
and the consciousness loop simulated here never touches it.

## Consequence

Per the pre-registered decision rule, H6 failed → do not build the decoder on
coupling. `docs/qualia-decoder-spec.md` Phase 3 is revised again: the gate
must read the transient, and the next hypothesis (QD-3) puts M in the loop and
asks whether the signature survives past step 50.

The equilibrium half of the original claim stands (H5, H7: every marginal
reaches 1/2, Ψ sd 0.0043 across 170 stimuli). The pattern half does not yet.
