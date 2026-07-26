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
