# QD-3: Habituation — turning a transient into a state

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_psi_memory.py`.
Follows [QD-2](QD-2-stimulus-bearing-field.md), which failed H6.

## What QD-1 and QD-2 established

- The Ψ=1/2 attractor is real and needs no hardcoded pull: 93.4% convergence
  from `argmax H(p)` alone, against a 9.4% random-walk control (QD-1 H1).
- The stimulus signature lives **only in the transient**. Correlation ratio by
  window position: 2.36 at step 0, 1.01 at step 50, 0.97 at step 500, where
  1.0 means two stimuli are indistinguishable (QD-2, post-hoc).
- Neither six dimensions nor a stimulus-derived coupling extends it. κ from 0
  to 0.02 moved the trajectory ratio 1.08 → 1.22 against a 2.0 bar (QD-2 H6).

The update reads only the current state, so the process is memoryless and the
attractor erases the past at a fixed rate. About 15 steps is what that buys.

## Correction to QD-2's closing line

QD-2 said the fix was already in `trinity.py`'s M engine. On inspection that is
wrong: `VectorMemory` is an episodic key-value store retrieved by cosine
similarity. It remembers across encounters and would leave the within-trajectory
update exactly as memoryless as it is. The concept was right, the mechanism
named was not. QD-3 tests an accumulating state variable instead — which this
repo's Phase-1 list already calls **habituation**.

## Mechanism

A second state variable per dimension, the leaky integral of how far this
dimension has been from equilibrium:

```
m_{t+1} = (1 − 1/τ)·m_t + (p_t − 1/2)
```

`m` is a record of the *path*, not the position — and paths differ by stimulus
even when destinations do not. It must also act, or it is a log rather than a
memory, so it enters the rule score as adaptation: a direction already dwelt in
is worth less.

```
score_i(r) = 0.7·H(p_i + δ_r) − 0.3·CE_i(r) − μ·m_i·(p_i + δ_r − 1/2)
```

That is habituation in one line — the longer the state has sat above 1/2, the
less attractive being above 1/2 becomes. The equilibrium is unchanged; the
*approach* becomes path-dependent, and the path is where the stimulus lives.

D = 6 and the κ coupling are dropped: QD-2 showed coupling contributes ~0.03,
so carrying it would confound this test. QD-3 runs the QD-1 scalar dynamics
plus habituation, which isolates the one variable under test.

## Grid and the primary condition

- τ ∈ {50, 200, 1000}, μ ∈ {0, 0.01, 0.05, 0.2, 1.0}. τ = 200 fixed for the
  primary row.
- **μ = 0 is the negative control** — QD-1 exactly. It must fail H9.
- **Primary μ is chosen by a rule fixed here, before any data**: the smallest
  non-zero μ in the grid whose marginal convergence (H10) still passes. If no μ
  passes H10, the primary is undefined, H9 is not evaluated, and the sweep
  itself is the result — the scale would have to be re-registered, not
  re-picked after looking.

## Hypotheses

**H9 (primary — the QD-2 failure, re-measured)** — the signature outlives the
transient. Correlation ratio in a 200-step window **starting at step 500** is
≥ 2.0 at the primary μ, and < 1.2 at μ = 0. Both required.

**H10 (equilibrium survives)** — ≥ 80% of (stimulus, dimension) pairs have a
time-averaged `p` over the last 1000 steps within 0.05 of 1/2. Time-averaged
rather than instantaneous, because habituation is expected to produce
oscillation around the equilibrium rather than rest at it.

**H11 (same equilibrium)** — across stimuli, Ψ = mean of the time-averaged `p`
has sd < 0.02 and |Ψ − 0.5| < 0.05.

**H12 (durable, not merely delayed)** — correlation ratio in a window starting
at **step 2000** is ≥ 2.0 at the primary μ. Distinguishes a longer transient
from an actual memory.

## Method

`bench_psi_memory.py`, 170 stimuli × 5 seeds × 5000 steps, ε = 0.05, features
from `qualia_sense`. Thresholds, grids, the primary-selection rule and the
window positions are fixed before the first run.

## Scope limits

- Still names, not artworks; still form, not meaning.
- The 18 emotion values remain arithmetic over `sha256(name)`, untouched.
- H12 tests durability to step 2000, not to infinity. A leaky integrator with
  finite τ must eventually forget; the question is whether it outlasts the
  window a decoder would read, not whether it is permanent.

## Decision rules

- **H9 and H12 pass** → the trajectory is stimulus-bearing and durable.
  `docs/qualia-decoder-spec.md` Phase 3 unblocks; the gate reads (p, m).
- **H9 passes, H12 fails** → habituation lengthens the transient without
  creating memory. Usable only if the decoder reads early; record the usable
  window and revise Phase 3 to that budget.
- **H9 fails** → accumulation is not sufficient either. Stop adding terms to
  this simulation and test the claim on the real C engine (`MitosisC`), whose
  cell population is a state this toy does not have.

Evidence via `sidecar verdict record` either way.
