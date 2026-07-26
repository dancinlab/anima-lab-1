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

The κ coupling is dropped — QD-2 measured its contribution at ~0.03, so
carrying it would only confound this test.

## Three arms — both memory kinds, against the current form

Amended at owner request, before any data was run: test **both** memory
devices rather than picking one, with the present dynamics as the control.

| arm | memory | mechanism |
|---|---|---|
| **N** none | — | QD-1/QD-2 dynamics unchanged — the current form |
| **H** habituation | within-trajectory | leaky integral `m` of the deviation, acting on rule selection |
| **E** episodic | across-trajectory | store states periodically, retrieve the nearest, let recall bias selection |

Arm E is the `.kosmos` anchor / `VectorMemory` shape, reduced to what this
numpy toy can carry: a bounded store written every `store_every` steps, top-1
cosine retrieval against the current state, and a recall term
`−ν·|p_i + δ_r − a_i|` that pulls the state toward what it remembers being.

In `anima` an anchor is written on emission and, during N3/REM sleep ticks,
by blending **co-replayed pairs** into a new `lane="dream"` node
(`core/dream_persist.py::dp_persist_sleep_replay`; prior dream nodes are
excluded from the pool, which is what stops runaway). That consolidation step
is **not** modelled here — arm E tests storage and recall only, and the
pair-blending is left for a later card. Stated so the result is not read as a
verdict on anima's dream consolidation.

The honest risk in arm E is that it is nearly circular: storing a
stimulus-specific state and pulling toward it will preserve stimulus identity
almost by construction. That is what makes H10/H11 the real test for this arm —
whether identity is bought at the cost of the equilibrium.

> **Amended before any data was run.** The first draft of this section said
> "D = 6 and the κ coupling are dropped … QD-3 runs the QD-1 scalar dynamics".
> That is incoherent: H9 and H12 are correlation ratios across dimensions and
> are undefined for a scalar. D = 6 is kept (uncoupled, κ = 0); only the
> coupling is dropped. Amended at pre-registration time, before the first run —
> the original wording is in commit `ae375df`.

## Grid and the primary condition

- Arm H: τ = 200 fixed, μ ∈ {0.01, 0.05, 0.2, 1.0}.
- Arm E: `store_every` = 25, capacity 64, ν ∈ {0.01, 0.05, 0.2, 1.0}.
- **Arm N is the negative control** — QD-1 exactly (μ = ν = 0). It must fail H9,
  or the test has no discriminating power.
- **The primary strength for each arm is chosen by a rule fixed here, before
  any data**: the smallest value in that arm's grid whose marginal convergence
  (H10) still passes. If no value passes H10 for an arm, that arm's primary is
  undefined, H9/H12 are not evaluated for it, and the sweep is the result —
  the scale gets re-registered, never re-picked after looking.

## Hypotheses

**H9 (primary — the QD-2 failure, re-measured)** — the signature outlives the
transient. Correlation ratio in a 200-step window **starting at step 500** is
≥ 2.0 for at least one memory arm at its primary strength, and < 1.2 for
arm N. Both required. Reported per arm, so "which memory works" is answered
rather than "does memory work".

**H10 (equilibrium survives)** — ≥ 80% of (stimulus, dimension) pairs have a
time-averaged `p` over the last 1000 steps within 0.05 of 1/2, per arm. Time-averaged
rather than instantaneous, because habituation is expected to produce
oscillation around the equilibrium rather than rest at it.

**H11 (same equilibrium)** — across stimuli, Ψ = mean of the time-averaged `p`
has sd < 0.02 and |Ψ − 0.5| < 0.05.

**H12 (durable, not merely delayed)** — correlation ratio in a window starting
at **step 2000** is ≥ 2.0 for any arm that passed H9. Distinguishes a longer transient
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

- **H9 and H12 pass for an arm** → that arm's trajectory is stimulus-bearing
  and durable. `docs/qualia-decoder-spec.md` Phase 3 unblocks on that arm; the
  gate reads its state. If both arms pass, the simpler one wins and the other
  is recorded as a second route.
- **H9 passes, H12 fails** → habituation lengthens the transient without
  creating memory. Usable only if the decoder reads early; record the usable
  window and revise Phase 3 to that budget.
- **H9 fails** → accumulation is not sufficient either. Stop adding terms to
  this simulation and test the claim on the real C engine (`MitosisC`), whose
  cell population is a state this toy does not have.

Evidence via `sidecar verdict record` either way.

---

# Results

`python3 bench_psi_memory.py` — 170 stimuli × 5 seeds × 5000 steps, D=6.
Verdict 🔴 **FAIL (2/4)**, evidence in `state/QD-3.txt`.

| arm | strength | marginals | Ψ | Ψ sd | r@500 | r@2000 |
|---|---|---|---|---|---|---|
| none (control) | — | 100.0% | 0.4851 | 0.0003 | 0.97 | 0.97 |
| habituation | 0.01 | 100.0% | 0.4921 | 0.0002 | 0.99 | 0.98 |
| habituation | 0.05 | 100.0% | 0.4970 | 0.0001 | 1.01 | 0.97 |
| habituation | 0.20 | 100.0% | 0.4991 | 0.0000 | 1.01 | 0.97 |
| habituation | 1.00 | 100.0% | 0.4998 | 0.0000 | 1.01 | 1.00 |
| episodic | 0.01 | 100.0% | 0.4851 | 0.0003 | 0.97 | 0.99 |
| episodic | 0.05 | 100.0% | 0.4841 | 0.0004 | 0.95 | 0.98 |
| episodic | 0.20 | 77.6% | 0.4676 | 0.0079 | 1.07 | 1.00 |
| episodic | 1.00 | 61.1% | 0.4557 | 0.0323 | 1.04 | 1.03 |

**H9 FAIL · H10 PASS · H11 PASS · H12 FAIL.** No arm reaches the 2.0 bar at any
strength in its grid. Stronger recall (ν ≥ 0.2) buys nothing (1.07) while
costing convergence (77.6%, then 61.1%). Stronger habituation makes Ψ *tighter*
— 0.4998 with sd 0.0000 at μ = 1.0 — which is the opposite of holding a pattern.

## Where the stimulus actually dies

Correlation between the stimulus signature and the state, post-hoc:

| step | state `p` | habituation `m` |
|---|---|---|
| 1 | 0.971 | 0.998 |
| 5 | 0.612 | 0.979 |
| 15 | **0.061** | 0.907 |
| 50 | 0.054 | 0.731 |
| 500 | 0.039 | **0.124** |
| 2000 | 0.057 | 0.067 |

**Habituation is a working memory.** At step 15 the state is at 0.061 — gone —
while `m` still holds 0.907. It is not that accumulation fails to remember; it
is that with τ = 200 the trace is down to 0.124 by step 500, which is where the
pre-registered window sits. Wrong time constant, right mechanism. That cannot
be rescued by re-picking τ after the fact; it is QD-4's question.

## The episodic arm fails for a different and sharper reason

At step 500:

| | correlation with the stimulus |
|---|---|
| best anchor in the store | **0.991** |
| the query (current state) | 0.039 |
| the anchor recall returned | **0.039** |
| picked the best anchor | **0.0%** of the time (chance 4.8%) |

The store holds the stimulus almost perfectly and recall never reaches it —
below chance, not merely at it. Retrieval is cued by the current state, and the
current state is the thing that has forgotten. A query made of noise matches
the many converged, noise-like anchors and systematically avoids the one
informative one.

**Storing is not the problem. Cueing is.** A memory addressed by the present
cannot return what only the past contained.

### Bearing on `anima`

anima's `.kosmos` anchors enter the engine through one door
(`kosmos_io → brain_decide`, `a_core_engine_map`). If that cue is derived from
a converged substrate state, this result says the anchors carrying identity are
exactly the ones retrieval will miss. Stated as a hypothesis about anima's
design, not a measurement of it: this bench models storage and recall, not
`brain_decide`'s actual cue, and not the N3/REM consolidation that blends
co-replayed pairs (`core/dream_persist.py`).

## Consequence

The pre-registered rule for an H9 failure said to stop adding terms and move to
the real C engine. The diagnostics point somewhere more specific first, and
both follow-ons are named by the evidence rather than by taste:

- **τ is the only thing between habituation and a pass** — it holds 0.731 at
  step 50 and 0.124 at step 500. Sweep it.
- **Cue recall by `m`, not by `p`** — `m` still has the stimulus when `p` does
  not, so it is the query that could actually find the right anchor.

`docs/qualia-decoder-spec.md` Phase 3 stays blocked. The equilibrium half of
the claim remains established (H10, H11: 100% marginal convergence, Ψ sd
0.0002–0.0003); the pattern half is still unbuilt after three attempts.
