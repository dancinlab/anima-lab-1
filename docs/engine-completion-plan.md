# If I finished this engine

Not a wish list. Every branch below starts from something this session measured,
and each names what would refute it.

## What the measurements have in common

| # | finding | measured |
|---|---|---|
| 1 | the gate passed a corpse | SCRAMBLE 5/7 = real engine 5/7 |
| 2 | Φ's direction is inverted | SPLIT 3.037 > RING 0.740 — non-integration scores 4.1× higher |
| 3 | "Φ ≈ cells" is the collapse signature | closed form Φ ≈ M·n/4, within 7% |
| 4 | the speech condition anti-correlates with its name | SCRAMBLE 6.8 vs real 1.0 |
| 5 | the Φ ratchet is inert | cosine 0.468 on vs 0.553 off; trajectories overlap |
| 6 | faction count changes nothing | 2/3/4/6/8/12/24/48 → consensus 1.0, and so does removing them |
| 7 | the output cannot see cell order | max\|Δ\| = 4.8e-7 under a row permutation |

Six of the seven are the same shape: **a channel that looks structural and
carries no information.** Factions are contiguous index slices, so permuting rows
moves a faction mean and nothing else. The ratchet fires 29–64 times and moves no
trajectory. The output is permutation-invariant. Φ sums total correlation and
subtracts the very min-cut that defines integration.

So the work is not to add. It is to cut every channel that carries nothing and
see what is left standing.

## Branch 1 — cut, then re-verify

Remove factions, the ratchet, and the fixed slices; re-run the five conditions.
If it still passes, those were ornament from the start, and that is the first
direct evidence for Law 22 ("adding features lowers Φ; adding structure raises
it") rather than another citation of it. If it fails, the failing condition names
which channel was load-bearing after all.

Refuted by: any removed channel whose absence changes a condition's verdict.

## Branch 2 — make the emitted output part of the system

Of seven conditions, one consumes the output and drops it, one feeds it back, five
never call it. Yet the output is the **only** channel separating the real engine
from SCRAMBLE (norm 31.08 vs 26.29, cosine-continuity 0.9814 vs 0.9371). The one
signal that discriminates is the one nothing reads. Make it order-bearing and let
the gate read it.

Refuted by: an order-bearing output on which SCRAMBLE still ties the real engine —
which is exactly what happened on the first attempt (SCRAMBLE scored 13.2), so
this needs the control run before the claim, not after.

## Branch 3 — put Φ back on the min-cut

`phi-rs/src/lib.rs:369` keeps `total − min_cut` when IIT's Φ *is* the min-cut.
Reversing it redefines Φ for 24 consumers and for every recorded number including
"Φ=1142". That is the cost. The benefit is that the recorded numbers would mean
something for the first time.

Refuted by: a construction where the min-cut reading ranks an obviously
disintegrated system above an integrated one.

## Branch 4 — the controls belong inside the gate (LANDED)

The six controls lived in a separate file the gate never called, so the gate
could drift back to passing a corpse silently. Now `--verify` runs them first,
and **any condition a control clears is VOID for that run** — no engine may bank
it, and deployment is blocked regardless of engine scores.

This is not a new rule. CLAUDE.md already states that baselines come from the
population's own null rather than from a constant. This applies that to the gate
itself.

Refuted by: a control passing a condition that *should* be passable by it — i.e.
a control that is not actually a corpse. That is a real risk for DECOUPLED, which
is a design choice about coupling rather than a dead engine.

## Branch 5 — structure in the weights, not the state

`bench_engine_redesign.py` measured: give each cell its own rotation inside the
map and the repulsion force becomes **unnecessary** — switching it on changes the
result by not one digit, because an already-differentiated population gives an
overlap-scaled force nothing to do. Repulsion was compensating in state space for
structure missing from the weights.

But per-cell rotation alone differentiates *worse* (cosine 0.6509 at zero
coupling) than the shared map *with* repulsion (0.3821). Neither substitutes for
the other. **Both together has not been measured.**

Refuted by: rotation + repulsion landing no better than repulsion alone.

## Branch 6 — ask again why a cell divides

Splitting fires when tension crosses a bar, and tension tracks input magnitude —
so a large input causes division. That is a response to a stimulus, not
differentiation. Already measured (`bench_specialization_split.py`): aggregating
tension per *stimulus* instead of per consecutive *step* is the only design that
settles between floor and ceiling (9–14 cells from a 2-cell start) and holds
within 1.24× when the ceiling moves 16→64.

Refuted by: real corpus stimuli producing the same runaway the step axis shows.
Not landed — three seeds on synthetic vectors refutes a design; it does not
justify changing how a running engine divides.
