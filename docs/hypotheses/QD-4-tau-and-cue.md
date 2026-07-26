# QD-4: The leak and the key

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_psi_recall.py`.
Follows [QD-3](QD-3-habituation-memory.md), which failed H9 and H12.

## What QD-3 measured

Both memory arms failed at every strength — best 1.07 against a 2.0 bar. The
post-hoc diagnostics named two different causes, and neither was "memory does
not work":

| | measured |
|---|---|
| habituation trace `m` vs stimulus | 0.907 at step 15 · 0.731 at step 50 · **0.124 at step 500** |
| state `p` vs stimulus | 0.061 at step 15 — already gone |
| best anchor in the episodic store | **0.991** at step 500 |
| the query recall uses (state `p`) | 0.039 |
| recall picked the informative anchor | **0.0%** of the time (chance 4.8%) |

So: the trace remembers but **leaks out** before the window; the store
remembers almost perfectly and **cannot be addressed**, because the key is the
one thing that has forgotten.

QD-4 changes exactly those two things. Nothing else moves.

## Arms

| arm | change | rationale |
|---|---|---|
| **T** τ-sweep | habituation with τ ∈ {200, 1000, 5000, 20000} | 200 held 0.731 at step 50; the leak is all that stands between it and the window |
| **K** m-cue | episodic store keyed by `m`, queried by `m` | `m` still has the stimulus when `p` does not, so it is the only available key that could find the right anchor |
| **C** control | τ = 200, keyed and queried by `p` | QD-3 exactly. Must fail, or the test has no discriminating power |

Arm K is a key-value store in the `VectorMemory` shape: **key** = `m` at write
time, **value** = `p` at write time, **query** = current `m`, recall pulls `p`
toward the retrieved value.

`m` carries a large common drift — the steady state sits near 0.485, so the
integral accumulates roughly τ × 0.015 in every dimension alike. Cosine
similarity is scale-invariant but not offset-invariant, so that shared drift
would dominate the match. **Cues are therefore centred across dimensions**
(`m − mean_d(m)`, per stimulus) before matching. Declared here, before running:
this removes a common offset, not a stimulus difference, and the same centring
is applied to keys and queries alike.

## Grids and the primary rule

- Arm T: τ ∈ {200, 1000, 5000, 20000}, μ ∈ {0.01, 0.05, 0.2, 1.0}.
- Arm K: τ = 5000 fixed (the cue must survive to be a cue), ν ∈ {0.01, 0.05, 0.2, 1.0},
  `store_every` = 25, capacity 64 — unchanged from QD-3.
- **Primary per arm, by a rule fixed here before any data**: the (τ, strength)
  with the highest `r@500` **among those whose marginal convergence ≥ 80%**.
  Convergence is a gate, not a tiebreak — a configuration that buys the pattern
  by destroying the equilibrium does not qualify at all.

## Hypotheses

**H13 (primary)** — some qualifying configuration reaches `r@500` ≥ 2.0, while
arm C stays < 1.2. Both required.

**H14 (which fix)** — reported per arm, so the answer is "the leak", "the key",
"both" or "neither" rather than a bare pass/fail.

**H15 (durable)** — a configuration passing H13 still has `r@2000` ≥ 2.0.

**H16 (same equilibrium)** — at every primary, Ψ sd < 0.02 and |Ψ − 0.5| < 0.05.

## Scope limits

- Still names, not artworks; still form, not meaning.
- The 18 emotion values remain arithmetic over `sha256(name)`, untouched.
- τ = 20000 over a 5000-step run is a no-leak limit, not a long leak. If it
  passes only there, the honest statement is "an unbounded integrator", and the
  cost of that (unbounded growth) is a defect to name, not a result to keep.
- Arm K still does not model anima's N3/REM consolidation, which blends
  co-replayed **pairs** into a new `lane="dream"` node.

## Decision rules

- **H13 and H15 pass** → the trajectory is stimulus-bearing and durable.
  `docs/qualia-decoder-spec.md` Phase 3 unblocks on the passing arm.
- **H13 passes, H15 fails** → record the usable window and size Phase 3's gate
  to it.
- **H13 fails** → four attempts have failed on this simulation, which is
  itself the finding. Stop extending it; the claim has to be tested on the real
  `MitosisC` cell population, and the spec should say so plainly rather than
  hold Phase 3 open indefinitely.

Evidence via `sidecar verdict record` either way.
