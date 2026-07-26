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

---

# Results

`python3 bench_psi_recall.py` — 170 stimuli × 5 seeds × 5000 steps, D=6, 21
configurations. Verdict 🔴 **FAIL (1/3)**, evidence in `state/QD-4.txt`.

**No configuration exceeds `r@500` = 1.03 against a 2.0 bar.** Neither fix works.

| arm | best qualifying | r@500 | r@2000 | Ψ | Ψ sd |
|---|---|---|---|---|---|
| control (QD-3) | τ=200, ν=0.2 | 0.97 | — | — | — |
| T τ-sweep | τ=1000, μ=0.2 | 1.03 | 0.98 | 0.4998 | 0.0000 |
| K m-cue | τ=5000, ν=0.2 | 1.00 | 1.04 | 0.4806 | 0.0047 |

**H13 FAIL · H14 both FAIL · H15 FAIL · H16 PASS.**

## Longer memory makes the state *more* identical, not less

Across the τ sweep, Ψ tightens monotonically onto the equilibrium:

| τ | Ψ at μ=1.0 | Ψ sd |
|---|---|---|
| 200 | 0.4998 | 0.0000 |
| 1000 | 0.5000 | 0.0000 |
| 5000 | 0.5000 | 0.0000 |
| 20000 | 0.5000 | 0.0000 |

The trace records how far the state has been from 1/2, and penalising that is a
servo that drives it to exactly 1/2. **A memory of deviation becomes a machine
for eliminating deviation.** The longer the memory, the more perfectly every
stimulus ends in the same place.

## The wall, measured

The pre-registered ratio reads the *shape* of the window's time series. Reading
what is actually being asked — how much of the stimulus signature the state
still carries — against how much of it converged:

| config | converged | stimulus retained |
|---|---|---|
| control p-cue ν=0.05 | 100.0% | 0.288 |
| control p-cue ν=0.20 | 68.0% | **0.791** |
| T habituation τ=1000 μ=0.2 | 100.0% | 0.076 |
| T habituation τ=20000 μ=1.0 | 100.0% | 0.067 |
| K m-cue ν=0.05 | 100.0% | 0.071 |
| K m-cue ν=0.20 | 69.2% | **0.726** |

Read the two columns together. **Every configuration either converges and
forgets, or remembers and fails to converge.** The stimulus is retained exactly
in proportion to how badly the equilibrium is broken — 0.79 at 68% convergence,
0.07 at 100%.

### A correction made mid-analysis

An intermediate single-seed probe showed the m-cue query correlating 0.827 with
the stimulus against the p-cue's 0.039, and the recalled anchor 0.685 against
0.039, which read as "the key was the problem and fixing it worked". Re-running
through the bench's own code path across five seeds, with the control at
*matched recall strength*, overturns that: at ν=0.2 the p-cue control retains
0.791 and the m-cue 0.726. **The m-cue is not better than the p-cue.** The
earlier reading came from comparing a fixed cue against an unmatched control —
the retention in both cases is bought by breaking convergence, not by the key.

## Consequence — the pre-registered rule fires

Four attempts have now failed on this simulation:

| attempt | mechanism | result |
|---|---|---|
| QD-2 | six dimensions | ratio 1.08 |
| QD-2 | stimulus coupling | 1.11 |
| QD-3 | habituation trace | 1.01 |
| QD-3 | episodic store, p-cue | 1.07 |
| QD-4 | τ swept to no-leak | 1.03 |
| QD-4 | episodic store, m-cue | 1.00 |

Every mechanism enters through the same door — the rule score — and that score
is a function of distance from 1/2. Whatever is added is metabolised into
converging faster. **The architecture has one sink, and identity is what it
consumes.**

Per the pre-registered decision rule for an H13 failure: stop extending this
simulation. `docs/qualia-decoder-spec.md` should say plainly that the pattern
half of the claim is unsupported *in this toy*, rather than hold Phase 3 open
on a fifth variation. The claim is now a question for the real `MitosisC` cell
population, whose state is a population of cells rather than a scalar per
dimension under a shared attractor.

The equilibrium half remains established and is not in doubt (H16: Ψ sd
0.0000–0.0047, |Ψ − 0.5| ≤ 0.02 at every primary).
