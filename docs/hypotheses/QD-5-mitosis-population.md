# QD-5: Does a population hold what a scalar could not?

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_psi_mitosis.py`.
Follows [QD-4](QD-4-tau-and-cue.md), where the toy simulation hit a wall.

## Why leave the toy

Six mechanisms were tried across QD-2..QD-4 and all six failed the same way:

| attempt | mechanism | ratio |
|---|---|---|
| QD-2 | six dimensions | 1.08 |
| QD-2 | stimulus coupling | 1.11 |
| QD-3 | habituation trace | 1.01 |
| QD-3 | episodic store, p-cue | 1.07 |
| QD-4 | τ swept to no-leak | 1.03 |
| QD-4 | episodic store, m-cue | 1.00 |

And the reason was measured: every configuration either converged and forgot
(0.07 stimulus retained at 100% convergence) or remembered and failed to
converge (0.79 at 68%). Each mechanism entered through the same rule score, a
function of distance from 1/2, and was metabolised into converging faster.

`MitosisC` is structurally different in the one way that matters: its state is
a **population of cells** that divide under tension and merge when they grow
too alike. Identity has somewhere to live that is not a position relative to an
attractor.

## Two defects found while wiring this up

Both measured, one already fixed:

1. **`MitosisC.measure_phi` returned a silent `0.0`** without the Rust build,
   because it overrode the `CEngine` fallback added in `8e855ac`. Fixed in
   `d0e3bc3`; Φ now reads 25.3 at step 1 where it read 0.0 before. The
   `phi_py` fallback is a weaker estimator than `phi_rs` and its magnitude is
   not comparable to one.
2. **`MitosisC.__init__` force-grows to `max_cells` by cloning `cells[0]`.**
   The clones are identical, the engine merges cells that are too similar by
   design, and the population collapses to `min_cells` = 2 within ten steps —
   measured 32 → 2 between step 5 and step 10 under a constant input. Φ over
   two cells is 0.

Defect 2 is not worked around here. **QD-5 starts from `initial_cells` = 2 and
lets the population grow by real mitosis under stimulus-driven tension**, which
is what the engine was built to do. Forced cloning would be testing an artifact.

## Method

Each stimulus becomes a fixed input vector from `qualia_sense`, fed to
`MitosisC.step(x)` every step. One engine per (stimulus, seed).

- 24 stimuli — a fixed stratified draw, two per category, taken in table order
  before any run — × 3 seeds × 600 steps. `dim` = 32, `hidden` = 64,
  `max_cells` = 32, `min_cells` engine default.
- **Scale reduced deliberately.** The toy ran 170 stimuli × 5 seeds; a torch
  engine per stimulus costs ~2.5 ms/step. 24 × 3 × 600 is about 2 minutes and
  is enough to separate 0.07 from 0.79. Stated so the reduction is not read as
  a like-for-like replication of QD-1..QD-4.

Measured, mirroring QD-4 so the numbers are comparable:

- **stimulus retained** — correlation between the pooled cell state
  (mean over cells) and the stimulus vector, at steps 1, 15, 50, 500.
- **settled** — whether the pooled state stops moving: mean step-to-step change
  over the last 100 steps, below 1% of its value over steps 1–10.
- **population** — live cell count and Φ over time.

## Hypotheses

**H17 (primary — the trade the toy could not escape)** — at step 500 the
population retains the stimulus at correlation ≥ 0.50 **while** settled.
The toy's best settled retention was 0.288, and everything above 0.7 there was
bought by failing to converge.

**H18 (the population is the reason)** — live cell count at step 500 is > 2.
If the population collapses to the floor, whatever H17 shows is a property of
two cells, not of a population, and H17 is reported but not attributed.

**H19 (identity is shared, not just present)** — between-stimulus / within-seed
distance on the pooled final state ≥ 2.0, the same bar QD-2..QD-4 used.

**H20 (Φ is alive)** — Φ at step 500 > 0 for at least half the runs. A dead Φ
means the population is degenerate regardless of what the correlations say.

## Scope limits

- Still names, not artworks; still form, not meaning.
- `phi_py` values are not comparable to `phi_rs` values.
- 24 stimuli × 3 seeds is a smaller design than QD-1..QD-4's 170 × 5.
- This tests `MitosisC` as it is, including defect 2's consequences for any
  configuration that does force-grow.

## Decision rules

- **H17 and H18 pass** → the population is the state the claim needed. The wall
  was the toy's, not the architecture's. `docs/qualia-decoder-spec.md` Phase 3
  reopens on `MitosisC`.
- **H17 passes, H18 fails** → two cells suffice, which makes the population
  story wrong even though the claim survives. Record it as such.
- **H17 fails** → the trade between identity and equilibrium is not an artifact
  of the toy. That is a claim about the architecture, and the spec should
  retire Phase 3 rather than keep looking for a substrate that rescues it.

Evidence via `sidecar verdict record` either way.
