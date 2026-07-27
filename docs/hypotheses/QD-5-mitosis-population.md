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

---

# Results

`.venv/bin/python bench_psi_mitosis.py` — 34 stimuli × 3 seeds × 600 steps.
Verdict 🔴 **FAIL (0/4)**, evidence in `state/QD-5.txt`.

| | prediction | measured | |
|---|---|---|---|
| **H17** | retained ≥ 0.50 while settled | 0.409, settled 100% | **FAIL** |
| **H18** | live cells > 2 | **2.0** | **FAIL** |
| **H19** | between/within ≥ 2.0 | within 0.764, between 0.158, **ratio 0.21** | **FAIL** |
| **H20** | Φ > 0 in ≥ half the runs | 0% alive, Φ = 0.00 | **FAIL** |

## The headline is H18, not the 0/4

**The population never formed.** Cell count stayed at the `min_cells` floor of
2 for all 600 steps, in every run. H18 exists precisely so this cannot be
glossed: whatever the other numbers show is a property of two cells, and
**QD-5 did not test the population hypothesis it was written to test.**

## Why — measured, not inferred

| | value |
|---|---|
| `split_threshold` | 0.30 |
| `merge_threshold` | 0.05 |
| tension actually produced | mean 0.0220 · max 0.0369 · min 0.0074 |
| steps above the split bar | **0 / 300** |
| steps below the merge bar | **300 / 300** |

The tension this engine generates is an order of magnitude below the bar that
gates division — max 0.037 against a 0.30 threshold, **8× short** — and sits
below the merge bar on every single step. **The engine is permanently in merge
territory and never reaches split.** The population can only shrink. Mitosis,
in the mitosis engine, never fires.

Checked against three input regimes — one fixed stimulus, fresh noise every
step, and 24 stimuli in rotation — the cell count was 2 in all three. This is
not a novelty-starved protocol.

### Correction, made in QD-6

The "8× short" above is measured with **this bench's** input. Sweeping the
input scale changes the picture and one sentence here was overstated:

| input | norm | tension max | cells |
|---|---|---|---|
| `qualia_sense` (this bench) | 3.66 | 0.0285 | 2 |
| `torch.randn` — the engine's own default | 6.09 | 0.0833 | 2 |
| `qualia_sense` × 3 | 10.98 | 0.1917 | 2 |
| `qualia_sense` × 10 | 36.61 | 2.3646 | **32** |
| `randn` × 5 | 32.55 | 2.3789 | **32** |

So it is **not** true that no input can reach the bar — inputs about 5× the
engine's own default do, and then the population grows to `max_cells`
immediately. `tension = (output ** 2).mean()` (mitosis.py:60) is an absolute
magnitude, so it tracks input scale directly.

What survives, and is stronger than the original claim: the bar is unreachable
at **every input scale the repo itself uses**. The engine's own default is 3.6×
short; `mitosis.demo()`'s `text_to_vector` output has norm 0.051 and produces
T=0.01 against that demo's `split_threshold=1.5`, 150× short. The demo's one
`MITOSIS` line comes from a `--- Forced Mitosis Demo ---` block calling
`split_cell()` directly; across its 30 threshold-driven steps, zero splits
occur. Threshold-driven mitosis has never fired on any path in this repo.

## What the run does show, attributed honestly

- **Retention 0.409 at 100% settled.** Better than the toy's best settled
  retention (0.288 at 100% convergence) and below the 0.50 bar. It is a
  two-cell number, not a population number.
- **H19 ratio 0.21 — the seed beats the stimulus.** Within-seed distance
  (0.764) is nearly five times the between-stimulus distance (0.158). Two runs
  of *different* stimuli end up closer together than two runs of the *same*
  stimulus with different random initialisation. Whatever this state encodes,
  it is mostly where it started, not what it saw.
- **Φ = 0 throughout**, which follows from two cells: the `phi_py` estimator has
  nothing to integrate over. This is a consequence of H18, not an independent
  failure.

## Consequence

The pre-registered rule for an H17 failure said to retire Phase 3. That rule
does not fire cleanly here, because H18 failed first and the test was therefore
void as a test of populations. Retiring the claim on a void test would be as
wrong as rescuing it on one.

What is now established instead is a defect in the substrate itself, with a
number attached: **a division threshold 8× above the tension the engine
produces**. That is the thing to fix before the population question can be
asked at all — and it is a question about `MitosisEngine`'s calibration, not
about the qualia decoder.

`docs/qualia-decoder-spec.md` Phase 3 stays where QD-4 left it: unsupported in
the toy, and now untested — not disproven — on the population.
