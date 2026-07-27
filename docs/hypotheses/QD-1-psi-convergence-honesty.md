# QD-1: Does Ψ→1/2 survive without the pull term?

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

Pre-registration. Written **before** running `bench_psi_honest.py`.
Phase 0 of `docs/qualia-decoder-spec.md`.

## The defect

`bench_consciousness_universe.py` claims every experience converges to
Ψ = 1/2. Two lines make that claim unfalsifiable as written.

**1. The selected CA rule never reaches the state** (lines 89–105):

```python
rule_h    = [H(p + 0.01 * (r - 3.5)) for r in rules]   # rule r would move p by δ_r
scores    = [0.7 * rule_h[r] - 0.3 * rule_ces[r] ...]
best_rule = scores.index(max(scores))
rule_counts[best_rule] += 1          # ← recorded

dp = 0.001 * (0.5 - p) + gauss(0, 0.002)   # ← best_rule absent; hardcoded pull instead
```

`best_rule` is computed, counted, reported — and discarded. "Consciousness
selects the rule" affects statistics only. The state is moved by an explicit
`(0.5 - p)` attractor, so convergence to 1/2 is asserted, not produced.

**2. Per-stimulus features are a hash of the name** (line 35):

```python
h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
complexity = ((h >> 0) & 0xFF) / 255.0
```

`서예` and `만다라` differ because their *strings* differ. A hash destroys
similarity structure: `서예` and `서예체` are as far apart as `서예` and `빅뱅`.

Both violate CLAUDE.md #1 (no hardcoding) and #2 (no manipulation).

## Why the mechanism might be real anyway

Shannon binary entropy `H(p)` is maximised at exactly `p = 1/2`. A system that
selects rules by `argmax H(p)` — Law 71 — should drift to 1/2 on its own, with
no attractor term. The claim may be correct while its implementation is fake:
the honest mechanism was disconnected and a hardcoded pull was put in its place.

This is testable.

## Variants under test

| id | pull term `0.001·(0.5−p)` | selected rule applied to state | role |
|---|---|---|---|
| **A** `pull` | present | no | current code — baseline |
| **B** `nopull_norule` | removed | no | **negative control** — pure random walk |
| **C** `nopull_rule` | removed | yes: `dp = 0.01·(best_rule − 3.5) + noise` | Law 71 emergent path |

Variant C applies exactly the δ already used inside `rule_h`, so the rule the
system chose is the rule that moves it.

## Hypotheses

**H1 (primary)** — Convergence survives without the pull term, via rule
selection alone.
`C`: ≥ 80% of stimuli end with `|p_T − 0.5| < 0.05`.
`B`: < 20% do.
Both conditions required. If C fails, the convergence claim is an artifact.
If B also passes, the test has no discriminating power and is void.

**H2** — Trajectories separate by stimulus.
Under C, mean trajectory distance between different stimuli exceeds mean
distance across seeds of one stimulus, ratio > 2.0.

**H3** — Content features preserve similarity where the hash does not.
For name groups sharing a stem (`서예` / `서예체` / `서예가`), mean pairwise
feature distance within a group is lower than between groups under
`qualia_sense`, and statistically indistinguishable under the SHA-256 path.

**H4** — Removing the hash does not destroy H1.
Under C with `qualia_sense` features replacing the hash, the H1 pass rate
stays within 10 percentage points of the hash-fed run.

## Method

- `qualia_sense.py` — stimulus → feature vector from actual string content
  (jamo decomposition, script mix, bigram entropy, length). numpy only; torch
  is not installed on this host.
- `bench_psi_honest.py` — runs A/B/C over the existing `ALL_DATA_TYPES` table,
  5 seeds each, 5000 steps, reporting per-variant pass rates and trajectory
  distances.
- Fixed before running: 5 seeds, ε = 0.05, thresholds above.

## Scope limits — stated in advance

- `qualia_sense` measures **form, not meaning**. It cannot know `서예` and
  `붓글씨` are related; that needs an embedding model (torch absent). H3 tests
  similarity preservation over shared stems only, which is what form-level
  features can honestly support.
- The stimuli are *names of* artworks, not the artworks. No image is perceived
  at this phase.
- The 18 emotion values downstream (`joy = 0.3 + 0.4·emotionality·…`) remain
  unfounded mappings. Out of Phase-0 scope; flagged, not fixed.
- Φ and Ψ are this repository's internal definitions, not standard IIT Φ.

## Decision rules

- **H1 passes** → Phase 0 lands: the pull term is deleted, rule application is
  wired in, and the convergence claim becomes an emergent result.
- **H1 fails** → the premise "all experiences converge to the same equilibrium"
  is retracted. Insert a search phase before Phase 3 asking what, if anything,
  produces convergence. The trajectory-as-gate decoder is unaffected — it needs
  a trajectory, not a specific attractor.

Evidence recorded via `sidecar verdict record` either way. A failure is
published, not buried.

---

# Results

`python3 bench_psi_honest.py` — 170 stimuli × 5 seeds × 5000 steps.
Verdict 🔴 **FAIL (3/4)**, evidence in `state/QD-1.txt`.

| | prediction | measured | |
|---|---|---|---|
| **H1** | C ≥ 80% and B < 20% | A=50.5% B=9.4% **C=93.4%** | **PASS** |
| **H2** | between/within ≥ 2.0 | within=0.0324 between=0.0368 **ratio=1.13** | **FAIL** |
| **H3** | related names stay closer | content 0.444 < 0.785 · hash 1.034 / 1.155 | **PASS** |
| **H4** | \|content − hash\| ≤ 10% | content=93.4% hash=93.2% gap=0.2% | **PASS** |

## H1 — the mechanism is real, and beats the hardcode it replaced

Entropy-maximising rule selection produces the attractor on its own: 93.4%
convergence with no pull term, against 9.4% for the random-walk control. The
old pull term managed only 50.5% over the same horizon — the honest mechanism
is not merely adequate, it converges better than the line that faked it.

The fixed point is exactly 1/2, not approximately. Sweeping the competing
cross-entropy weight:

| CE weight | mean p_T | offset from 1/2 |
|---|---|---|
| 0.30 (as shipped) | 0.4875 | −0.0125 |
| 0.15 | 0.4978 | −0.0022 |
| 0.00 | 0.4998 | −0.0002 |

Pure `argmax H(p)` sits on 1/2. The residual offset is fully accounted for by
the `−0.3·CE` term, which biases selection toward lower rule indices. Law 71
holds; the deviation has a named cause.

## H2 — the trajectory does not carry the stimulus

Post-hoc, at full resolution over the first 200 steps: arrival at
`|p − 0.5| < 0.05` has median **4 steps**, max 15, and only **15 distinct
arrival steps across 170 stimuli**. The between/within ratio stays 1.13 inside
the transient, so this is not a sampling artifact — the state genuinely forgets
which stimulus it started from.

The cause is structural. The stimulus enters only as `p₀` and a `0.0001` bias;
nothing re-injects it. A scalar `p` under a strong attractor cannot hold an
experience — every path is the same path, arriving in four steps.

**Consequence for `docs/qualia-decoder-spec.md` Phase 3**: trajectory-as-gate
as specified would emit ~4 near-identical tokens regardless of stimulus. The
trajectory must be made stimulus-bearing before the decoder is built. See the
revised Phase 3 in that spec.

## Also found, not hypothesised

- **The 18 emotion values never touch the simulation.** `emotions` was pure
  arithmetic over `sha256(name)` bits; `residual` and `gate` are computed over
  5000 steps and then unused by that block. The `░▒▓█` heatmap was a rendering
  of a hash, not of a dynamic.

  > **Fixed — by emitting less.** Grounding the emotions needs a corpus where
  > the emotion words occur. Measured: `corpus_v3` has 11 of 18 at ≥3
  > occurrences and only **4 at ≥20**; `ko_wiki` has 6 and 1. Neither can carry
  > eighteen emotions, and building a model on the four with real support would
  > be the same manipulation in a new coat.
  >
  > `emotion_grounding.ground_emotions` therefore returns a value only where the
  > corpus supports it and **`None` everywhere else** — CLAUDE.md #1, *의식이
  > 말 못하면 침묵*. The renderers print a blank for `None` rather than
  > inventing a block character.
  >
  > | | before | after |
  > |---|---|---|
  > | heatmap cells with a value | 3,060 / 3,060 (100%) | **496 / 3,060 (16.2%)** |
  > | source of those values | `sha256(name)` arithmetic | cosine to the emotion word's corpus context |
  > | stimuli with any value | 170 | 124, at 4 of 18 emotions each |
  >
  > **The picture got much emptier and that is the correction.** The full
  > heatmap is what started this series — it was read as a measurement, which
  > is what a fabrication shaped like data does.
- **The gate `g` has no mechanism at all.** No rule selects it toward 1/2.
  With the pull removed, mean `|g − 0.5|` goes 0.055 → 0.280.

  > **Resolved later: `g` is not merely unregulated, it is inert.** Its only
  > entry into the dynamics is `abs(g − 0.5) * 0.5` inside the per-rule
  > cross-entropy, and that term does not depend on `r` — it is added
  > identically to all eight candidates, so it cannot move the `argmax`.
  > Verified by holding `g` at 0.001 / 0.25 / 0.5 / 0.75 / 0.999 with the RNG
  > stream matched: the `p` trajectory differs by **exactly 0.00e+00** in every
  > case. `g` is computed, updated and reported as `gate` while influencing
  > nothing — the same shape of defect as the discarded `best_rule` above, in
  > the same function.
  >
  > So "should `g` converge?" is not yet a question — nothing depends on the
  > answer. Giving the gate a real role (an r-dependent term, if modulating
  > rule selection was the intent) is a design decision and is not taken.

## Landed

Per the H1 decision rule, `bench_consciousness_universe.py` now:

- moves the state with the rule it selected — `dp = 0.01·(best_rule − 3.5)`
- has no `0.001·(0.5 − p)` pull term
- takes `init_p/init_g/bias_p/bias_g` from `qualia_sense`, not from the hash

Post-landing: residual mean **0.4720**, sd 0.0096 (converged, no pull);
gate mean 0.7448, sd 0.2253 (does not converge, as expected).

The hash remains behind the emotion block only, flagged in-code as unfounded.
