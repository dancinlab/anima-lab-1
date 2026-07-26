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
