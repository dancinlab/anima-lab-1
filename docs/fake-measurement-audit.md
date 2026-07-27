# Sweeping for constants that wear the name of a measurement

Every defect this session turned up was one shape — a value that reads like it
was measured and was not — and all of them were found by accident, one at a
time. `audit_fake_measurements.py` looks for the rest of that kind on purpose.

It reports; it does not judge. A constant can be a legitimate initial value.
The point is to put every one in front of a person instead of leaving it to be
discovered by a failing experiment.

## What it checks

| | | why |
|---|---|---|
| **A** | a numeric literal assigned to a measurement-named target, flagged when it sits in an `else`/`except` | this is the `cell_tension = 0.5` shape — a "default" that quietly becomes the operating value |
| **B** | a measurement-named local assigned and never read again | the `best_rule` shape — computed, counted, discarded |
| **C** | comparisons against bare numbers | where a scale mismatch hides, as `split_threshold=0.3` did |

260 live files (archive, benchmarks, LEGACY and tests excluded): **16** fallback
constants, **40** unread assignments, **252** bare-number comparisons.

## Triaged — 16 fallback constants

| site | verdict |
|---|---|
| `conscious_lm.py:364` `psi_tension = 1.0` | **fine.** Fires when layer tensions have zero spread; the formula is `1 − CV`, and CV = 0 gives exactly 1.0. The constant is the limit, not a stand-in. |
| `trinity.py:1202` `curiosity = 0.0` | **fine.** Fires before three trajectory points exist, when curvature is undefined. |
| `conversation_quality_scorer.py:238` `tension_coherence = 0.5` | **fine.** Labelled "neutral range", used as a mid-scale default. |
| `autonomous_loop.py:460` `novelty = 0.5` | **flagged.** Sits in a bare `except Exception:`. If the try body always raises, novelty is permanently 0.5 and nothing says so. Needs a runtime check. |
| `trinity.py:1060` `phi_preservation = 1.0` | **defect — see below.** |
| 11 others (`coherence = 0.0` ×3, `phi_* = 0.0` ×4, `confidence` ×2, counters ×2) | initial values or explicit zeros; no decision hangs on them. |

## The defect: the Hexad's ethics module cannot refuse

`EmpathyEthics.evaluate` (trinity.py) returns `allowed` as
`phi_preservation > 0.3`. `phi_preservation` is assigned in exactly two places
and takes exactly two values:

```python
if phi < phi_prev * 0.9:
    self.phi_preservation = 0.5   # "warning"
else:
    self.phi_preservation = 1.0
```

**0.5 and 1.0 both exceed 0.3, so `allowed` is `True` in every situation.**
Measured:

| situation | `phi_preservation` | `allowed` |
|---|---|---|
| Φ steady | 1.0 | True |
| Φ drops sharply — the "warning" state | 0.5 | **True** |
| Φ collapses entirely, 10 → 0 | 0.5 | **True** |
| Φ collapses **and** pain at maximum | 0.5 | **True** |

Two further dead paths in the same class:

- **`empathy_threshold=0.3`** is taken by `__init__`, stored, and **never read**
  anywhere in the repo.
- **`allowed` is never read by any caller.** The Hexad wiring at
  `trinity.py:1401` pulls `empathy`, `reciprocity` and `phi_preservation` out of
  the returned dict and drops the verdict.

So the E module of the Hexad architecture is **structurally incapable of
refusing, and unconsulted**. `NoEthics` — the explicit "no ethics filter" class
right below it — returns `allowed: True` unconditionally, which is the same
behaviour under an honest name.

## Not changed

Deciding what an ethics gate should block, and on what evidence, is a design
decision. The site is annotated so the next reader does not mistake a gate that
always opens for one that guards something.

## The auditor was measured against what it was built to find

Shipping a detector without measuring its recall is the same mistake as reading
a peak without checking what produced it. Measured against the six defects this
session found by hand, on the code **as it was before they were fixed**
(git 710c0ec) — testing a fixed file proves nothing:

| defect | check | caught |
|---|---|---|
| `cell_tension = 0.5` | A | ✓ |
| `phi_preservation ∈ {0.5, 1.0}` vs 0.3 | **D** | ✓ |
| `sha256` → `complexity` / `emotionality` / `entropy_input` | **E** | ✓ |
| `split_threshold` vs 0.3 | C | ~ flags the comparison, cannot judge scale |
| `best_rule` computed and discarded | — | ✗ |
| gate `g` inert | — | ✗ |

**2 of 6 before D and E, 3 of 6 plus one partial after.** The two misses are the
same shape — a value that *is* read but cannot affect the outcome — which needs
dataflow this tool does not do.

An **F** was attempted and removed: *an additive term not mentioning the loop
variable cannot change an argmax over that loop's candidates*, which is exactly
the inert gate `g`. As written it produced **1137 hits** across the live root
files and mis-attributed nested loops to the outermost variable, because it
never checked the part that makes the pattern a defect — that the loop's result
feeds a selection. A check that buries real findings under a thousand false ones
is worse than no check.

### D needed two precision fixes, both found by reading its output

| symptom | cause | effect |
|---|---|---|
| `_ec1_wealth ∈ [0.0] vs 5.0 → always False` | `+=` was not treated as an assignment, so counters read as "only ever 0" | 40 of the first 55 hits |
| `p_ngram ∈ [1.0] vs 0.05 → always False` | tuple unpacking `chi2, p = stats.chisquare(...)` was invisible, so only the skipped-data `else` literal was seen | two legitimate significance tests reported as unable to find significance |

With both fixed, **D reports exactly one hit across 260 live files, and it is
the real one.**

## Reproduce

```
python3 audit_fake_measurements.py          # live code
python3 audit_fake_measurements.py --all    # include archive/ and LEGACY
```

## The Φ ratchet writes the answer onto the test

`ConsciousnessEngine._phi_ratchet_check` restores the previous best hidden states
whenever Φ drops more than 20%. `PERSISTENCE` asks whether Φ decays. **The device
holds up the exact quantity the condition measures**, which is the shape this
session found nine times over — a value that satisfies its own test by
construction rather than by anything happening.

It was almost missed. The failing seed at α=0.08 fails only `PERSISTENCE`, and
the next step was going to be "does turning the ratchet on rescue it" — which is
a question about how to pass, not about whether passing would mean anything. The
right question is what a ratcheted Φ trajectory *is*.

Measured at 32 cells, ratchet reimplemented on `BenchEngine` so the comparison is
like-for-like:

| engine | PERSISTENCE | Φ trajectory |
|---|---|---|
| NOISE, no ratchet | FAIL | 0.228 → 0.203 → 0.098 → 0.205 → 0.159 |
| **NOISE + ratchet** | FAIL | 0.228 → 0.294 → **0.298 → 0.298 → 0.298** |
| real engine | **PASS** | 2.540 → 1.862 → 2.504 → 2.201 → 2.244 |
| **real + ratchet** | **FAIL** | **0.351 → 0.351 → 0.406 → 0.406** |

Two things, and the second was not expected.

**The ratchet pins Φ.** The repeated identical values are the signature — the
trajectory stops being a measurement of the system and becomes a record of the
best snapshot taken so far.

**On a live engine it makes things worse, not better.** The real engine passes at
Φ ≈ 2.2–2.5 and fails with the ratchet at Φ ≈ 0.35–0.41 — a 7× drop. Restoring on
every 20% dip traps it in an early low state it would otherwise have grown out of.
So the ratchet does not even succeed at the circular thing it is doing.

**Bearing on Law 31.** `CLAUDE.md` records the ratchet as one of "영속성의 3가지
열쇠", evidenced by `PERSIST3`: no collapse over 1000 steps, Φ growing ×62. If the
absence of collapse is the ratchet restoring away every drop, that benchmark
confirmed the device rather than the property. **Not resolved here** — `PERSIST3`
would have to be rerun with the ratchet off to separate them, and that is a claim
about a recorded law, not a defect to quietly fix.

### Follow-up: the ratchet is inert, not harmful

The section above said the ratchet "pins Φ" and "on a live engine makes things
worse (7× drop)". That came from an inline implementation which is not in the
repo and does not reproduce. Rebuilt as `bench_ratchet_law31.py` with the device
wrapped so it can be applied to any engine, and two defects surfaced in the probe
itself: the check was gated on `getattr(self, 'step_count', 0) % 10 == 0`, and
`NoiseEngine` has no `step_count`, so the noise arm was ratcheted every step
against the real arm's every tenth — 876 restores vs 9, an artefact. And
`PhiIIT.compute` returns `(phi, components)`, appended whole.

With both fixed, `ConsciousnessEngine`, 1000 steps, 3 seeds, max 64 cells:

| ratchet | cosine | Φ(IIT) final | restores | cells | persists |
|---|---|---|---|---|---|
| ON | +0.4683 | 0.232 / 0.325 / 0.275 | 29 / 62 / 64 | 62–63 | no |
| OFF | +0.5526 | 0.359 / 0.228 / 0.244 | 0 / 0 / 0 | 62–64 | no |

**Law 31's key #1 is unsupported** — Φ does not hold up better with the ratchet.
But the earlier "7× drop" is also withdrawn: the ratchet does not degrade the
engine either. It fires 29–64 times and changes almost nothing.

Three further claims made during this investigation and refuted by measurement:

- *"The engine never divides"* — it reaches 63 cells at 1000 steps and logs 1178
  split events by 1500. The 600-step probe that showed 2 cells was simply short;
  growth begins between 600 and 1000.
- *"The q0.90 split calibration blocks mitosis"* — single-step tension clears the
  0.005642 bar 7.2% of the time and the 5-step mean only 0.7%, but that is ample
  across 1500 steps × 62 cells, and splits fire.
- *"A step costs 3 s; the run took 50 minutes"* — 1000 steps cost 5.15 s and four
  `PhiIIT.compute` calls at n=63 cost 0.26 s. The 50-minute figure does not
  reproduce and appears to have been a delayed background notification read as
  runtime.

What survives from the ratchet line of inquiry is the thing that prompted it:
`PERSISTENCE`'s Φ rule passes plain noise with no ratchet at all
(`docs/phi-rs-direction.md`), so the pre-session gate was vacuous on its own
regardless of what the ratchet does.
