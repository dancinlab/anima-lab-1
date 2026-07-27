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

## Reproduce

```
python3 audit_fake_measurements.py          # live code
python3 audit_fake_measurements.py --all    # include archive/ and LEGACY
```
