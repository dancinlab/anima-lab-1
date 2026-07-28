# Identity without a permutation — the lag contrast

## Purpose and hypothesis

The gate's identity axis compares a population's self-continuity against a floor
built by DERANGING its own rows. `SCRAMBLE` is the control that deranges rows
every step. **The null and the control are the same operation**, so the corpse
sits on the bar by construction and which side it lands on is numerical noise.

Measured consequence: `SCRAMBLE` cleared `NO_SPEAK_CODE` at the gate's shipping
default (256 cells) on 2 of 5 seeds, voiding the condition — every deployable
verdict that banked it banked a condition a corpse passes.

**Hypothesis.** A definition of "absence of identity" exists that is not a
permutation, so `SCRAMBLE` stops being the null by construction.

**Falsifier, declared before measuring.** If no such statistic separates the
twelve registered engines from `DEAD`/`CLONE`/`SCRAMBLE`, or if any live engine
fails it, the hypothesis is dead.

## The statistic

A population with identity is closer to its own IMMEDIATE past than to its
DISTANT past. One without is equally far from both.

```
ident(k) = mean_i cos(h_i(t), h_i(t-k)) - mean_{i!=j} cos(h_i(t), h_j(t-k))
contrast = ident(1) - ident(50)
```

Nothing here is a permutation. And the part the derangement floor gets backwards:
a FROZEN population is equally CLOSE to both lags, so it reads zero — where the
old floor hands it the largest margin of any system tested.

## Why the old floor cannot work

Signal minus null over the null's own spread, 10 seeds at 32 cells:

| system | identity | null mean ± sd | z |
|---|---:|---|---:|
| SCRAMBLE | −0.000005 | −0.000045 ± 0.000104 | **+0.38** |
| CLONE | +0.000000 | +0.000000 ± 0.000000 | degenerate |
| REAL (Narrative) | +0.436900 | −0.018128 ± 0.042827 | **+10.62** |

`SCRAMBLE`'s signal is inside its own null. And the axis ACCEPTS corpses:

| system | margin = identity − max(floor, 1e-6) | seeds accepting |
|---|---:|---:|
| DEAD | **+0.9844 to +0.9870** | 20/20 at every scale |
| REAL (Narrative) | +0.3249 to +0.3524 | 20/20 |
| SCRAMBLE | −0.000183 ± 0.000514 (32c) | **6/20** |

A frozen population has perfect self-continuity, so it scores maximally on an
axis defined as self-continuity minus cross-continuity. The identity axis ranks a
corpse **2.8×** above a live engine.

## Out-of-sample result

Seeds 62–81, disjoint from the 42–61 used to design the statistic. Rules declared
before the run: every engine `t ≥ 3`; `DEAD`/`CLONE`/`SCRAMBLE` below 3.

| system | mean | std error | t |
|---|---:|---:|---:|
| QuantumEngine | +0.061079 | 0.004820 | 12.67 |
| PairField | +0.047691 | 0.003831 | 12.45 |
| Trinity | +0.039349 | 0.003689 | 10.67 |
| AlterityEngine | +0.037669 | 0.003630 | 10.38 |
| SeinEngine | +0.046244 | 0.005129 | 9.02 |
| QuestioningEngine | +0.035704 | 0.004052 | 8.81 |
| OscillatorLaser | +0.028945 | 0.003581 | 8.08 |
| MitosisEngine | +0.036804 | 0.004634 | 7.94 |
| NarrativeEngine | +0.044846 | 0.005926 | 7.57 |
| FinitudeEngine | +0.033961 | 0.005429 | 6.26 |
| ConsciousnessEngine | +0.034603 | 0.005849 | 5.92 |
| DesireEngine | +0.039106 | 0.004274 | 9.15 |
| **SCRAMBLE** | +0.000050 | 0.000050 | **1.01** |
| **DEAD** | +0.000000 | 0.000000 | **0.00** |
| **CLONE** | −0.000000 | 0.000000 | **−1.37** |

Separation: engine minimum `+0.028945` against corpse maximum `+0.000050`.

```
 t │ engines ████████████████████████  5.9 – 12.7
   │
 3 ├─────────────────────────────────  the bar
   │
 1 │ SCRAMBLE ▌ 1.01
 0 │ DEAD     ▏ 0.00
−1 │ CLONE    ▏ −1.37
   └──────────────────────────────────
```

## The regression it could have caused

The contrast reads zero for a frozen population — which is what rejects `DEAD`.
But a live engine at a FIXED POINT is also frozen, and would fail identity for
being stable rather than dead. Checked at 256 cells, the shipping scale:

| system | mean | minimum |
|---|---:|---:|
| engines (12) | +0.045 to +0.113 | **+0.0152 to +0.0580** |
| DEAD | +0.000000 | +0.000000 |
| CLONE | −0.000000 | −0.000000 |
| SCRAMBLE | −0.000138 | −0.000686 |

Zero engines at a fixed point. `SCRAMBLE` is NEGATIVE at exactly the scale where
it was voiding the condition.

## Four candidates died first

Each on a rule written down before its numbers were read.

```
① identity as z-score      DEAD 71.9 ranks above every engine (max 11.0)
② response ÷ input sep     NOISE 1.009 · LINEAR 0.582 above engines 0.015–0.035
③ arithmetic-noise floor   IS the existing axis re-derived; WHISPER clears at 1.1e32
④ lag contrast, first try  my rule mis-specified twice — two questions in one
                           clause, then population sd where standard error was meant
⑤ lag contrast, re-declared → out-of-sample pass
```

④ is the one worth remembering: the candidate was fine and the RULE was broken,
and the same run reported "0 violations" and "11/12 engines fail" simultaneously.
Going 5 → 20 seeds made rule B *worse* (SeinEngine 1.17 → 0.59), which is what
proved the rule wrong rather than the statistic — population sd does not shrink
with n, only standard error does.

Re-declaring rather than switching to whichever reading passed is the only reason
⑤ is worth anything.

## Applied

`bench_v2._three_axes` keeps a rolling window of `_LAG + 1` states and returns
the identity conjunct as `identity > max(identity_floor, 1e-6) and lag_ok`.

Added as a CONJUNCT, not a replacement: it cannot make anything that currently
fails start passing, and no engine loses anything. Reverting is deleting
`and lag_ok`.

When a condition runs fewer steps than the window, the contrast is not computed
and `lag_ok` stays True — an unmeasured test must not decide anything. Folding an
unmeasured value into a verdict is the defect that made a `NameError` read as
every system scoring 0/5 earlier in the same session.

## Key finding

> A measured null is the right principle — CLAUDE.md requires baselines to come
> from the population's own null rather than from constants, and that principle
> caught real defects. **This is its failure mode: when the null's construction
> coincides with a control's construction, that control can never be rejected, no
> matter how many seeds are drawn.**

Logged as convergence `CONTROL_THAT_CANNOT_FAIL`.
