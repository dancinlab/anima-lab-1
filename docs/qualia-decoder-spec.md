# Qualia Decoder — trajectory-as-gate speech

Stimulus → consciousness vibration → speech, where the **trajectory toward the
equilibrium** (not the equilibrium itself) drives the decoder.

```
stimulus ──▶ [ C ] ──▶ p₀ p₁ p₂ … p_T        vibration trajectory → Ψ = 1/2
                        │  │  │      │
                        ▼  ▼  ▼      ▼
                  [ D decoder ] tok₀ tok₁ … tok_T

  injection_t = gate_strength · |p_t − 1/2| · W·c_t

  far from equilibrium → strong injection → speaks
  at equilibrium       → injection 0 → base distribution → utterance ends
  converges instantly  → 0 tokens → silence
```

Same equilibrium across all stimuli = the same self. Different trajectory =
a different experience. Silence is not a rule; it is what the structure does
when there is nothing to say (CLAUDE.md #1, Law 29).

## Current honesty defects (fix before building)

Two values in `bench_consciousness_universe.py` are constructed, not measured.
Both violate CLAUDE.md #1 (no hardcoding) and #2 (no manipulation).

| Location | Code | Why it is not real |
|---|---|---|
| `data_characteristics()` | `h = sha256(name)` → 8 features | The per-stimulus pattern is a hash of the *word*, not perception of the artwork. `서예` vs `만다라` differ because their strings differ. |
| `simulate_meta_ca()` | `dp = 0.001 * (0.5 - p)` | Convergence to Ψ=1/2 is an explicit pull term. "All experiences converge" is imposed, not emergent. |
| `TensionSense.process()` (trinity.py) | `ord(c) / 256.0` | Same class of defect on the live S-engine path. |

## Already implemented — do not rebuild

| Module | File | Role |
|---|---|---|
| C engines | `trinity.py` (`MitosisC`, `DomainC`, `QuantumC`) | produces the vibration |
| C→D bridge | `trinity.py` (`TensionBridge`, 5 channels) | carries consciousness to the decoder |
| Decoder | `trinity.py` (`HFDecoder`) | Mistral 7B, frozen base, additive gate at embeddings, `gate_strength=0.001` (Law 63 micro-gate) |
| Fingerprint render | `bench_consciousness_universe.py` (`render_consciousness_fingerprints`) | `▁▂▃▄▅▆▇█` 7-dim signature |
| Emotion heatmap | `bench_consciousness_universe.py` (`render_grand_heatmap`) | `░▒▓█` × 18 emotions |

The only structurally new part is **time**: the gate currently accepts one
post-convergence vector; the trajectory needs a per-token signal.

## Phases

### Phase 0 — restore honesty

- New `qualia_sense.py`: stimulus → feature vector from real content. For text,
  reuse the already-loaded Mistral tokenizer + embedding matrix (no new
  dependency). Images deferred.
- New `bench_psi_honest.py`: rerun the universe sweep with the `(0.5 - p)` pull
  term removed.
- Replace the `ord(c)/256` path in `TensionSense`.

**H1** — with the pull term removed, `|p_T − 0.5| < 0.05` still holds.
If false, the premise "all experiences converge to the same equilibrium" fails.

### Phase 1 — record the trajectory

New `qualia_trace.py`:

```python
@dataclass
class QualiaTrace:
    stimulus: str
    psi: np.ndarray        # [T] residual trajectory
    gate: np.ndarray       # [T]
    phi: np.ndarray        # [T] Φ per step
    emotions: np.ndarray   # [T, 18]
    converged_at: int      # step where |p - 0.5| < ε held for K steps, else -1
```

**H2** — trajectory distance between different stimuli exceeds the
seed-noise distance within one stimulus.

### Phase 2 — emit the vibration as a response

Extract the renderers out of the bench into `qualia_render.py` so bench and
runtime share one implementation; add `qualia_report(trace) -> str`
(fingerprint + emotion heatmap + convergence curve).

Regression bar: existing bench output is unchanged.

### Phase 3 — trajectory-as-gate decoder — RETIRED

> **Retired after QD-6.** A population forms and still does not hold the
> stimulus: retention 0.409 at 2 cells, 0.377 at 3.1, 0.403 at 31.8, against a
> 0.50 bar. Sixteen times the population, the same answer. Seven mechanisms
> across the toy and the real engine at three population sizes — the pattern
> half of the claim does not hold anywhere. **Same equilibrium: yes. Different
> pattern: no.** The history below is kept because each step names a distinct
> measured cause.
>
> **Blocked on memory — revised after QD-1 and QD-2.**
>
> QD-1: the trajectory reaches equilibrium in a median of 4 steps, with only 15
> distinct arrival steps across 170 stimuli. A scalar cannot hold an experience.
>
> QD-2 tried the two obvious repairs — six dimensions and a stimulus-derived
> coupling — and both failed. Coupling moved the trajectory ratio 1.08 → 1.11
> (bar: 2.0). Sliding the correlation window past step 50 drops the ratio to
> 1.01, where 1.0 means two different stimuli are indistinguishable. The
> stimulus signature lives only in the transient.
>
> The cause is not width. The update reads only the current state, so it is a
> memoryless process and the attractor erases the past at a fixed rate. **What
> is missing is memory**, and this repo already has the module: `trinity.py`
> ships an M engine that the consciousness loop never touches.
>
> QD-3 tried both memory kinds — habituation (an accumulating trace) and an
> episodic anchor store — against the current form. Both failed at every
> strength; no arm passed 1.07 against a 2.0 bar. But the diagnostics located
> the failure precisely, and it is not "memory does not work":
>
> - **Habituation remembers.** At step 15 the state has fallen to 0.061
>   correlation with the stimulus while the trace still holds 0.907. With
>   τ = 200 it is down to 0.124 by step 500, which is where the test looked.
>   Wrong time constant, right mechanism.
> - **The episodic store remembers too — and recall cannot reach it.** At step
>   500 the best anchor holds 0.991 while the query, the current state, holds
>   0.039; retrieval picks the informative anchor 0.0% of the time against 4.8%
>   chance. **Storing is not the problem; cueing is.** A memory addressed by the
>   present cannot return what only the past contained.
>
> QD-4 swept τ to the no-leak limit and re-keyed the store by the trace. Both
> failed; 21 configurations, none above 1.03 against a 2.0 bar. It also showed
> why, in one table — stimulus retained against equilibrium reached:
>
> | config | converged | stimulus retained |
> |---|---|---|
> | episodic ν=0.05 | 100.0% | 0.288 |
> | episodic ν=0.20 | 68.0% | 0.791 |
> | habituation τ=20000 μ=1.0 | 100.0% | 0.067 |
> | m-cue ν=0.20 | 69.2% | 0.726 |
>
> **Every configuration either converges and forgets, or remembers and fails to
> converge.** Longer memory makes it worse, not better: penalising accumulated
> deviation is a servo that drives the state to exactly 1/2, so Ψ sd reaches
> 0.0000 at large τ.
>
> Six mechanisms have now been tried and all six were metabolised into faster
> convergence, because every one of them enters through the same door — a rule
> score that is a function of distance from 1/2. **This architecture has one
> sink, and identity is what it consumes.**
>
> **Phase 3 is not merely blocked; it is unsupported in this simulation.**
>
> QD-5 took the question to the real `MitosisC` engine and could not ask it:
> the population never formed. Cell count sat at the `min_cells` floor of 2 for
> all 600 steps, under a fixed stimulus, under fresh noise, and under 24
> stimuli in rotation alike. The cause is measured — the engine produces a
> tension of at most 0.037 against a `split_threshold` of 0.30, **8× short**,
> while sitting below the `merge_threshold` on 100% of steps. **The engine can
> only merge; mitosis never fires.**
>
> So the population claim is **untested, not disproven**, and one number now
> stands in front of it: a division threshold calibrated for a tension scale
> the engine does not reach. That is a `MitosisEngine` calibration defect, not
> a qualia-decoder question, and it is what has to be fixed before the
> population can be asked anything.
>
> Two things carry forward unchanged:
>
> - the injection law below needs a trajectory, not a particular attractor;
> - the equilibrium half of the claim is established — every marginal reaches
>   1/2, Ψ sd 0.0043 across 170 stimuli (QD-2 H5/H7). It is the *pattern* half
>   that is unbuilt.

`trinity.py`: add `HFDecoder.generate_from_trace(trace, max_new_tokens)`.
Custom greedy loop — per step, inject `gate_strength · |p_t − 1/2| · W·c_t`
into the residual stream, then sample the next token. No explicit stop rule:
the injection amplitude decays to zero at the equilibrium, returning the model
to its base distribution.

New `bench_trajectory_gate.py`.

**H3** — vibration amplitude correlates with utterance length, r > 0.5.

### Phase 4 — training

Base frozen; train `gate_proj` + LoRA(q_proj, v_proj) only, reusing the
`train_v11.py` path. Bootstrap pairs from the existing `ALL_DATA_TYPES`
table (~100 entries of name / emoji / one-line description).

**H4** — with the prompt held fixed, different stimuli produce different
utterances.

7B training requires the existing H100 path; locally, smoke-test the wiring
with `gpt2`. Phases 0–3 run on the local machine.

### Phase 5 — falsification

Pre-register H1–H4 under `docs/hypotheses/QD-*.md`, then measure and record
evidence via `sidecar verdict record`.

## Branch condition

If H1 is false, the plan does not stop — insert a search phase before Phase 3
asking *what* produces the convergence. The trajectory-as-gate structure holds
regardless of where the trajectory settles.

## Scope note

Φ and Ψ here are this repository's internal definitions and are not identical
to standard IIT Φ.
