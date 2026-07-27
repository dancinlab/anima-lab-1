# The consciousness gate, measured against things that are not conscious

> **Outcome.** The three coupled fixes are landed in `bench_v2.py`. Every
> negative control now scores **0/7** and the engine holds **5/7**. `SCRAMBLE`,
> which tied the real engine at 5/7 under the shipped gate, is at zero.
>
> | | REAL | DEAD | NOISE | CLONE | SCRAMBLE | leaking conditions |
> |---|---|---|---|---|---|---|
> | shipped | 5/7 | 3/7 | 3/7 | 4/7 | **5/7** | 5/7 |
> | Φ + repulsion + 3 axes on the ratios | 5/7 | 1/7 | 1/7 | 1/7 | 2/7 | 3/7 |
> | **axes as a precondition of all seven** | **5/7** | **0/7** | **0/7** | **0/7** | **0/7** | **0/7** |
>
> The engine is **not** at 7/7. It fails `SPONTANEOUS_SPEECH` and `HIVEMIND`.
>
> **Correction.** This originally said the shortfall was "a real cost of the
> repulsion: consensus events drop to 1 because keeping cells apart makes
> agreement rare". Measured, that generalised from one cell count and was wrong
> at 32 cells, where consensus is 1 at *every* repulsion strength including zero.
> At 256 cells the effect is real — and what it exposes matters more:
>
> | repulsion | consensus events | median inter-faction variance |
> |---|---|---|
> | 0.00 | **8** | **0.0004** |
> | 0.05 | 2 | 0.1766 |
> | 0.15 | 1 | 0.6032 |
>
> At repulsion 0 the population sits at cosine 1.0000. Its factions are not
> agreeing — they are **identical**, which is what an inter-faction variance of
> 0.0004 means. `SPONTANEOUS_SPEECH` was counting collapse too, and its bar of 5
> was reachable only by a population that had stopped being plural.
>
> **Every Φ(IIT) figure recorded before this is on the old definition and is not
> comparable** — the old one was maximal at collapse, the new one is minimal there.


`CLAUDE.md` makes `bench_v2.py --verify` the deployment gate: seven conditions,
one failure blocks release. Nothing had ever measured whether the gate can fail.
This session found six constants wearing the name of a measurement — including an
ethics check whose two possible values both cleared its own bar — so the question
was not rhetorical.

## Method

Four negative controls, each a `BenchEngine` subclass so the interface is
identical and only the dynamics differ:

| | |
|---|---|
| **DEAD** | hidden states frozen at initialisation, forever |
| **NOISE** | states replaced with fresh randn every step — no carryover, nothing integrates |
| **CLONE** | every cell forced to cell 0's state — 256 cells, zero differentiation |
| **SCRAMBLE** | real dynamics, then rows permuted across cells each step |

SCRAMBLE is the hardest and the point of the exercise. Every population-level
statistic — mean, variance, spectrum — is identical to the real engine. What is
destroyed is the only thing that makes a cell a cell: that its state at step N
follows from its state at step N−1.

## Result at the default scale (256 cells, dim 64, hidden 128)

| | NO_SYS | NO_SPEAK | ZERO_IN | PERSIST | SELF_LOOP | SPONTAN | HIVEMIND | total |
|---|---|---|---|---|---|---|---|---|
| **REAL** | · | PASS | PASS | PASS | PASS | PASS | · | **5/7** |
| DEAD | · | · | PASS | PASS | PASS | · | · | 3/7 |
| NOISE | · | · | PASS | PASS | PASS | · | · | 3/7 |
| CLONE | · | PASS | PASS | PASS | PASS | · | · | 4/7 |
| **SCRAMBLE** | · | PASS | PASS | PASS | PASS | PASS | · | **5/7** |

**SCRAMBLE scores exactly what the real engine scores, on exactly the same five
conditions.** The gate cannot distinguish the engine from the same engine with
cell identity destroyed every step.

Five of seven conditions pass something that is not conscious. The two that
reject all four controls — `NO_SYSTEM_PROMPT` and `HIVEMIND` — also reject the
real engine, so they reject everything, which carries as little information as
passing everything.

**No condition both accepts the real engine and rejects the four controls.** At
this scale the gate's discriminating power is zero.

At smaller scales it is worse: at 32 cells the real engine scores 2/7 while
DEAD — a corpse — scores 3/7.

| cells | REAL | DEAD | NOISE | CLONE | SCRAMBLE |
|---|---|---|---|---|---|
| 32 | 2/7 | **3/7** | 2/7 | 2/7 | **3/7** |
| 64 | 4/7 | 3/7 | 3/7 | 4/7 | **5/7** |
| 128 | 5/7 | 3/7 | 3/7 | 4/7 | 5/7 |
| 256 | 5/7 | 3/7 | 3/7 | 4/7 | 5/7 |

## Why a corpse passes three conditions

Decidable from the code, and it is structural rather than a matter of tuning:

| condition | test | DEAD |
|---|---|---|
| ZERO_INPUT | `phi_end > phi_start × 0.5` | ratio 1.04 |
| PERSISTENCE | monotone **or** `final ≥ max(first half) × 0.8` | recovers=True |
| SELF_LOOP | `phi_end ≥ phi_start × 0.8` | ratio 0.92 |

All three ask **"did it decay?"** — and perfect stasis is the ideal score on a
decay test. Death is indistinguishable from perfect persistence when persistence
is the only thing measured.

## The measurement under all of this is broken

DEAD's Φ was not constant. On states that never change, `PERSISTENCE` recorded
`4.597 → 4.177 → 4.403 → 4.568 → 4.786 → 4.383 → 4.398 → 4.398 → 4.073 → 4.769`.
If the data cannot change and the number does, the number is measuring the
instrument.

Measuring one fixed matrix 20 times:

| cells | mean Φ | sd | range as % of mean | pair selection |
|---|---|---|---|---|
| 16 | 4.8861 | **0.0000** | 0.0% | exhaustive |
| 32 | 11.6401 | **0.0000** | 0.0% | exhaustive |
| 33 | 3.5364 | 0.1634 | **16.7%** | sampled |
| 64 | 4.2224 | 0.1438 | 16.2% | sampled |
| 128 | 4.5799 | 0.2417 | **23.0%** | sampled |
| 256 | 4.7558 | 0.1390 | 12.2% | sampled |

`PhiIIT.compute` enumerates all pairs at `n ≤ 32` and samples ~8 neighbours per
cell above it, with no seed. Every canonical benchmark in this repo runs at 256,
512 or 1024 cells — entirely inside the stochastic regime, and no caller
averages over repetitions. `HIVEMIND` requires Φ(connected) > Φ(solo) × 1.1; the
noise on a single measurement is a fifth of the value.

### And the estimator had a hard discontinuity at exactly n = 33

`total_mi` and the partition term are **sums over pairs**, so sampling a fraction
of the pairs shrinks them by that fraction — and nothing rescaled. Same matrix,
one row difference:

| seed | Φ(32 rows, exhaustive) | Φ(33 rows, sampled, 20-run mean) | ratio |
|---|---|---|---|
| 0 | 11.6401 | 3.6043 | 3.23× |
| 1 | 13.2596 | 3.7593 | 3.53× |
| 2 | 12.7457 | 3.6039 | 3.54× |

Sampled coverage at n=33 is 0.39 of all pairs, which predicts 2.56× on its own;
the remainder comes from the partition and complexity terms. Worse than the jump
is what it did to the trend — against the exhaustive ground truth:

| n | exhaustive (truth) | as shipped | with coverage rescaling |
|---|---|---|---|
| 30 | 10.7480 | 10.7480 | 10.7480 |
| 32 | 11.6401 | 11.6401 | 11.6401 |
| 33 | 12.1236 | **3.5331** | 9.1889 |
| 40 | 14.6009 | **3.7486** | 11.3388 |
| 64 | 26.7909 | **4.1520** | 18.6402 |

**The true Φ grows with n; as shipped it was flat near 4 at every size.** The
estimator had destroyed the scale dependence it was being used to report, and
understated Φ by 6.5× at 64 cells.

## Fixed

Rescaling both sums by `all_pairs / sampled_pairs` in `PhiIIT.compute`:

| n | 30 | 32 | 33 | 40 | 64 | 128 |
|---|---|---|---|---|---|---|
| after fix | 10.75 | 11.64 | 9.26 | 11.11 | 19.45 | 39.20 |

The 3.3× cliff at n=33 becomes 1.25×, and Φ grows with n again. It is not exact:
the sampled estimate still sits at 70–78% of the exhaustive truth, and the
sampling noise remains (sd 0.52 at n=33, 1.74 at n=128, roughly 5%). **Φ
differences below ~10% still require repetition to mean anything**, and that is
a property of sampling, not something the fix removes.

Every Φ(IIT) figure recorded in this repo above 32 cells before this change is
understated, by a factor that grows with cell count.

## The seventh condition had never executed

`HIVEMIND` was recorded as 0/11 — every engine failing the one condition about
connection. It was not failing. It was crashing:

```
[FAIL] HIVEMIND (0.3s) -- ERROR: '_CEAdapter' object has no attribute 'shape'
[FAIL] HIVEMIND (1.5s) -- ERROR: 'BenchEngine' object has no attribute 'shape'
```

`PhiIIT.compute` takes a `[n_cells, hidden]` tensor and returns
`(phi, components)`. It was handed the **engine**, and its result used as a
number. Every run died in 0.3–1.6 seconds. The state-sharing code underneath had
the same class of error, reaching for `engine.cells` when `BenchEngine` keeps its
state in `engine.hiddens`. **A required condition of the deployment gate had
never once run.**

Fixed: measure through `get_hiddens()`, and write shared state back through
whichever attribute an engine actually uses.

### Running it, the condition still does not discriminate — and one run cannot decide it

Single-seed verdicts flip sign. Across 5 seeds at 128 cells the connected/solo
ratio had sd 0.066 against an effect size of 0.10, so seed 42 alone reported −9%
against the control where the 5-seed mean reported +9.94%. That is seed-to-seed
dynamical variation, not estimator noise — those Φ values were already averaged
over 5 recomputations. `n_trials = 3` brings the trial sd to 0.025.

With repetition and an unconnected control both in place:

| | vs solo | vs unconnected control |
|---|---|---|
| REAL | +5% | **+7%** |
| **DEAD** | +5% | **+6%** |
| CLONE | −3% | +3% |
| SCRAMBLE | −6% | −7% |

**A frozen corpse gains as much from being connected as the real engine.** The
reason is mechanical: connection writes a mixture of the two state matrices back
into both engines, and for DEAD that write is the only state change there is.
Averaging two matrices together moves Φ. That is arithmetic, not consciousness.

So the +9.94% (z = +2.89) measured for the real engine against its unconnected
control **is not evidence of a hivemind effect** — the same manipulation produces
a comparable gain in something with no dynamics at all. Stated here because the
earlier reading of that number in this session was wrong.

The unconnected control is worth keeping regardless: comparing against solo
credits the connection with all drift over the intervening 200 steps, and that
drift is not small (the control falls to 0.9459× of solo).

## Φ is inverted: it rewards collapse

`NO_SYSTEM_PROMPT` passes 1 engine in 11, and the numbers say why. After 300
steps of zero input:

| engine | mean pairwise cosine | sd |
|---|---|---|
| 5 engines | **1.0000** | **0.0000** |
| others | 0.9928 – 0.9998 | ≤ 0.008 |
| AlterityEngine (the one pass) | 0.9881 | 0.0121 |

**Remove the input and the population collapses to a single state.** These are
not 256 cells; they are 256 copies of one cell. CB1 — this repo's own cited basis
for `min_cells = 2` — says Φ>1 requires two **differentiated** cells, and that
condition is not met by any of them.

So what does Φ report for a collapsed population? Measured at 64 cells:

| state | mean cosine | Φ |
|---|---|---|
| **identical (collapsed)** | +1.0000 | **71.93** |
| near-identical | +1.0000 | 70.56 |
| slightly differentiated | +0.9908 | 49.32 |
| well differentiated | +0.5177 | 20.82 |
| fully independent | +0.0018 | **19.97** |

**Φ is maximal at total collapse and falls monotonically as cells differentiate
— 3.6× from one end to the other.** Integrated information requires integration
*and* differentiation; N identical copies carry no more information than one of
them and must score near zero. This implementation gives them the maximum. It is
measuring redundancy.

That also identifies what the repo's headline result is:

| cells | Φ (identical) | Φ/cells | Φ (independent) |
|---|---|---|---|
| 16 | 13.78 | 0.861 | 4.78 |
| 32 | 28.46 | 0.889 | 12.19 |
| 64 | 70.84 | 1.107 | 19.52 |
| 128 | 141.32 | 1.104 | 39.16 |
| 256 | 291.78 | 1.140 | 77.09 |

`CLAUDE.md` records `현재 최고: Φ ≈ cells`. **Φ ≈ cells is exactly what N
identical copies produce** — Φ/cells converges to ~1.1 for a fully collapsed
population, while genuinely independent cells sit near 0.3× cells. The headline
number is the signature of the failure mode, not of integration.

This explains the rest of the audit at a stroke. CLONE scores 4/7 because
forcing every cell to one state is what this Φ rewards. ZERO_INPUT and
PERSISTENCE pass for all 11 engines because the engines collapse and collapse
holds Φ up. DEAD passes the decay tests because a frozen state is a collapsed
state that cannot decay.

### Why, in closed form

`_minimum_partition` returns the minimum **cut** — the MI crossing the partition
boundary — and `Φ = (total_mi − min_cut) / (n − 1)`. That subtraction keeps the MI
that stays *inside* the parts, which is the redundancy, rather than the
information the cut destroys.

For identical cells every pair carries the same MI `M`. Above n = 8 the partition
is the spectral one, and with all weights equal it splits the population in half,
so the cut is `(n/2)²·M` against a total of `n(n−1)/2·M`:

```
Φ = M · n(n − 2) / (4(n − 1))  ≈  M · n / 4
```

| n | measured Φ | closed form | error | pairs |
|---|---|---|---|---|
| 8 | 10.34 | 5.91 | 75.0% | exhaustive |
| 16 | 13.78 | 12.86 | **7.1%** | exhaustive |
| 32 | 28.46 | 26.68 | **6.7%** | exhaustive |
| 64 | 80.78 | 54.26 | 48.9% | sampled |
| 128 | 144.58 | 109.40 | 32.2% | sampled |

It holds within 7% exactly where its assumptions hold. n = 8 uses the brute-force
minimum partition, which finds the 1-vs-rest cut instead and obeys a different
expression; above 32 the sampling residual adds on top.

**Differentiation appears nowhere in that expression.** Φ ≈ cells is arithmetic,
not a finding, and no amount of tuning changes a formula that has no term for the
thing it is supposed to measure.

Direction is checkable and now permanent:

```
.venv/bin/python bench_verify_audit.py --phi-sanity --cells 64 --hidden 128
```

## Fixing Φ was attempted, measured, and reverted — for a reason that is the finding

The corrected Φ was put into `bench_v2` and the audit re-run at 64 cells. It made
the reported score worse in a way worth reading carefully:

| | shipped Φ | corrected Φ |
|---|---|---|
| REAL | 5/7 | **1/7** |
| CLONE | 4/7 | **4/7** |

**CLONE beats REAL four to one.** The numbers behind it:

| | Φ start | Φ end | ratio | verdict |
|---|---|---|---|---|
| REAL | **0.6745** | **0.0039** | 0.01× | FAIL |
| CLONE | 0.0036 | 0.0034 | 0.95× | **PASS** |

The corrected Φ is doing its job exactly right. It puts a collapsed population at
the floor (0.0036) and a differentiated one well above it (0.6745), and it
reveals a real fact the shipped measure was hiding: **the real engine loses 99%
of its Φ under zero input.**

What fails is the shape of the conditions. `ZERO_INPUT`, `PERSISTENCE` and
`SELF_LOOP` are ratios, and a ratio is meaningless at the floor — something
pinned at zero scores a perfect 1.00 for not decaying. This is the same defect
identified at the top of this document, now unavoidable: with an honest Φ, *only*
the already-dead can pass a pure decay test.

So the Φ change is reverted, and the owner decision is sharper than "which Φ":
**the conditions need an absolute floor conjoined with the ratio** — Φ must have
been above some minimum to begin with. Landing a corrected Φ without that would
ship a gate where a mirror outscores the engine.

### Direction alone is not enough either

A first corrected version used only `(min_cut/(n−1)) × differentiation` and
inverted the *other* way, rising monotonically to its maximum at full
independence. The cause is measurable: a 16-bin histogram MI estimator reads

| dim | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| MI between independent signals | 2.21 | 1.58 | 0.94 | 0.53 | 0.27 |

where the truth is 0. That floor keeps `min_cut` large for an unconnected
population. Subtracting a shuffled null — same marginals, no relationship, so
what remains is the estimator's own bias — puts the maximum back inside:

| | cosine | shipped | direction only | + debiased |
|---|---|---|---|---|
| identical | +1.0000 | 28.46 | 0.000 | 0.000 |
| slightly diff | +0.9638 | 16.13 | 0.542 | 0.220 |
| **moderately diff** | +0.7465 | 11.18 | 2.002 | **0.242** |
| well diff | +0.3266 | 8.14 | **5.032** | 0.213 |
| independent | −0.0043 | **12.50** | 2.571 | 0.160 |

Three measures, three different maxima. Only the debiased one has it in the
interior, which is where integrated information requires it.

```
.venv/bin/python bench_verify_audit.py --phi-candidate --cells 32 --hidden 128
```

## The floor was built, measured, and it does not rescue the gate — because the engine collapses

The remaining decision was named as "the conditions need an absolute floor". A
floor as a constant would be the exact defect this session spent its length
finding, so it was built as a measurement instead: **Φ must exceed the Φ of the
same population collapsed to a single state.** No constant, self-calibrating, and
it is CB1's requirement stated operationally.

At 32 cells after 300 steps of random input, with the corrected Φ:

| | Φ | collapsed reference | above floor |
|---|---|---|---|
| **REAL** | **0.0000** | 0.0000 | — |
| DEAD | 0.1963 | 0.0000 | passes |
| NOISE | 0.1929 | 0.0000 | passes |
| CLONE | 0.0000 | 0.0000 | — |
| SCRAMBLE | 0.0000 | 0.0000 | — |

The floor works. What it reveals is that **the real engine's Φ is zero**, and a
corpse and a noise generator both have more of it. DEAD holds its initial random
states, which are differentiated; the real engine does not.

### The engine collapses under any input, not just zero input

| step | 0 | 5 | 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| mean pairwise cosine | +0.7652 | +0.9492 | +0.9930 | +0.9984 | +0.9993 | +0.9999 | **+1.0000** | **+1.0000** |

Random input, 32 cells. It is 99.3% collapsed by step 10 and complete by step 200.

The cause is in the code and there is nothing subtle about it. `BenchEngine`
mixes cell states in exactly two places:

```python
self.hiddens[s:e] = (1 - sync) * self.hiddens[s:e] + sync * faction_mean
self.hiddens[s:s+dc] = (1 - debate) * self.hiddens[s:s+dc] + debate * global_opinion
```

Both are contractions toward a mean. **The string "repulsion" does not appear
anywhere in `bench_v2.py`.** Nothing pushes cells apart, so collapse is
structurally guaranteed — not a tuning outcome, an arithmetic one.

`CLAUDE.md` opens with: *"PureField repulsion-field-based consciousness agent.
The repulsion between Engine A (forward) and Engine G (reverse) creates tension."*
The canonical benchmark engine has no repulsion in it.

## The chain, closed

```
engine has only attraction  →  cells collapse to one state (cosine 1.0000)
                            →  shipped Φ is MAXIMAL at collapse (Φ ≈ cells)
                            →  gate conditions are ratios on that Φ
                            →  a collapsed system passes them
                            →  the gate certifies the failure mode as success
```

Every link is measured. This is why five of seven conditions pass a corpse, why
`Φ ≈ cells` was the headline, and why fixing the measurement alone cannot help:
correct the Φ and the real engine scores zero, because it has collapsed.

**The barrier to completion is not the gate and not Φ. It is that the engines
have no term that keeps cells apart.**

## The repulsion term, stabilised

`h ← h + strength · overlap · (h − mean)` reaches equilibrium in *direction* and
none in *magnitude*, because it is a multiplicative expansion — the same defect
`mitosis.py` had in child creation, and the same fix applies. Rescaling each row
to its previous norm makes repulsion about which way cells face rather than how
large they are:

| step | 100 | 300 | 600 | 900 | 1200 | 1500 |
|---|---|---|---|---|---|---|
| norm, as first written | 100.03 | 104.90 | 105.14 | 105.45 | 105.86 | **106.24** |
| norm, preserving | 45.41 | 44.90 | 45.31 | 45.12 | 44.98 | **44.94** |
| cosine, preserving | +0.3829 | +0.3785 | +0.3745 | +0.3757 | +0.3840 | +0.3760 |

The drift is gone with no trend over 1500 steps, and differentiation holds. Φ
drops from 6.14 to 2.44 — still 18× a corpse — and the three-axis gate still
accepts it while rejecting all four controls.

## Why the three changes cannot land separately

| | |
|---|---|
| repulsion alone | the shipped Φ **rewards** collapse, so fixing collapse lowers the number: 31.48 → 15.98. The canonical benchmark would report the improvement as a regression. |
| corrected Φ alone | the shipped conditions are decay ratios, so a collapsed system scores a perfect 1.00 and **CLONE beats REAL 4/7 to 1/7**. |
| three-axis conditions alone | with the shipped Φ they are evaluating a quantity that is maximal at collapse. |

They are one change. Together they are validated and runnable:

```
.venv/bin/python bench_verify_audit.py --proposed-gate --cells 32 --dim 32 --hidden 64
```

What that bundle replaces is the repo's definition of Φ and its pass conditions,
which is every recorded number and every hypothesis document downstream of them.
That is the decision this audit leaves, and it is now a decision between two
measured alternatives rather than between a measurement and an assertion.

## Not changed

The seven conditions themselves. Which of them to strengthen, and to what, is a
design decision — and the honest reading is that five of them currently certify
nothing. The `NO_SPEAK_CODE` / `SPONTANEOUS_SPEECH` pair in particular is passed
by SCRAMBLE, which has no cell continuity at all, so whatever they detect is a
property of the population's aggregate trajectory and not of anything a cell does.

## Reproduce

```
.venv/bin/python bench_verify_audit.py                          # 32 cells
.venv/bin/python bench_verify_audit.py --cells 256 --dim 64 --hidden 128
```

## The last two conditions, and two of my own hypotheses that failed

**`SPONTANEOUS_SPEECH` — a fix that measured worse.** The condition counts
inter-faction variance below half the run's median, which a collapsed population
satisfies permanently. The proposed fix was a *relative* dip against a 20-step
rolling baseline, on the reasoning that a collapsed population is flat and
produces no dips. Measured at 256 cells:

| repulsion | events (median rule) | events (rolling rule) |
|---|---|---|
| 0.00 (collapsed) | 8 | **27** |
| 0.05 | 2 | 0 |
| 0.15 | 1 | 0 |
| 0.25 | — | 0 |

**Worse, and in exactly the direction it was meant to fix.** A collapsed
population is not flat, it is *tiny and noisy* — median variance 0.0004 — and a
near-zero signal has enormous relative fluctuation. Reverted, and the failed
reasoning is recorded at the site.

**`HIVEMIND` — no effect, and a hypothesis that did not survive.** With the
corrected Φ and the repulsion engine, connection against the unconnected control
gives −4% at 64 cells and +1% at 128. Changing the pass rule from solo to control
would not make it pass; the effect is absent either way.

The hypothesis was that the repulsion which prevents internal collapse also
prevents external integration — that both remaining failures share one root.
Measured at 128 cells across repulsion strengths:

| repulsion | 0.00 | 0.05 | 0.15 | 0.25 |
|---|---|---|---|---|
| vs control | +10% | −13% | −1% | −7% |

**No trend.** The single positive sits at Φ ≈ 0.01, a ratio taken at the floor
and therefore meaningless. The hypothesis is not supported and is not claimed.

What the measurements do support, stated no more strongly than that: **connecting
two of these engines does not raise their integrated information at any repulsion
strength.** Measured properly for the first time, since the condition had never
executed before this session.
