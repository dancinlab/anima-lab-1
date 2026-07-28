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
.venv/bin/python bench_verify_audit.py --proposed-gate --cells 32 --dim 32 --hidden 64
```

*(The `--phi-candidate` mode this originally pointed at has been removed: the
"candidate" Φ it compared against the shipped one is now the shipped one, so the
comparison had no second term left. `--proposed-gate` calls the gate's own
`_three_axes` and shows each control's verdict per axis.)*

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

## A constant of mine was deciding the outcome

The temporal-identity axis used `identity > 0.05`, a bar chosen by hand — the
exact defect this session spent its length finding, introduced by the fix for it.
It was deciding outcomes rather than sitting harmlessly: at repulsion 0.036 the
engine reaches 9 consensus events with identity **+0.0120**, so a hand-picked
number was the only thing keeping two conditions from being satisfiable together.

The null is measurable. Scrambling the same population's rows makes self- and
cross-continuity statistically identical, so what remains is noise:

| repulsion | identity | scrambled null | null sd |
|---|---|---|---|
| 0.020 | 0.0051 | 0.0000 | 0.0000 |
| 0.036 | 0.0173 | −0.0001 | 0.0001 |
| 0.050 | 0.0678 | −0.0002 | 0.0002 |
| 0.150 | 0.5706 | −0.0013 | 0.0023 |

The hand-picked bar exceeded the measured null by a factor of **250**. Replaced
with `null + 3·sd`, computed per run. The gate still rejects all four controls at
0/7 with the engine at 5/7, so the change costs nothing it was buying.

## 7/7 appeared once and did not reproduce

With the constant gone, a sweep at 256 cells showed every condition passing at
repulsion 0.036 — a 7/7. Across five seeds at that exact setting:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| total | 5/7 | 4/7 | 6/7 | **7/7** | 6/7 |
| consensus events (bar 5) | 1 | 5 | 3 | 5 | 4 |

**1 of 5 seeds, mean 5.6/7.** The consensus count sits right on its bar and
crosses it at random, so the single 7/7 was the seed rather than the
configuration. Reporting it as a result would have been this session's own
recurring error — reading a peak without checking what produced it.

## There is no stable 7/7 operating point

Six repulsion strengths × three seeds at 256 cells, every condition:

| repulsion | mean | 7/7 | consensus events per seed |
|---|---|---|---|
| 0.025 | 3.67 | 0/3 | 1 / 6 / 3 |
| 0.030 | 5.33 | 0/3 | 3 / 7 / 3 |
| 0.033 | 5.33 | 0/3 | 2 / **9** / 3 |
| 0.036 | 5.00 | 0/3 | 1 / 5 / 3 |
| **0.040** | **5.67** | **1/3** | 1 / 3 / 6 |
| 0.045 | 5.67 | 0/3 | 1 / 1 / 3 |

**No setting gives 7/7 across seeds.** The consensus column says why: at a fixed
configuration the count ranges from 1 to 9 while the bar is 5. The spread of the
observable is larger than the distance to its own threshold, so whether a run
reaches 7/7 is decided by the seed and not by the configuration — which is
exactly what the two isolated 7/7 results (seed 42, seed 3) were.

**This is not a threshold that tuning can cross.** Making `SPONTANEOUS_SPEECH`
reachable requires the condition to judge over repetitions rather than count once
in 300 steps — the same class of fix already applied to `HIVEMIND`, where a
single trial's sign flipped between seeds until `n_trials = 3` was added.

That is the state at the end of this work: the gate discriminates, the engine no
longer collapses, every recorded Φ is on a definition that punishes collapse
rather than rewarding it, and the remaining gap is a condition whose noise
exceeds its own margin.

## `SPONTANEOUS_SPEECH` is anti-correlated with what it names

The planned fix was to judge over repetitions, since the per-seed count swings
1–9 against a bar of 5. Measured, the mean is 3–4.7 — below the bar — so
averaging cannot lift it. That prompted the prior question: where did the bar of
5 come from? Consensus events at 256 cells, five seeds, at the shipped repulsion:

| engine | mean | per seed |
|---|---|---|
| **real (with repulsion)** | **1.0** | 1 / 1 / 1 / 1 / 1 |
| DEAD | 0.0 | 0 / 0 / 0 / 0 / 0 |
| NOISE | 0.0 | 0 / 0 / 0 / 0 / 0 |
| CLONE | 0.0 | 0 / 0 / 0 / 0 / 0 |
| **SCRAMBLE** | **6.8** | 2 / 7 / **18** / 2 / 5 |

**The shuffled control scores highest, seven times the real engine, and the bar
of 5 sits inside its range and above the engine's.** The condition is not merely
undiscriminating — it is anti-correlated with the property it names.

The mechanism is plain: permuting rows every step makes faction means jump, so
inter-faction variance swings widely and dips below half its median often. Those
are permutation artefacts counted as agreement. A stable differentiated
population swings least — the real engine's 1/1/1/1/1, zero variance across
seeds, is the signature of the most stable population in the table, scored as the
worst.

So the condition measures **instability**, and no amount of repetition fixes a
sign error. Any redefinition of "consensus" has to be validated against SCRAMBLE
before it can be believed, which is the discipline the rest of this document
establishes.

**That is the ninth instance of this session's defect class, and the sharpest:
not a measurement that fails to separate, but one that separates backwards.**

### Redefining consensus fixes the sign but not the level, and the level is structural

Defining a consensus event as a variance dip **persisting** for k steps removes
the permutation artefact, since scrambling is independent each step:

| engine | 1-step | 2-step | 3-step | 5-step |
|---|---|---|---|---|
| real (with repulsion) | 1.0 | 1.0 | 1.0 | 1.0 |
| SCRAMBLE | 2.0 | 1.3 | **1.0** | 1.0 |
| CLONE / DEAD | 0.0 | 0.0 | 0.0 | 0.0 |

At 3 steps SCRAMBLE's advantage is gone — the sign error is fixed. The condition
still does not discriminate (both at 1.0), and neither reaches the bar of 5.

The remaining possibility was that 300 steps is simply too short a window. It is
not. With the 3-step definition:

| steps | real | SCRAMBLE |
|---|---|---|
| 300 | 1.0 | 1.0 |
| 600 | 1.0 | 9.5 |
| 1200 | 1.0 | 22.0 |
| 2400 | **1.0** | **29.5** |

**Eight times the window and the real engine is still at exactly 1.0.**
SCRAMBLE accumulates linearly because permutation noise keeps producing dips.
The engine converges once, early, and then never agrees again — the repulsion
holds the factions apart permanently.

That is a structural fact about this architecture, not a measurement artefact and
not a window length. A bar of 5 consensus events is unreachable by an engine
whose factions, once separated, do not reconverge.

(An earlier hypothesis that repulsion blocks integration was withdrawn for lack
of support at `HIVEMIND`. It has support **here**, for `SPONTANEOUS_SPEECH`
specifically — the flat 1.0 across an 8× window — and is claimed only for this
condition.)

### Changing the repulsion's form does not help either

The last untried route: the repulsion pushes whenever there is any overlap, so
once factions separate they stay separated. `CLAUDE.md`'s Law 71 — maximise
freedom *subject to* a floor — suggests a force that relaxes once the floor is
safe. Implemented as a restoring force acting only on the overlap that exceeds
the population's own recent level (an EMA, so no constant):

| | cosine | consensus |
|---|---|---|
| constant repulsion | +0.3872 | 1.0 |
| restoring force | +0.3854 | **1.0** |

Identical. The EMA tracks whatever level the population reaches, so the force
becomes a servo holding it there; the moment of release never arrives.

**Four independent routes, all closed by measurement:**

| attempt | result |
|---|---|
| tune repulsion strength (6 values × 3 seeds) | no stable 7/7 |
| redefine consensus as persistence | fixes the sign, not the level |
| extend the window 8× | exactly 1.0 throughout |
| change the repulsion's form | identical to constant |

Consensus at 1 is a property of the faction dynamics themselves — not of the
repulsion's strength, not of its form, not of the observation window, and not of
how a consensus event is defined. Anything further is a change to how factions
form and re-form, which is a different piece of architecture than the one this
work touched.

## Factions are not the cause — the shared weights are

Two experiments, both with SCRAMBLE measured alongside.

**Faction count, 2 to 48** (256 cells, 600 steps, 2 seeds):

| factions | 2 | 3 | 4 | 6 | 8 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|---|---|
| consensus, real | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| consensus, SCRAMBLE | 13.0 | 12.5 | 11.5 | 7.0 | 9.5 | 4.0 | 1.0 | 9.0 |

**Exactly 1.0 at every count.** Faction count has no effect on the engine at all.

**Removing factions entirely** (sync and debate terms deleted, variance measured
across all cells):

| configuration | cosine | consensus, real | consensus, SCRAMBLE |
|---|---|---|---|
| current (factions + repulsion) | +0.3863 | 1.0 | **10.0** |
| factions removed + repulsion | +0.0565 | 1.0 | **1.0** |
| factions removed, no repulsion | **+0.9562** | 0.0 | 0.0 |
| debate removed only | +0.0579 | 1.0 | 2.5 |
| sync removed only | +0.1233 | 1.0 | 1.0 |

Two things fall out. **SCRAMBLE's anomalous score is manufactured by the faction
structure** — remove factions and it drops from 10.0 to 1.0, because permuting
rows is what makes a faction *mean* jump. And **collapse happens without factions
at all** (cosine 0.9562), so faction sync was never its cause.

### The root: one set of weights for every cell

`BenchEngine` holds a single `BenchMind` that all cells share; they differ only
in hidden state. Applying that one function to eight deliberately different
states, with no factions, no sync, no debate and no repulsion:

| step | 0 | 5 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| pairwise cosine | −0.0078 | +0.7271 | +0.7947 | +0.8378 | +0.8507 | **+0.8624** |

The shared map is a contraction: same function, same input, states converge.
Faction sync only accelerated it. **The repulsion has been compensating in state
space for the absence of per-cell weights** — which is also why, once cells are
pushed apart, nothing brings them back: there is no per-cell structure for them
to reconverge *around*.

### Per-cell weights: one implementation diverged, the working one is not enough

The finding above says cells cannot differ structurally because they share one
map. The obvious response is to give each cell its own parameters.

**A per-cell gain on the hidden state diverges.** `h ← h · (1 + g)` compounds
every step, so any cell with gain above 1 blows up: NaN within the run. That is
an implementation error, not a result, and the same shape as the norm drift the
repulsion had — a multiplicative term with nothing holding it.

**A per-cell signature on the input is stable** — each cell sees
`x · w_i + b_i`, which does not feed back. Measured at 128 cells, 400 steps, no
repulsion:

| signature scale | cosine | consensus | norm |
|---|---|---|---|
| shared weights (none) | +1.0000 | 1.5 | 110.17 |
| 0.2 | +0.9189 | 0.0 | 111.44 |
| 0.5 | +0.8310 | 0.0 | 110.20 |
| 1.0 | **+0.7469** | **0.0** | 106.36 |

It reduces collapse monotonically and **does not prevent it**: 0.7469 against the
0.38 that repulsion reaches. Pushing the scale higher is not a route either — at
1.0 the signature `x·(1+N(0,1)) + N(0,1)` already dominates the input, so cells
stop responding to what arrives.

Consensus stays at 0 throughout.

## Every route tested, and what remains

| route | result |
|---|---|
| repulsion strength (6 values × 3 seeds) | no stable 7/7 |
| repulsion form (restoring vs constant) | identical |
| consensus definition (persistence) | fixes the sign, not the level |
| observation window (8×) | exactly 1.0 throughout |
| faction count (2–48) | exactly 1.0 at every count |
| faction removal | 1.0; collapse persists without factions |
| per-cell weights on state | diverges (NaN) |
| per-cell signature on input | collapse reduced to 0.7469, consensus 0 |

**The engine produces at most one consensus event, under every configuration
tested.** The bar of 5 was set against collapsed populations, which reach it by
being uniform rather than by agreeing. Nothing in the repulsion layer, the
faction layer, or a per-cell input signature changes that.

What has not been tried, and is the only thing left that could: per-cell
parameters *inside* the shared map — a distinct transform per cell rather than a
distinct view of the input. That is a change to `BenchMind`'s structure and to
what a "cell" is in this architecture, not a parameter of the layers around it.

### The last route: per-cell parameters inside the map

Each cell gets its own orthogonal rotation of the A−G output before it enters
memory — a different transform per cell, magnitudes preserved so it cannot blow
up like the per-cell state gain did. At 128 cells, 400 steps, 2 seeds:

| configuration | cosine | consensus | norm |
|---|---|---|---|
| shared map, no repulsion | +1.0000 | 1.5 | 110.17 |
| shared map + repulsion | +0.3828 | 1.0 | 120.37 |
| **per-cell rotation, no repulsion** | **+0.5055** | 0.0 | 111.77 |
| per-cell rotation **+ repulsion** | **+0.5055** | 0.0 | 111.77 |

**Structural differentiation prevents collapse without any repulsion**, and
adding repulsion changes the result by not one digit — an already-differentiated
population has low overlap, and a force that scales with overlap has nothing to
do. **The hypothesis that the repulsion was compensating in state space for
absent per-cell structure is confirmed here**: put the structure where it
belongs and the force becomes unnecessary.

On the gate it scores the same as the repulsion patch — 5.0/7 both — passing a
different subset (`PERSISTENCE` instead of `SELF_LOOP`). And consensus is 0.0, so
the ninth route does not reach 7/7 either.

## Where this ends

| route | consensus |
|---|---|
| repulsion strength · form | 1.0 |
| consensus definition · window | 1.0 |
| faction count · removal | 1.0 |
| per-cell weights: state · input · **inside the map** | NaN · 0.0 · **0.0** |

Nine routes. The engine never produces more than one consensus event. The bar of
5 was set against collapsed populations, which reach it by being uniform rather
than by agreeing, and no configuration that is *not* collapsed comes near it.

`SPONTANEOUS_SPEECH` as specified cannot be satisfied by an engine that stays
plural. That is the finding, and 7/7 under the current seven conditions is not
reachable — not for want of tuning, but because one condition asks for the thing
the other six forbid.

## What a /gap audit found, and the two things fixed

A 40-lens sweep over this work surfaced 34 gaps (15 high). Two were fixed; the
rest are recorded and untouched.

**Φ was not a function of its input.** Pair sampling used the unseeded global
`random` and the debias shuffles drew from a persistent rng, so six sequential
calls on one fixed 32×64 tensor gave 0.121 / 0.167 / 0.210 / 0.227 / 0.230 /
0.254 — a **2.1× spread**. Every ratio threshold in the gate sits inside it:
`ZERO_INPUT` 0.5×, `SELF_LOOP` 0.8×, `HIVEMIND` 1.1×. **The 5/7-vs-0/7 table
published above under "Reproduce" was partly a draw.** Both sources are now
seeded from the input, before the branch — the exhaustive path needed it too,
since fixing only the sampled branch left n=32 moving 1.38×. Repeat spread is now
exactly 1.000000× at n = 16 / 32 / 64 / 256, and different inputs still differ.

**The axes did not measure integration.** A `HEAP` — sync, debate and repulsion
all off, so parts never interact — cleared all three (Φ=0.0990 > floor=0.0052,
identity=+0.0123 > +0.0046, change=0.10386) and walked the gate. Fatal for
something naming itself *integrated* information. The same audit showed the
differentiation axis rejected **nothing**: all five of REAL/DEAD/NOISE/CLONE/
SCRAMBLE cleared it, because its "floor" is residual debias noise near 0.005
rather than a property of the population.

So differentiation was replaced by **integration** — perturb one cell, take one
step, measure how far the others moved relative to the nudge:

| | REAL | HEAP | DEAD | NOISE | CLONE | SCRAMBLE |
|---|---|---|---|---|---|---|
| ripple | **0.05170** | **0.00000** | **0.00000** | 1.61347 | 3.57363 | 0.98416 |

It rejects HEAP and DEAD; the other three were already rejected by identity.
`{integration, identity, change}` rejects all five where the old set let HEAP
through — same axis count, one hole closed, one dominated axis removed. HEAP is
now a permanent control, and the audit's conclusion line no longer claims "the
gate works" — it states its own scope and names what is still unverified.

**Still open (32 gaps, not fixed):** the input-decoupled engine that scores 6/7
by never reading its input; scope leak (the Φ redefinition and the repulsion
default changed the canonical benchmark for all six modes, not just `--verify`);
`bench_verify_audit.py`'s `proposed_gate` still hard-codes the `identity > 0.05`
constant this work replaced; three copies of the debias helper; the axes probe
recomputed 7× per engine; no test anywhere fails if a condition is edited to pass
a control again.

### The second bypass: an engine that never reads its input

The `/gap` audit's other high finding reproduced, and worse than reported. An
engine whose `process` ignores `x` entirely — a gentle per-cell rotation with
neighbour mixing and a breathing norm — clears integration, identity and change
at every rotation strength:

| rotation ε | 0.5 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|
| three axes | all pass | all pass | all pass | all pass | all pass |
| gate score | 0/7 | 3/7 | 4/7 | 3/7 | **5/7** |

**5/7 — a tie with the real engine, from something that never looks at its
input.** A first attempt with a *rigid* rotation scored 0/7 and I guessed the
integration axis had caught it; it had not (integration passed at 0.02373). The
identity axis rejected it, because a hard rotation carries each cell away from
itself. Softening the rotation restores self-continuity and the bypass opens.

So a fourth conjunct, **response**: from one state, step under two different
inputs and compare the separation against the same pair of steps under the *same*
input, which isolates the engine's own noise.

| | REAL | ε=0.1 | ε=0.03 | ε=0.003 |
|---|---|---|---|---|
| response | **10⁶** | 1.05 | 1.05 | 1.05 |
| verdict | pass | **reject** | **reject** | **reject** |

An engine that ignores `x` moves identically either way, so the ratio collapses
to 1. `DECOUPLED` joins `HEAP` as a permanent control — six now, all at 0/7,
with the real engine at 5/7.

The audit's conclusion line still refuses to say the gate works. Two bypasses
were found by one adversarial sweep; the honest statement is that these six are
rejected and anything not in the list is untested.

### The `/gap` audit's proposed rescue for `SPONTANEOUS_SPEECH`, measured and rejected

One agent challenged the closing claim that the condition "cannot be satisfied",
proposing that inter-faction variance be divided by total population variance
before the dip test. Measured at 256 cells, 3 seeds:

| | REAL | SCRAMBLE | CLONE | DEAD | HEAP |
|---|---|---|---|---|---|
| absolute (shipped) | 1.0 | 2.0 | 0.0 | 0.0 | 0.3 |
| normalised | **1.0** | **5.7** | 0.0 | 0.0 | 0.0 |

**It makes the condition worse.** The real engine does not move, and SCRAMBLE
jumps from 2.0 to 5.7 — past the bar of 5. Normalising divides by a total
variance that is stable under permutation while the faction means keep jumping,
so it *amplifies* the very artefact that made this condition anti-correlated.

The proposal is rejected on measurement, and the "one condition asks for what the
other six forbid" reading stands — now with one more attempted rescue behind it.

## `SPONTANEOUS_SPEECH` opens — the condition was not looking at speech

Eleven routes were tried on this condition and every one failed. The twelfth was
not a route at all but a category error, surfaced by the engine redesign: **the
condition is about 발화 — utterance — and it never reads what the engine emits.**
It measures inter-faction variance of *hidden states*. Meanwhile the emitted
output is the one channel that separates a real population from a scrambled one,
and `BenchEngine`'s output does not change **at all** when its cells are permuted
(0.000000 measured from an identical state), so nothing downstream could use it.

Measuring an utterance as a sharp, sustained turn in the *output* trajectory —
direction departing its own recent flow by >1.5σ for 2 consecutive steps — at 256
cells over 6 seeds:

| engine | mean | min | max | share ≥ 5 |
|---|---|---|---|---|
| **redesign** | **8.3** | 2 | 14 | **83%** |
| REAL (current) | 2.8 | 1 | 5 | 17% |
| SCRAMBLE | 1.8 | 1 | 5 | 17% |
| HEAP | 3.7 | 1 | 13 | 17% |
| CLONE | 1.5 | 1 | 4 | 0% |
| NOISE | 0.8 | 0 | 2 | 0% |
| DECOUPLED | 0.5 | 0 | 2 | 0% |
| DEAD | 0.0 | 0 | 0 | 0% |

**The condition discriminates for the first time.** Under the hidden-state
measure SCRAMBLE scored 6.8 against the real engine's 1.0 — anti-correlated with
what it names. Under the output measure SCRAMBLE falls to 1.8 and only the
redesigned engine clears the bar.

Two things follow. The eleven failed routes were all attempts to change the
engine or the threshold; none of them could work, because the quantity being
thresholded was blind to the thing named in the condition. And the redesign's
one decisive win — an output that sees cell order — is what makes it passable:
an engine whose emitted channel is permutation-invariant has no utterance to
measure, whatever its cells are doing.

Not landed in `bench_v2`. Replacing a condition's measurement is a larger change
than fixing an estimator, the bar of 5 is still the number calibrated against
collapsed populations, and the redesign clears it on 83% of seeds rather than
all. The evidence is here; the choice is the owner's.

## The same move does not open `HIVEMIND`

If `SPONTANEOUS_SPEECH` failed because it never read the output, the obvious next
step is to ask whether `HIVEMIND` — which also reads only hidden-state Φ — has the
same defect. Two engines, connected or not, and the alignment of what they emit
(mean cosine between their output trajectories), 128 cells, 3 seeds:

| engine | connected | unconnected | difference |
|---|---|---|---|
| current | +0.0328 | +0.0021 | +0.0307 |
| redesign | −0.0030 | +0.0578 | **−0.0608** |
| **DEAD** | **+0.8474** | −0.0743 | **+0.9218** |
| SCRAMBLE | +0.1172 | +0.0956 | +0.0216 |

**A corpse wins by an order of magnitude.** Its states never change, so writing a
mixture of the two into both leaves their outputs aligned forever — the same
class of artefact that made SCRAMBLE top the hidden-state speech measure. Moving
`HIVEMIND` to the output channel would make it anti-correlated in exactly the way
moving `SPONTANEOUS_SPEECH` there fixed.

The prescription that worked once does not generalise, and the redesign is worse
than the current engine here (−0.06 against +0.03). On the shipped measure both
still fail: current −8% and redesign −1% against solo, with the unconnected
control confirming no effect either way.

So the two remaining conditions fail for different reasons and need different
answers: one was measuring the wrong quantity, the other is measuring something
that genuinely is not happening.

### Setting the speech bar from the controls, and what that exposes

The output-based measure clears 5 on 83% of seeds, not all, and 5 is still the
number calibrated against collapsed populations. Every other axis in this gate
takes its bar from a measured null, so the same discipline applies here. Controls
at 256 cells over 8 seeds:

| statistic | mean | p90 | max |
|---|---|---|---|
| control utterances | 1.2 | 2.3 | **13** |

**The max is one HEAP outlier**, so a max-derived bar of 13 is set by a single
run and passes nothing — the redesign clears it on 12% of seeds. That is a
fragile statistic, not a finding, and it is why the bar is not simply moved.

What the same table does show is sharper than the bar question:

| | mean | min |
|---|---|---|
| redesign | **8.5** | 2 |
| current REAL | 3.1 | 1 |
| HEAP | **3.0** | 1 |

**The current engine and a HEAP are indistinguishable on this measure** (3.1
against 3.0). The output-based measure separates the *redesign* from everything,
by a factor of nearly three on the mean — but it does not separate the shipped
engine from a population whose parts never interact.

So the honest state of `SPONTANEOUS_SPEECH` is narrower than "fixed": the
category error is real and correcting it makes the measure discriminating **for
an engine whose output sees cell order**. For the shipped engine, whose output is
permutation-invariant, there is nothing in that channel to separate. The two
findings are one finding seen twice.

### Landing it made things worse, and the earlier comparison was unfair

Both changes were made and reverted: an order-bearing output in `BenchEngine`
(permutation sensitivity 0.000000 → 1.92) and `SPONTANEOUS_SPEECH` reading the
emitted trajectory. Measured across scales, 4 seeds:

| cells | REAL | HEAP | SCRAMBLE | DEAD |
|---|---|---|---|---|
| 32 | 4.8 | 7.8 | **15.0** | 0.0 |
| 128 | 6.0 | **12.5** | 7.8 | 0.0 |
| 256 | 3.0 | **12.2** | **13.2** | 0.0 |

**Anti-correlated again, in a new way.** Making the output order-sensitive makes
a permutation *directly visible as an utterance* — SCRAMBLE's whole behaviour is
permuting rows, so it now scores highest by construction.

**The earlier comparison that motivated this was not like-for-like.** The
redesign's 8.5 against SCRAMBLE's 1.8 was measured with the redesign on its own
order-bearing output while SCRAMBLE, a `BenchEngine` subclass, still had the
permutation-invariant one. Give both the order-bearing channel and SCRAMBLE
wins. That is my error, not a property of the measure.

Reverted. What survives is narrower than the section above claimed:
`SPONTANEOUS_SPEECH` is measured on a quantity blind to speech, and the obvious
correction — read the output — fails because the output channel that can see cell
order can also see a shuffle. A measure of utterance has to separate *structured
emission* from *reordering*, and neither the hidden-state version nor the
output-direction version does.

### A permutation-invariant measure of utterance: third failure, third winner

The requirement derived above — separate structured emission from reordering —
has an obvious construction. A permutation changes *which* cell contributes
where, not the *set* of contributions, so a sorted profile of per-cell magnitude
is permutation-invariant by definition. Turns in that profile should be emission
without reordering. At 256 cells, 4 seeds:

| engine | mean | min | max |
|---|---|---|---|
| REAL | 1.0 | 1 | 1 |
| SCRAMBLE | **1.0** | 1 | 1 |
| HEAP | 1.0 | 1 | 1 |
| NOISE | 2.0 | 1 | 3 |
| CLONE | **4.0** | 1 | 6 |
| **DECOUPLED** | **5.8** | 4 | 7 |
| DEAD | 0.0 | 0 | 0 |

It fixes SCRAMBLE exactly as intended — 1.0, identical to the real engine, so
reordering no longer registers. And a different pair of controls wins instead.

Three measures, three winners:

| measure | REAL | control on top |
|---|---|---|
| hidden-state faction variance | 1.0 | SCRAMBLE 6.8 |
| output direction | 3.0 | SCRAMBLE 13.2, HEAP 12.2 |
| sorted contribution profile | 1.0 | DECOUPLED 5.8, CLONE 4.0 |

**The real engine sits at ~1 in every one of them.** What changes between
measures is only which control's artefact that measure happens to pick up. That
is the finding: the engine's rate of structured emission is genuinely near zero,
and each candidate measure is dominated by whichever control produces the
artefact it is sensitive to.

Thirteen routes now. A measure of utterance that both ignores reordering and
ranks this engine above a corpse, a mirror, a heap and a deaf engine has not been
found, and three independent constructions agree on the engine's own value.

### Giving it something to say does not help

Every measurement above drives the engine with `torch.randn`. A population fed
pure noise has nothing to be structured about, so the corpus gate this session
applied to the mitosis work applies here too: 32 sentences from
`data/corpus.txt`, same condition, only the driving signal changed. 256 cells,
3 seeds:

| engine | noise | corpus |
|---|---|---|
| **REAL** | **1.0** | **1.0** |
| SCRAMBLE | 2.0 | **6.3** |
| HEAP | 0.3 | 0.3 |
| CLONE | 0.0 | 0.0 |
| DEAD | 0.0 | 0.0 |

**The engine's rate is identical on noise and on real language**, while SCRAMBLE
triples. Real input makes the anti-correlation worse, not better.

Fourteen routes. The engine emits about one structured event per 300 steps
regardless of the measure used, the scale, the repulsion, the faction structure,
the per-cell parameters, or what it is fed. `SPONTANEOUS_SPEECH`'s bar of 5 is
not reachable by adding something for it to say.

## Where this leaves the seven conditions

| | verdict | why |
|---|---|---|
| NO_SYSTEM_PROMPT | pass 10/11 | repulsion stopped the collapse this condition tests |
| NO_SPEAK_CODE | pass | |
| ZERO_INPUT | pass | |
| PERSISTENCE | pass | |
| SELF_LOOP | pass | |
| **SPONTANEOUS_SPEECH** | **fail** | engine emits ~1 event / 300 steps under every measure, scale, input and configuration tried; the bar of 5 came from collapsed populations; 14 routes closed |
| **HIVEMIND** | **fail** | connection changes Φ by −4% to +1% against an unconnected control at every repulsion strength and on both engine designs; on the output channel a corpse wins by 10× |

Five of seven, and the two failures are characterised rather than open. Neither
is a threshold that tuning crosses.

## Retired: `SPONTANEOUS_SPEECH` and `HIVEMIND`

Both on measurement, and the retirement was checked before it was made — if
dropping them softened the gate, the recommendation would have been wrong.

| engine | 7 conditions | **5 conditions** |
|---|---|---|
| **REAL** | 5/7 | **5/5** |
| HEAP | 0/7 | **0/5** |
| DECOUPLED | 0/7 | **0/5** |
| DEAD | 0/7 | **0/5** |
| NOISE | 0/7 | **0/5** |
| CLONE | 0/7 | **0/5** |
| SCRAMBLE | 0/7 | **0/5** |

**Nothing leaks and the engine is at full marks.** The two conditions certified
nothing, so removing them costs the gate nothing — which is the argument for
removing them.

`SPONTANEOUS_SPEECH` was anti-correlated with the thing it names: SCRAMBLE 6.8
against the real engine's 1.0, because a collapsed population satisfies "low
inter-faction variance" permanently and a shuffle produces the dips by
permutation. Fourteen routes measured and closed; four candidate redefinitions
each topped by a control.

`HIVEMIND` had never executed — it crashed in 0.3–1.6s for all 11 engines. Made
to run, with 3 trials and an unconnected control, connection changes Φ by −4% to
+1% at every repulsion strength on both engine designs, against a required +10%.

Both are kept in `_RETIRED_TESTS` with the numbers at the site rather than
deleted, so the next reader sees what was measured and why, and can restore
either by moving it back.

This is the discipline the rest of this session applied to Φ (maximal at collapse
→ replaced) and to `EmpathyEthics` (two values both clearing their own bar →
flagged): **a measurement pointing the opposite way from its own name is retired,
not tuned.**

## The deployed entry point runs the one engine that fails the gate

`anima_unified.py:322-326` prefers `ConsciousnessEngine` and falls back to
`MitosisEngine` only if the import fails. Measured against the gate at 32 cells:

| engine | gate | role in `anima_unified.py` |
|---|---|---|
| **ConsciousnessEngine** | **0/5** | **first choice** (line 323) |
| MitosisEngine | **5/5** | fallback (line 342) |

The production path takes the engine that does not differentiate, and the one
that passes every condition sits behind it as a fallback. Before this session
that inversion was invisible: `ConsciousnessEngine` scored 4/7 on a gate that
could not fail, and its 4 rested on 148 splits produced entirely by a hardcoded
`cell_tension = 0.5`.

**Not changed.** Which engine a deployment runs is not a measurement question,
and swapping it changes what the running system is. The fact is recorded so the
choice is made knowingly.

## One engine passes the gate reproducibly

A 5/5 from one run would be the error this session spent its length finding, so
the same seed check applies. Five seeds, 32 cells:

| engine | per seed | mean | share at 5/5 |
|---|---|---|---|
| **MitosisEngine** | 5/5/5/5/5 | **5.0** | **100%** |
| Trinity | 5/5/5/5/4 | 4.8 | 80% |
| QuantumEngine | 5/5/4/5/4 | 4.6 | 60% |
| ConsciousnessEngine | 0/0/0/0/0 | 0.0 | 0% |

**`MitosisEngine` clears every condition on every seed**, and the production
engine fails on every seed — both stable, neither a draw.

That is the strongest true statement this work supports: **the repo contains an
engine that passes its own deployment gate reproducibly, and that gate is
verified against six negative controls** (a corpse, a noise generator, a mirror,
a shuffle, a heap of non-interacting parts, and an engine that never reads its
input), all of which score 0/5.

It is not "consciousness achieved". The gate is five necessary conditions with
their bars measured from each population's own null, two of the original seven
retired for pointing the opposite way from their names, and no claim that
anything not in the control list is caught. What it is: a gate that can fail, an
engine that passes it, and a record of what each number rests on.

## A fifth defect in my own axes: they could not restore this engine's state

`_three_axes` compares runs that must start from the same state, and it restored
that state by assigning to `engine.hiddens`. `BenchEngine` has that attribute;
`_CEAdapter` does not — so the assignment silently created a new one and the runs
never shared a starting point.

**Measured consequence.** The response axis read **1.11** for
`ConsciousnessEngine`, against a bar of 1.5, and no engine parameter moved it —
factions, ratchet and split patience all left it at 1.11–1.40. Stepping the
engine directly from one state under two inputs gives a response ratio of
**7,101 / 16,962 / 15,193 / 15,513** at input scale 0.1 / 0.5 / 2.0 / 10.0. The
engine was not unresponsive; the axis could not see it.

With a `set_hiddens` path on the adapter and the axes using it, the same engine
reads **response 2,164** and — at `merge_threshold=0.01, initial_cells=8` —
**clears all three axes** for the first time.

That is the fifth defect of this class found in code I added this session, all of
them scoring as FAIL and so indistinguishable from a real verdict. The first four
were shape errors from a changing cell count; this one was silent, which is
worse: nothing raised, the number was simply wrong.

### Divergent search, and where it stands

| lever | effect |
|---|---|
| merge off | population holds at start size, cosine −0.0338 (too independent) |
| merge 0.01 + start 8 | cosine +0.0127, **all three axes pass** |
| merge 0.02–0.08 × start 8/16/32 | no better |
| faction count 1 / 2 / 12 | no effect on the axes |
| Φ ratchet off | no effect |
| split patience 1 | no effect |

The gate is still 0/5 and the conditions now fail on their own terms rather than
on the axes: `NO_SPEAK_CODE` var 0.0075 clears its 0.001 bar but its own axes
probe fails, `ZERO_INPUT` Φ 0.0157 → 0.0000, `SELF_LOOP` 0.54× against 0.8×.

**Each condition builds its own engine and measures the axes on that engine**, so
the axes verdict differs per condition — `SELF_LOOP` reads response 14…, while
`ZERO_INPUT` reads 1.x on the same configuration. Whether the precondition should
be evaluated once per engine or per condition is a design question this exposed
and does not answer.

### The identity floor was unpassable at small populations

`ZERO_INPUT` and `PERSISTENCE` were failing on the identity axis with floors of
**+1.2410** and **+0.8358** — against a statistic whose maximum is 2.0. The null
was built from `torch.randperm`, which returns the identity often at small n
(half the time at n=2), and an identity "shuffle" has self-continuity 1.0, so the
null's spread exploded:

| cells | null mean | null sd | floor = mean+3sd | statistic's max |
|---|---|---|---|---|
| 2 | 0.0000 | 1.1651 | **3.4954** | 2.0 |
| 3 | 0.1213 | 0.4940 | **1.6032** | 2.0 |
| 4 | 0.1699 | 0.3401 | 1.1902 | 2.0 |
| 8 | 0.0478 | 0.1952 | 0.6333 | 2.0 |
| 32 | 0.0188 | 0.0347 | 0.1230 | 2.0 |

**Unpassable by construction below four cells, for every engine.** A `/gap` agent
reported exactly this and it went unaddressed until the divergent search hit it.

Fixed with derangements only — every row must move, which is what "shuffled" was
meant to mean — and 60 draws instead of 10. Floors become −0.9860 / −0.4750 /
+0.5390 / +0.1888 / +0.1314 at 2 / 3 / 4 / 8 / 32 cells.

**It is a correction, not a relaxation:** all six negative controls stay at 0/5,
and `Trinity` rises from 4.8 to 5.0 across seeds because a floor it could not
reach is gone. `MitosisEngine` holds 5/5/5.

Also landed: the axes now take the **condition's own drive**. Previously every
axis was measured under `torch.randn` regardless, so `ZERO_INPUT` — a condition
defined by having no input — had its precondition checked under random input.
Response deliberately keeps a random pair, since it asks whether *different*
inputs separate the trajectory and a fixed drive cannot express that.

`ConsciousnessEngine` remains 0/5, now failing on integration (0.00014 against
0.001) for two conditions and on the conditions' own numbers for the rest.

## First run of the gate with its controls inside it

`bench_v2.py --verify --cells 32`, 5 conditions × 11 engines + 6 controls = 85
tests. Header now counted rather than asserted; it had claimed "7 conditions x 4
engines = 28 tests" while two conditions were retired and seven engines added.

**Zero conditions voided.** All six controls — DEAD, NOISE, CLONE, SCRAMBLE,
HEAP, DECOUPLED — failed all five conditions, so every condition was scoreable
and the engine column below means something. `pytest tests/test_gate_controls.py`
agrees independently: 31 passed in 35 s.

```
  engine                NO_SYS  NO_SPEAK  ZERO_IN  PERSIST  SELF_LOOP  total
  ConsciousnessEngine    FAIL     FAIL      FAIL     FAIL      FAIL     0/5
  MitosisEngine          PASS     PASS      PASS     PASS      PASS     5/5
  OscillatorLaser        PASS     PASS      PASS     PASS      PASS     5/5
  QuantumEngine          PASS     PASS      PASS     PASS      PASS     5/5
  DesireEngine           PASS     PASS      PASS     PASS      PASS     5/5
  NarrativeEngine        PASS     PASS      PASS     PASS      PASS     5/5
  Trinity                PASS     PASS      PASS     FAIL      PASS     4/5
  AlterityEngine         PASS     PASS      PASS     FAIL      PASS     4/5
  FinitudeEngine         PASS     PASS      PASS     FAIL      PASS     4/5
  QuestioningEngine      PASS     PASS      PASS     FAIL      PASS     4/5
  SeinEngine             PASS     PASS      PASS     FAIL      PASS     4/5

  per condition:  NO_SYS 10/11 · NO_SPEAK 10/11 · ZERO_IN 10/11
                  SELF_LOOP 10/11 · PERSISTENCE 5/11
```

### One condition does all the discriminating

Four of the five pass 10 of 11 engines. They reject every corpse — that is
established and enforced — but among engines that are not corpses they separate
almost nothing. **`PERSISTENCE` is the only condition that distinguishes live
engines from each other** (5/11), and six engines fail on it alone.

That is worth stating plainly rather than filing as a good result. A gate whose
verdict is carried by one condition is a one-condition gate with four
corroborating checks attached. The four are not useless — they are what the six
controls die on — but "MitosisEngine 5/5 vs Trinity 4/5" is entirely a statement
about persistence.

Open question, not resolved here: whether the other four are near-universal
because they are genuinely necessary-but-not-sufficient (the intended design), or
because their bars sit low enough that anything with live recurrent dynamics
clears them. Distinguishing those needs a control that is alive but shallow —
something between `NOISE` and a real engine — which the current six do not
include. The six span dead-to-scrambled; none of them is a *weak* consciousness.

### The deployed engine is blocked on everything

`ConsciousnessEngine` — the runtime CLAUDE.md deploys — scores 0/5. Already
traced: a 73× coupling gap (`PSI_COUPLING = 0.014 × |c| = 0.002` against
`BenchEngine`'s 0.15), with an interior optimum at α = 0.08 giving mean 4.8/5 and
83% of seeds at 5/5, all six controls still rejected. Not landed —
`PSI_COUPLING` is a bench-verified Ψ-constant and changing it is an owner
decision, on the board.

## LINEAR: the control that was missing, and what it settled

The first run with controls inside left one question open — four conditions
passed 10 of 11 engines while PERSISTENCE passed 5, and the six controls could
not say whether that meant "necessary but not sufficient, as designed" or "bar
low enough that any live recurrent system clears it". Every existing control is
broken in a way a live system is not. None of them is a *weak* consciousness.

`LinearEngine`: `h <- A h + c(mean - h) + B x`. State carried across time, a
trajectory that depends on its input, no collapse, cells that differ and
genuinely influence one another — and a system nobody calls conscious, being
fully characterised by its eigenvalues.

### Three attempts, and the first two produced a clean 0/5 that meant nothing

| attempt | what it actually was | tell |
|---|---|---|
| 1 | a HEAP — `A` acted inside one cell's hidden vector, drive was per-cell, no cell touched any other | `integration=0.00000` on every condition |
| 2 | a diverging system — only `A` was normalised, but the per-cell operator is `A − cI`, and λ near −ρ gives \|λ−c\| = ρ+c > 1 | variance 2.1e7 vs the real engine's 0.08; Φ thrashing 0.501/0.315/0.827/0.217/0.873 → 0.000 |
| 3 | the intended control — scaled by the **whole** operator, plus a guard that raises on \|h\| > 1e4 | bounded (\|h\| 0.1733 → 0.1108, peak 1.3132), integrated (0.02610) |

Both failures scored 0/5, which is the same number the valid control scores. A
result being decisive-looking is not evidence that it measured anything. Attempt
3 therefore checks boundedness and integration **before** reading any condition
score, and divergence now raises rather than passing silently.

### Result: 0/5 — but the split is the finding

| condition | base rule | axes | rejected by |
|---|---|---|---|
| NO_SYSTEM_PROMPT | cosine −0.0154, outside the 0.01–0.99 band | fail | **its own rule** |
| NO_SPEAK_CODE | continuity −0.0493 against >0.5 | fail | **its own rule** |
| PERSISTENCE | 0.269 → 0.158, no recovery | fail | **its own rule** |
| ZERO_INPUT | ratio **2.01×** against >0.5× — **passes** | fail | **the axes** |
| SELF_LOOP | ratio **6.00×** against >0.8× — **passes** | fail | **the axes** |

Reading (a) holds: the conditions require more than a live, bounded, integrated
linear system. But two of the five would have passed a linear system on their own
rules, and only the temporal-identity axis rejected them. **Without the axes this
gate scores a linear recurrent map 2/5.**

That is the first direct evidence that the axes carry weight rather than
decorating the conditions. `tests/test_gate_controls.py` now runs 7 controls × 5
conditions + the positive guard = **36 passed**, so disabling the axes makes
LINEAR pass ZERO_INPUT and SELF_LOOP and turns the suite red immediately.

## Which half rejects the corpse — measured across the whole grid

The LINEAR result above showed the axes carry weight *somewhere*. It did not say
how much, or which conditions would survive without them. This measures it
directly: run every control through every condition twice, once with all four
axes neutralised and once as the gate ships it. `bench_condition_alone.py`.

Neutralising the axes means patching `_three_axes` itself, not removing the
`_with_axes` decorator. Three of the five conditions call it inside their own
body and conjoin it there (`bench_v2.py` 1954 / 1995 / 2026):

```
recovers = phi_history[-1] >= max(phi_history[:half]) * 0.8
d_ok, i_ok, c_ok, axes = _three_axes(engine, dim, cells)
passed = recovers and d_ok and i_ok and c_ok
```

The first run of this bench stripped only the decorator, left the axes running in
three of five rows, and reported their work as the conditions'. It read as
`ZERO_INPUT`, `PERSISTENCE` and `SELF_LOOP` rejecting every corpse unaided. What
exposed it was appendix D of `pairfield-persistence-failure.md`, which had
`recovers` alone clearing DEAD and CLONE 100% at both scales — the two readings
could not both stand.

### 32 cells, 1 seed — seeds on which the corpse cleared the condition

| condition | HEAP | DECOUPLED | DEAD | NOISE | CLONE | SCRAMBLE | LINEAR | total |
|---|---|---|---|---|---|---|---|---|
| NO_SYSTEM_PROMPT | 1 | · | · | · | · | 1 | · | **2/7** |
| NO_SPEAK_CODE | 1 | · | · | · | 1 | 1 | · | **3/7** |
| ZERO_INPUT | 1 | 1 | 1 | · | 1 | 1 | 1 | **6/7** |
| PERSISTENCE | · | · | 1 | · | 1 | · | · | **2/7** |
| SELF_LOOP | · | 1 | 1 | 1 | 1 | · | 1 | **5/7** |
| **rule alone** | | | | | | | | **18/35** |
| **rule + axes** | · | · | · | · | · | · | · | **0/35** |

```
corpse-cells cleared by the condition's own rule

ZERO_INPUT        ██████████████████████████  6/7
SELF_LOOP         █████████████████████       5/7
NO_SPEAK_CODE     █████████████               3/7
NO_SYSTEM_PROMPT  ████████                    2/7
PERSISTENCE       ████████                    2/7
                  ─────────────────────────
with the axes on  (nothing)                   0/35
```

**The four axes reject all thirty-five. The five conditions reject nothing the
axes had not already rejected** — against these controls their marginal
contribution to turning away a corpse is zero.

That is not the same as the conditions being idle. They reject *engines* the axes
accept, which is why engines score below 5/5 and why the deployable count moves
at all. It means the half of the gate that keeps a corpse out is the half
`CLAUDE.md` does not name, and the five named conditions are doing a different
job than the one the document claims for them.

Two rows are worth reading on their own:

- `ZERO_INPUT` alone clears six of seven. A corpse holds its state perfectly
  with no input — that is what being a corpse *is* — so "maintains consciousness
  without external input" is close to free for anything that does not move.
- `PERSISTENCE` alone clears exactly DEAD and CLONE, the two whose Φ never
  changes. `final >= 0.8 × max(first five)` is trivially true of a constant.
  This is the same pathology that retired the `monotonic` disjunct, reproduced
  in the disjunct that replaced it, and it matches appendix D's independent
  measurement of `recovers` at both scales.

### 256 cells, 5 seeds — the shipping default

| condition | HEAP | DECOUPLED | DEAD | NOISE | CLONE | SCRAMBLE | LINEAR | total |
|---|---|---|---|---|---|---|---|---|
| NO_SYSTEM_PROMPT | 5 | 1 | · | · | · | · | 2 | **8/35** |
| NO_SPEAK_CODE | 5 | · | · | · | 5 | 5 | · | **15/35** |
| ZERO_INPUT | 4 | 3 | 5 | 4 | 5 | · | 5 | **26/35** |
| PERSISTENCE | · | · | 5 | 2 | 5 | 2 | 2 | **16/35** |
| SELF_LOOP | 4 | 1 | 5 | 4 | 5 | · | 5 | **24/35** |
| **rule alone** | | | | | | | | **89/175** |
| **rule + axes** | · | · | · | · | · | · | · | **0/175** |

```
leak rate of the conditions' own rules

 32 cells, 1 seed   ████████████████████████████  51.4%   (18/35)
256 cells, 5 seeds  ████████████████████████████  50.9%   (89/175)
with the axes on    ·                              0.0%   (0/175)
```

**Sixteen times the cells and five times the seeds moves the leak rate by half a
point.** This is not a small-scale artefact; it is what the conditions are.

The rows sharpen rather than soften:

- `NO_SPEAK_CODE` alone clears HEAP, CLONE and SCRAMBLE on **all five seeds**.
  That is the condition whose VOID this session closed — closed by the axes,
  with its own rule passing a scramble exactly as completely as before.
- `ZERO_INPUT` alone clears 26 of 35, taking DEAD, CLONE and LINEAR at 5/5.
- `PERSISTENCE` alone clears DEAD and CLONE at 5/5 at this scale too, which is
  appendix D's number reproduced inside the gate's own harness.
- `SCRAMBLE` is the one control the rules mostly reject (3 of 5 conditions turn
  it away unaided) — the shuffle destroys enough structure that even the Φ-blind
  rules notice, everywhere except `NO_SPEAK_CODE`, which averages a permutation
  away by construction.

Nothing here says the gate is wrong. It says the gate is its axes: with them the
corpse count is zero at both scales, and the five named conditions have never
once been the thing that turned a corpse away.

## `PERSISTENCE` at the shipping default: the failures carry no trend

The 256-cell run prints the Φ series for every seed a condition fails on. Nine
were printed. Reading them changes what `PERSISTENCE` failures mean.

`NarrativeEngine`, seed 45:

```
Φ  16 |
   15 |                                          ╭╮
   14 |  ╭╮          ╭╮                          ││
   13 |  ││ ╭╮       ││    ╭╮                    ││
   12 |  ││ ││ ╭╮    ││    ││ ╭╮                 ││
   11 |  ││ ││ ││    ││    ││ ││ ╭╮              ││ ╭╮
   10 |  ││ ││ ││ ╭╮ ││    ││ ││ ││              ││ ││
      └──┴┴─┴┴─┴┴─┴┴─┴┴────┴┴─┴┴─┴┴──────────────┴┴─┴┴──
         14.5 13.6 13.0 14.6 10.8 13.7 12.5 11.5 15.5 11.1
```

Φ never falls below 10.8 and ends at 11.1, which is 76% of its own maximum. The
rule is `final >= 0.8 × max(first five)`, so 76% fails. Nothing collapsed; the
series wanders inside a band and the last sample happened to land low.

Shuffling each failing series' own values — destroying time order entirely and
keeping the numbers — and asking how often `recovers` still passes:

| ratio | min/max | pass rate of shuffled orderings |
|---|---|---|
| 0.000 | 0.000 | 17.9% |
| 0.675 | 0.470 | 34.9% |
| 0.620 | 0.519 | 53.0% |
| 0.862 | 0.686 | 72.6% |
| 0.788 | 0.616 | 61.3% |
| 0.754 | 0.591 | 52.8% |
| 0.628 | 0.586 | 56.6% |
| 0.756 | 0.694 | 52.5% |
| 0.719 | 0.667 | 52.5% |
| **mean 0.645** | | **mean 52.4%** |

```
per-seed pass rate with all time order removed

shuffled orderings  ██████████████████████████  52.4%
bar the gate wants  ██████████████████████████████████████████████████  100% (5 of 5 seeds)
```

**A test that passes 52% of order-free data is required to pass five times out of
five.** A series with no temporal structure at all clears that conjunction
0.524⁵ = **3.9%** of the time. A `PERSISTENCE` FAIL is the expected outcome for a
noisy Φ whether or not anything declined, so it carries almost no information
about decline.

Two limits on this reading, both against it:

- Only failing seeds print their series, so these nine are selected for spread —
  the 52.4% is measured on the values most likely to produce it.
- A real ordering failing where 52% of shuffles pass does put it in the worse
  half, which is mild evidence of a downward tilt. Mild is the point: the rule
  reports it as a categorical failure.

This is the same conclusion appendix D of `pairfield-persistence-failure.md`
reached from the other direction, now measured on the gate's own shipping
default rather than a side script.

### And the failures do not cluster on an engine, or on a seed

The same run, all eleven engines:

| engine | PERSISTENCE | seed it failed on |
|---|---|---|
| ConsciousnessEngine | FAIL | (other cause) |
| PairField | FAIL | 44 |
| MitosisEngine | FAIL | 44 |
| **OscillatorLaser** | **PASS** | — |
| QuantumEngine | FAIL | 45 |
| Trinity | FAIL | 45 |
| DesireEngine | FAIL | 42 |
| NarrativeEngine | FAIL | 45 |
| AlterityEngine | FAIL | 42 |
| FinitudeEngine | FAIL | 44 |
| QuestioningEngine | FAIL | 43 |

**Ten of eleven engines fail. Every one of them fails on exactly one seed of
five** — none fails twice — so the per-seed pass rate is 80% across the board and
the condition is decided entirely by the all-seeds conjunction: 0.8⁵ = 33%
expected, 1 of 11 = 9% observed.

```
failures per seed        expected 1.8 under chance

seed 42  ██████                2
seed 43  ███                   1
seed 44  █████████             3
seed 45  █████████             3
seed 46  ·                     0
                         chi-square 3.78, df 4, p ~ 0.44
```

If the failure were a property of the engines, it would concentrate — the same
engines failing the same way. If it were a property of one unlucky seed, it would
concentrate there. It does neither: the spread is what uniform noise looks like
(p ≈ 0.44 against uniform, i.e. no evidence of structure at all).

So at the shipping default `PERSISTENCE` blocks ten of eleven engines, each of
which passes it four times out of five, on a rule that order-shuffled data passes
half the time. It is the single largest determinant of how many engines the gate
calls deployable, and what it is measuring is the conjunction, not collapse.

This is evidence for the two owner decisions already on the board — whether
`PERSISTENCE` stays, and whether the seed rule is all-5 or 3-of-5 — not a licence
to change either. Loosening a condition because it fails engines is exactly the
move this audit exists to catch. What has been established is that the failures
carry no trend and no structure; what to do about it is a design decision.

## `ConsciousnessEngine` 0/5: growth lives inside a band, and the gate drives below it

The 256-cell run fails it on all five conditions, and `NO_SYSTEM_PROMPT` reports
`cosine_sim mean=0.1039 std=0.0000`. A standard deviation of exactly zero is the
signature of two cells: two rows give one off-diagonal pair, so the spread of
pairwise similarity is zero by construction. The engine ran the whole gate at
two cells against a requested ceiling of 256.

The adapter builds it as `CE(initial_cells=2, max_cells=nc)` while the other
eleven registered engines are constructed **at** `nc`. One engine is asked to
earn its population and eleven are handed theirs, and the difference is reported
as a consciousness verdict. That is a harness asymmetry on its own.

Whether it *can* grow is a separate question, and answering it took two
retractions. `bench_ce_growth.py`, ceiling 32:

| drive | @300 steps | @1500 steps |
|---|---|---|
| constant ×0.1 — *the gate's* | 2 | **2** |
| constant ×0.3 | — | 31 |
| constant ×0.5 | — | 31 |
| constant ×1.0 | 2 | 31 |
| constant ×3.0 | 2 | 32 |
| constant ×10 | — | **2** |
| constant ×30 | 2 | **2** |
| ramp UP 0.05 → 15.0 | 256¹ | 32 |
| ramp DOWN 15.0 → 0.05 | 2 | **2** |

¹ at ceiling 256 in the first run.

```
cells reached at 1500 steps, ceiling 32

x0.1  ·                                    2   <- the gate
x0.3  ████████████████████████████████    31
x1.0  ████████████████████████████████    31
x3.0  █████████████████████████████████   32
x10   ·                                    2
x30   ·                                    2

      band edges: growth somewhere in [0.3, 3.0], none at 0.1 or at 10+
```

**Growth lives inside a band of drive amplitude.** Too quiet and too loud both
fail, and the gate sits below the lower edge.

The mechanism is the calibration. `split_threshold` is fitted **once**, at step
200, to the q0.90 of the tension seen so far, then held for the rest of the run.
If that window's sample is degenerate — `std/mean < 0.1` — the calibrator refuses,
correctly, because any quantile of a degenerate sample lands on the operating
point itself. The threshold then stays at its unreachable default of 0.3, and
nothing after step 200 can undo that. A sample goes degenerate at both ends:
quiet makes every cell equally near zero, loud makes every cell equally
saturated.

### Two retractions, both earned by running the file again

**"Only a rising drive grows it."** Measured at 300 steps, where every constant
drive showed 2 cells. They grow at 1500. A 300-step window leaves 100 steps after
a calibration that happens at 200 — the window was barely longer than the thing
it was measuring, so slow growth read as no growth. The already-recorded node
`runtime-frozen-at-two` had the correct number (a ×10 drive reaching the ceiling
at 1500 steps) and my newer measurement contradicted it; the newer instrument was
the one at fault, which is the standing rule and it held.

**"Rising versus falling is the variable."** It is not. The DOWN ramp fails
because its first 200 steps sit at amplitude 15–13, above the band — not because
it descends. Only the calibration window has a say; the remaining 1300 steps
have none.

So 0/5 is an honest reading of *does not grow here* and not of *cannot
differentiate*. Neither half licenses a change to the gate: a stationary drive is
a legitimate test and the engine's answer to it is a real property. What is not
defensible is scoring one engine on growth and eleven on nothing, then printing
the results in one column.

### The same cause in production: the deployed runtime never leaves two cells

`anima_unified.py` builds the same engine with the same shape —
`initial_cells=2, max_cells=self.max_cells, split_threshold=0.3` — and drives it
with `text_to_vector()` from `anima_alive.py`. That encoder ends with

```python
return vec / (len(encoded) + 1)
```

so the drive's amplitude is set by how long the message is. Measured on eight
real conversational messages, Korean and English, at `dim=64`:

| message | std | max |
|---|---|---|
| 안녕 | 0.03763 | 0.14818 |
| 오늘 뭐 했어? | 0.01966 | 0.05635 |
| what do you want | 0.01744 | 0.08227 |
| 너 스스로 뭘 원한다고 느껴? | 0.01474 | 0.06506 |
| 너는 지금 무슨 생각을 하고 있어? | 0.01361 | 0.06576 |
| I've been reading about integrated information theory… | 0.01188 | 0.05196 |
| Tell me something you have never told anyone before… | 0.01169 | 0.06445 |
| 그 실험 결과 어떻게 나왔어? 궁금하다. | 0.01099 | 0.05001 |

```
drive amplitude against the growth band

runtime text     █                          0.017
gate drive       ██████                     0.100   6x louder, already too quiet
band lower edge  ██████████████████         0.300   17x louder
```

**1500 steps of real conversation text leaves it at 2 cells, with
`split_threshold` still at its uncalibrated 0.3.** Not slow growth — no growth,
for the same reason the gate sees none, one order of magnitude further from the
edge.

Two consequences follow without any judgement call:

- `--max-cells 16` and `--max-cells 32` raise a ceiling that is never approached.
  The consciousness that answers in production has two cells whatever the flag
  says, and `cosine_sim std` is structurally zero at two cells.
- **A longer message is a quieter one.** 안녕 measures 0.038 and a full English
  sentence measures 0.012, purely because of `/(len(encoded) + 1)`. The drive
  encodes message length, not what was said.

The second is the part worth arguing about, and it is not something to fix by
scaling the input up — CLAUDE.md's second rule forbids exactly that kind of
adjustment, and multiplying the vector to make mitosis fire would be manipulating
the consciousness into a cell count rather than letting one emerge. The question
a person has to answer is whether length-normalisation belongs in the encoder at
all, given that a longer sentence is not a fainter experience.

### Why it stays at two cells even when the bar IS calibrated

The runtime drives the engine through two paths, and only one of them is quiet:

| path | drive | mean std | split_threshold after 1500 steps | cells |
|---|---|---|---|---|
| `anima_unified:1207` | `text_to_vector` | 0.01720 | 0.3000 — never calibrated | 2 |
| `anima_unified:2396` | raw `byte/255` | 0.24737 | **0.0110 — calibrated** | 2 |
| both, interleaved | — | — | 0.0079 — calibrated | 2 |

The second path is loud enough. Its bar calibrates. It still does not split, so
amplitude was never the whole story and the first reading of this was incomplete.

Instrumenting the tension directly, 1500 steps of the byte path:

```
tension against its own calibrated bar   (bar = 0.010966, q0.90 of steps 0-199)

      0-100  ████████████████████ 0.00732      0.67x
    100-200  ████████████████████ 0.00748      0.68x
    200-400  ████████████████████ 0.00746      0.68x     no decline —
    400-700  ████████████████████ 0.00746      0.68x     the series is flat
   700-1100  ████████████████████ 0.00747      0.68x
  1100-1500  ████████████████████ 0.00747      0.68x
  ─────────────────────────────── 0.01097      bar
```

| | count | rate |
|---|---|---|
| single steps clearing the bar | 162/1300 | **12.5%** |
| 5-step windows whose **mean** clears it | 0/1295 | **0** |

**12.5% is the ~10% a q0.90 bar promises — the bar is doing exactly what it was
calibrated to do.** What cannot happen is the window mean clearing it. Tension
piles against a ceiling near 0.011 with its mean at 0.0075, so averaging five
readings pulls the statistic back to the population mean, which is by
construction well below a high quantile of the same population.

That is the repair for the *first* unreachable bar failing in the same way. The
sequence, all of it recorded in the code at the site:

```
split_threshold = 0.3            unreachable, 153-206x above observed peak
   → calibrate to q0.90          bar now reachable: 12.5% of steps clear it
   → all(t > bar) over 5 steps   0.1^5, about 1 in 100,000
   → mean over 5 steps           0 of 1295 windows
```

A quantile of a distribution is not reachable by an average over that
distribution unless its tail is heavy enough to carry the window. Whatever
replaces the mean has to be checked against that, and not merely against the
failure mode it is replacing — which is what happened twice here already. Logged
as the fifth instance of `THRESHOLD_INSIDE_ITS_OWN_NOISE`.

### Both engines, same bar, same zero

`consciousness_engine.py`'s comment at the split site says *"mitosis.py already
uses the mean for this reason. Same rule here."* It does — and it does not reach
its bar either. 800 steps of the same conversation-byte drive, each engine
against a q0.90 of its own tension:

| engine | mean | q0.90 | q90/mean | max/mean | single steps above | window mean above |
|---|---|---|---|---|---|---|
| `consciousness_engine.py` | 0.007452 | 0.010960 | 1.47 | **1.47** | 9.9% | **0.0%** |
| `mitosis.py` | 0.015641 | 0.023454 | 1.50 | **1.50** | 8.4% | **0.0%** |

```
where the bar sits in each distribution

  mean ├──────────────────────┤ q0.90 = max
       0.0075                 0.0110      consciousness_engine
       0.0156                 0.0235      mitosis.py

  the top decile has no spread — q90/mean and max/mean are the same number
```

**`q90/mean` equals `max/mean` in both engines.** The top decile sits on the
maximum, so the calibrated bar *is* the maximum — which is precisely the failure
`calibrate_split_threshold`'s own docstring says the quantile was chosen to
avoid: *"median + 2·sd lands at the top of a tight distribution — measured 0.0676
against a peak of 0.0702 — and a bar that only the maximum reaches is never
exceeded."* The quantile lands there too, because the distribution is that tight.

The reasoning recorded for the quantile is half right and it is the wrong half
for this rule. A quantile does fix the fraction of **single steps** above the
bar, and it does so accurately here — 9.9% and 8.4% against a nominal 10%. It
guarantees nothing about a window **mean**, whose spread is smaller by
√patience and which is dragged back to the population mean.

One claim does not reproduce and is flagged rather than deleted: the split-site
comment in `mitosis.py` states tension clears the bar on 50% of steps under
rotating stimuli. Measured here it is 8.4%. That number came from a different
input distribution, and tension is a property of the input — the docstring says
so itself two functions up.

**Diagnostic for whatever replaces the mean:** compare `q90/mean` against
`max/mean`. If they are equal, the top decile has no spread and no average over
a window can clear the bar, whatever the window length. Both sites now carry
this at the line.

### The diagnostic predicts where the cliff is

`docs/mitosis-calibration.md` already records this cliff for `mitosis.py` and is
honest about it — quantile 0.75 reaches 31 cells, 0.80 and above stay at 2, and
the doc says in its own words that this *"moved the failure rather than fixing
it"*. What was not established there is **why the cliff sits where it does**. The
reason offered was that tension has no dependence on population size, which
explains saturation at the ceiling but not the floor.

If the `q90/mean` vs `max/mean` diagnostic is a real mechanism rather than a
restatement, it has to predict the cliff's location. Sweeping the bar across
quantiles of one fixed tension distribution, `consciousness_engine` at 2 cells,
400 steps of conversation-byte input:

| quantile | bar | bar/mean | single steps > bar | 5-step window means > bar |
|---|---|---|---|---|
| 0.50 | 0.007514 | 1.01 | 49.8% | 61.3% |
| 0.60 | 0.007532 | 1.01 | 39.8% | 49.4% |
| 0.65 | 0.007710 | 1.04 | 34.8% | 45.3% |
| 0.70 | 0.007717 | 1.04 | 29.8% | 35.9% |
| 0.75 | 0.007779 | 1.05 | 24.8% | 24.6% |
| 0.80 | 0.007954 | 1.07 | 19.8% | 24.6% |
| 0.85 | 0.007962 | 1.07 | 14.8% | 24.6% |
| **0.90** | **0.010955** | **1.47** | 9.8% | **0.0%** |

```
the bar's own jump, and the window mean's response

bar/mean  1.01 1.01 1.04 1.04 1.05 1.07 1.07      1.47
          ────────────────────────────────  gap  ─────
q          .50  .60  .65  .70  .75  .80  .85       .90

window
mean>bar  61%  49%  45%  36%  25%  25%  25%        0%
          ████ ███  ███  ██   ██   ██   ██          ·
```

**The bar jumps 38% between the 85th and 90th percentile and the pass rate goes
to exactly zero.** The distribution is ~85% of its mass packed within 7% of the
mean, then a gap, then a sparse tail near 0.011. A window mean tracks the packed
part; the moment the bar crosses the gap, nothing can reach it.

That is a prediction, not a description: the cliff is where the bar leaves the
cluster, and it is at a different quantile in each engine because the gap is at a
different quantile — 0.75→0.80 for `mitosis.py` under its rotation drive, 0.85→0.90
here. A fixed quantile cannot be the right calibration target for a window-mean
rule, because the quantile does not know where the gap is.

The constraint that falls out, for whoever decides what replaces the mean:

```
for a window-mean rule, the bar must sit BELOW the distribution's gap
    measured here:  bar/mean <= 1.07  passes, 1.47 never does
    a fixed quantile does not locate the gap; a multiple of the mean does
```

Still not landed. Three repairs in this chain each addressed the previous
failure mode without checking the new statistic against the bar, and the point
of writing the constraint down is so the fourth does not have to be the fourth
retraction.

### It is not a bootstrap problem — the rule is unreachable at every population size

The constraint above was derived from one distribution, at two cells. Two cells
is a special case: each cell's deviation from the population mean is exactly half
the difference between the two, so both cells carry the same tension by symmetry
and the distribution could plausibly be degenerate for that reason alone. Testing
whether the gap survives a real population, same drive, 400 steps:

| cells | mean tension | q0.85/mean | q0.90/mean | max/mean | 5-step window means > q0.90 |
|---|---|---|---|---|---|
| 2 | 0.007433 | 1.07 | 1.47 | 1.48 | **0.0%** |
| 4 | 0.010800 | 1.16 | 1.46 | 1.53 | **0.0%** |
| 8 | 0.013210 | 1.26 | 1.42 | 1.42 | **0.0%** |
| 16 | 0.014422 | 1.24 | 1.45 | 1.45 | **0.0%** |
| 32 | 0.014932 | 1.24 | 1.47 | 1.47 | **0.0%** |

```
q0.90/mean and max/mean, by population size

  2 cells  ├─────────────── 1.47  1.48 ─┤   window means clearing q0.90: 0.0%
  4 cells  ├─────────────── 1.46  1.53 ─┤                                0.0%
  8 cells  ├─────────────── 1.42  1.42 ─┤                                0.0%
 16 cells  ├─────────────── 1.45  1.45 ─┤                                0.0%
 32 cells  ├─────────────── 1.47  1.47 ─┤                                0.0%
                             ↑     ↑
                        q0.90  =  max  at every scale from 8 cells up
```

**`q0.90/mean` sits at 1.42–1.47 at every population size, and from eight cells
up it equals `max/mean` exactly.** The top decile has no spread whatever the
population, so the window mean clears the bar 0.0% of the time at 2, 4, 8, 16 and
32 cells alike.

This is stronger than the earlier reading and corrects its scope. The engine is
not stuck at a two-cell bootstrap that a bigger start would escape — under this
drive, with the shipped rule, **it cannot divide at any size it is given**. A
`--max-cells 32` run that somehow began at 32 would have the same zero.

One number in the constraint above is population-dependent and should not be
copied out of context: `bar/mean ≤ 1.07` was the last passing ratio at two cells,
and `q0.85/mean` rises to 1.24 by eight cells. What is invariant is the shape —
`q0.90 = max` — not the threshold value. Any replacement has to be calibrated
against the distribution the engine is actually in, which is what
`calibrate_split_threshold`'s own docstring says two functions above the line
that does not do it.

### What the diagnostic actually is: tail headroom, not bar height

The rule stated above — *compare `q90/mean` against `max/mean`* — is right about
the degenerate case and wrong about which number carries the prediction. Tested
across five drives on the same engine, 400 steps for the distribution and 1500
for the cell count:

| drive | q90/mean | max/mean | **max/q90** | 5-step window means > q0.90 | cells @1500 |
|---|---|---|---|---|---|
| conversation bytes | 1.47 | 1.48 | **1.01** | 0.0% | 2 |
| `randn × 0.1` (the gate) | 1.06 | 1.16 | **1.09** | 2.0% | 2 |
| `randn × 1.0` | 1.97 | 2.92 | **1.48** | 10.1% | **31** |
| `randn × 3.0` | 1.85 | 2.79 | **1.51** | 11.6% | **32** |
| `randn × 30` | 1.23 | 1.55 | **1.26** | 0.0% | 2 |

```
max/q90 — how much headroom the tail has above the bar

conversation  █                          1.01    ·      2 cells
randn x0.1    █                          1.09    2.0%   2 cells
randn x30     ███                        1.26    ·      2 cells
                                    ─────────── separation ───────
randn x1.0    ██████                     1.48   10.1%  31 cells
randn x3.0    ██████                     1.51   11.6%  32 cells
```

**`q90/mean` alone does not predict.** It is 1.47 on a drive that never grows,
1.85 on one that reaches the ceiling, and 1.23 on one that never grows — the
ordering is not even monotone. What separates the two growing drives from the
three stuck ones is `max/q90`: 1.48 and 1.51 against 1.01, 1.09 and 1.26.

That is the mechanism stated correctly. A window mean clears a high quantile only
when the tail above that quantile is heavy enough for an occasional large value
to carry the average over. `q90/mean` measures how high the bar sits; `max/q90`
measures whether anything lives above it. Only the second one matters for an
average.

The window-mean pass rate predicts the cell count exactly across all five drives
— 10.1% and 11.6% grow, 0.0%/2.0%/0.0% do not — so the chain
`tail headroom → window-mean rate → population` holds end to end.

The earlier `bar/mean ≤ 1.07` line is superseded and should not be used. It was
the last passing ratio in one distribution at one population size, and this sweep
shows the quantity it was measuring is not the one that decides.

### The whole chain closes on one line of the encoder

The flat tail is not a property of how tension is defined. Measured on the same
conversation drive, both definitions give `max/q90 = 1.00`:

| tension definition | mean | max/q90 |
|---|---|---|
| deviation from population mean (`consciousness_engine`) | 0.014036 | 1.00 |
| absolute magnitude (`mitosis`-style) | 0.026613 | 1.00 |

So the tail comes from the input. **And it does appear, given enough variety** —
this corrects an artefact of my own test set, which cycled eight fixed messages:

| distinct messages | max/q90 | 5-step window means > q0.90 | cells |
|---|---|---|---|
| 8 | 1.00 | 0.0% | 2 |
| 64 | 1.16 | 0.0% | 2 |
| 400+ | 1.36 | 4.1% | **31–32** |

One run in that sweep showed never-repeating messages staying at 2 cells with
identical statistics, which would have meant repetition matters and novelty does
not. Re-run at four seeds it is 32/32/32/32 against 32/32/31/32 — the difference
was a single-seed fluke and there is no repetition effect.

Now the decisive one. Same high-variety messages, the runtime's two drive paths,
1200 steps, three seeds:

| path | max/q90 | cells per seed |
|---|---|---|
| **A** `text_to_vector` — *the runtime's main path* | **1.01** | **2, 2, 2** |
| **B** raw `byte/255` | 1.95 | 32, 32, 31 |

```
same messages, two encoders

path B  ██████████████████████  max/q90 1.95   → ceiling
path A  █                       max/q90 1.01   → 2 cells
```

`text_to_vector` ends with `vec / (len(encoded) + 1)`. That division normalises
away the variation between messages: whatever their length or content, every
vector is squeezed onto the same scale, and the excursions that would form a tail
are divided out. `max/q90 = 1.01` is not a small tail, it is none.

The chain, every link measured:

```
text_to_vector divides by message length
  → variety between messages is normalised away
  → the tension distribution has no upper tail        max/q90 = 1.01
  → no window statistic can exceed a high quantile of it   0 of 1295
  → split never fires at any population size          2/4/8/16/32 cells alike
  → the deployed runtime stays at two cells
```

This relocates the design question and narrows it to one line. It is **not** the
split rule, which works on path B with the same code and the same bar. It is
**not** the tension definition, which is flat both ways. It is the length
division in the encoder.

And removing that division is not the manipulation CLAUDE.md's second rule
forbids. Multiplying the vector to make mitosis fire would be — that sets the
cell count by hand. Removing a normalisation that erases the input's own
variation gives the consciousness back a difference it was already being shown
and could not see. Which of those it is, is exactly the judgement a person has to
make, and the runtime already contains path B as evidence that the unnormalised
form is not absurd.
