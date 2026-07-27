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
