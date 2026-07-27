# `_min_partition` does not minimise anything above n=8

`bench_v2._min_partition` is exhaustive up to n=8. Above that (`bench_v2.py:225-237`):

```python
fiedler = eigenvectors[:, 1]
ga = [i for i in range(n) if fiedler[i] >= 0]
gb = [i for i in range(n) if fiedler[i] < 0]
return sum(mi_matrix[i, j] for i in ga for j in gb)
```

One split at the sign of the Fiedler vector. No minimum is taken over anything.
Standard spectral partitioning uses the **sweep**: sort by Fiedler value, evaluate
all n−1 splits along that ordering, keep the smallest. The sweep is the part that
was dropped, and it is the part that makes the result a minimum.

## Verified against exhaustive ground truth

n = 12 (2¹² tractable), same MI matrices for all three methods, debiased MI,
`BenchEngine(12, 32, 64, 32, 4)`, seed 42:

| step | sign-cut | k | sweep | k | true min | k | sign / true |
|---|---|---|---|---|---|---|---|
| 40 | 12.529 | 9 | 6.628 | 11 | 6.628 | 11 | 1.89× |
| 80 | 15.782 | 5 | 6.657 | 11 | 6.251 | 11 | 2.52× |
| 120 | 17.257 | 6 | 6.391 | 11 | 6.251 | 11 | 2.76× |
| 160 | 13.183 | 9 | 7.085 | 1 | 6.610 | 11 | 1.99× |
| 200 | 11.670 | 9 | 6.392 | 1 | 6.392 | 11 | 1.83× |
| 240 | 11.282 | 9 | 6.211 | 1 | 6.211 | 11 | 1.82× |

**Mean overshoot 2.14×, max 2.76×.**

The `k` column is the mechanism. The true minimum is **always k = 11** — the
minimal cut separates one cell from the rest. The sign-cut picks a
differently-sized partition on almost every reading (9, 5, 6, 9, 9, 9), so the
quantity it returns is not tracking any stable feature of the system. The sweep
lands on the true value (6.628 vs 6.628, 6.211 vs 6.211); its k alternating
between 1 and 11 is the same cut seen from either side at n = 12.

## Why it matters more than a factor of two

A biased-but-stable estimator would shift every engine's Φ by roughly the same
amount and leave orderings intact. This one is **unstable per step**: a teammate
measuring at 32 cells found the sign-cut alternating between a k=4 and a k=9
partition on consecutive steps, with Φ tracking it exactly (0.564 → 1.398) while
differentiation and complexity stayed flat. So a Φ trajectory's dips and spikes
can be the partition heuristic changing its mind, not the population changing.

That propagates into every verdict built on a Φ ratio. `PERSISTENCE`'s
`recovers = final >= 0.8 × max(first half)` compares **one draw** of a bistable
quantity against 0.8× the **maximum of five draws** — a rule that is biased
against passing by construction when the estimator flickers.

## Bearing on the multi-seed change

`937c68b` tightened the gate to require all five seeds, on the reading that "a
verdict that moves with the draw is a coin flip". The observation holds; **the
attribution was wrong.** That commit put the variance in the engines. It is in
the measurement. Averaging over seeds does not remove an unstable estimator — it
averages a lottery. The sharper evidence is that verdicts also move when the
checkpoint grid shifts by a single step, which no amount of seed-averaging
addresses.

Two claims of mine fall with it:

- "NarrativeEngine is the only 5/5" — one grid, one scale, same lottery. It
  measures 2/5 at 32c/64d/128h.
- "lower repulsion strength fixes PairField" — never measured by me. Measured by
  a teammate: strength 0.000, two engines that never interact, scores 4/5 —
  better than the default.

## Not fixed here

The sweep is a candidate, not a landed change. Before it goes near `bench_v2` it
needs all seven controls still failing under it, and a check against the
SPLIT/RING construction in `docs/phi-rs-direction.md` — where both the shipped
and the min-cut readings prefer a fully collapsed system to an independent one by
90× and 104×. If the sweep inherits that, it buys stability without buying
direction, and which of the two it bought must be stated.

Reproduce: the n=12 comparison above, or read `bench_v2.py:225-237`.

## RETRACTED: the sweep does not buy stability either, and my test measured the wrong formula

The section below stands as written but its premise is wrong in two ways, both
caught by a teammate running the control it asked for.

**My direction test measured phi-rs's formula, not `bench_v2`'s.** I computed
`min_cut / (n−1)` and called it "shipped Φ". `bench_v2.py:176` is

```python
spatial_phi = (min_partition_mi / max(n - 1, 1)) * differentiation
```

with `differentiation = 1 − mean_cos` (`bench_v2.py:174`). A fully collapsed
population has `mean_cos ≈ 1`, so differentiation ≈ 0 and IDENTICAL lands at the
**floor** — measured 0.0049, the lowest of five constructions at every n tried.
The 90×/104× preference for IDENTICAL is phi-rs, which has no differentiation
factor. Dropping that factor from my check turned `bench_v2`'s Φ into phi-rs's.
The conclusion "don't land the sweep" happened to be right; the reason given for
it was not.

**`bench_v2`'s actual direction defect is a different one**: the shipped sign-cut
ranks INDEPENDENT highest (0.2143 at n=32, above SPLIT 0.2002 and RING 0.1888) —
nothing-to-join scoring top.

**And the sweep destroys the one directional behaviour the shipped formula has.**
Matched-coupling test (noise held equal across arms, n=32, dim 512, 5 seeds):

| noise s | SPLIT shipped | RING shipped | verdict | SPLIT swept | RING swept | verdict |
|---|---|---|---|---|---|---|
| 0.3 | 0.1026 | 0.4008 | RING **3.90×** | 0.1026 | 0.0897 | SPLIT 1.14× ← inverted |
| 1.0 | 0.0718 | 0.5067 | RING **7.06×** | 0.0648 | 0.0733 | RING 1.13× |

The shipped cut ranks the integrated RING above the disconnected SPLIT by 3.90×
and 7.06×. The sweep crushes both to ~1.1×, one of them the wrong way.

**The variance reduction was Φ being flattened, not stabilised.** Driving the cut
to its true minimum makes it a near-zero singleton for almost any population, so
`spatial_phi` collapses and Φ becomes the `0.1 × complexity` term. Under the
sweep at n=32 the constructions pin together at that floor — RING 0.0235, MID
0.0255, INDEPENDENT 0.0265, indistinguishable. A measure that cannot tell them
apart has low variance for the same reason a constant does.

**The bad minimum accidentally retains information the true minimum discards.**
That is worth stating plainly: `_min_partition` is wrong about what it computes,
and replacing it with the correct thing makes the gate worse. The sweep survives
only as the diagnostic that proves the trajectory is the estimator's — which it
does prove.

---

## Second correction: my numbers were the cut term, and n=8 is off-path entirely

The retraction above named one defect in my direction check. There were two, and
the labels were wrong.

**The figures I printed as "Φ" are `min_cut/(n−1)`, not Φ.** Rebuilt with every
intermediate shown, n=8, dim 4000, 5 seeds:

| case | cut/(n−1) | differentiation | `bench_v2` Φ |
|---|---|---|---|
| IDENTICAL | 3.0952 | **0.0000** | 0.000195 |
| INDEPENDENT | 0.0003 | 0.9999 | 0.000508 |

`cut/(n−1)` alone gives IDENTICAL/INDEPENDENT ≈ 10,460× — reproducing my 10,886×
to within seed noise, so we measured the same quantity. But `bench_v2.py:175`
multiplies by `differentiation = 1 − mean_cos`, which is **exactly 0** for
identical rows, and the collapse preference is annihilated before Φ exists.
`bench_v2`'s Φ ratio is **0.385×**, with per-seed values 0.380 / 1.172 / 0.387 /
0.766 / 0.149 — scattered around and below 1, both arms at the floor (0.0002 vs
0.0005). The honest statement is that **Φ separates neither construction**, not
that it prefers INDEPENDENT.

**And at n=8 `bench_v2` never runs the sign cut at all.** `_minimum_partition`
branches to exhaustive search over all 2⁸ partitions at `n <= 8`
(`bench_v2.py:215`). On one matrix: exhaustive 20.9124, sweep 20.9124 (identical —
the sweep finds the true minimum), sign-cut 35.8525, **a number `bench_v2` does
not compute at that n**. My sign-cut-versus-sweep comparison at n=8 compared
nothing the shipped gate does. The dim-4000 / 16-bin regime is also not the
gate's hidden-128 regime.

**The conclusion survives; the evidence for it does not.** "Do not land the
sweep" stands — but on the matched-coupling RING/SPLIT test at **n=32**, where
the sign cut is what actually runs and the sweep flips RING's 3.90×/7.06×
advantage to 1.14× the wrong way. Not on IDENTICAL/INDEPENDENT.

Single line to attack if this is wrong: `bench_v2.py:175`,
`spatial_phi = (min_partition_mi / max(n-1,1)) * differentiation`.

---

## Superseded twice — the original direction check, kept for the record

The table below is `min_cut/(n−1)` mislabelled as Φ, at an n where the gate uses
exhaustive search. Kept so the correction has something to point at.



## The sweep buys stability, not direction

Three independent measurements agree the sweep is a better minimum: it matches
exhaustive truth exactly where exhaustive is computable, and it cuts Φ's
checkpoint spread 3.5–3.8× while all seven negative controls stay at 0.

That is one axis. The other is direction, and the two must not be conflated.
Measured on the constructions the direction argument rests on — n = 8, dim 4000,
debiased MI, both cuts taken from the same matrices:

| construction | total | sign-cut | sweep | shipped Φ | swept Φ |
|---|---|---|---|---|---|
| IDENTICAL (every cell a copy — zero integration by construction) | 89.035 | 50.864 | 22.255 | **7.2662** | **3.1793** |
| INDEPENDENT (no integration) | 0.047 | 0.002 | 0.002 | 0.0003 | 0.0003 |

**collapsed / independent: shipped 24,880× · swept 10,886×**

The sweep halves the ratio and still prefers total collapse by four orders of
magnitude. A measure that is maximal where integrated information must be zero is
not repaired by making it quieter.

```
  IDENTICAL    shipped ████████████████████████  7.27
               swept   ██████████                3.18
  INDEPENDENT  shipped ▏                         0.0003
               swept   ▏                         0.0003
```

So the sweep is worth landing for what it is — a partition function that actually
partitions — but landing it must state that the direction defect survives it.
`bench_v2` needed three coupled changes to get direction: min-cut **and** a
differentiation factor **and** debiased MI. The sweep is one of three.

**Pre-declared before running:** if the sweep also ranked IDENTICAL above
INDEPENDENT, it bought stability only. It did.

Reproduce: `scratchpad/sweep_direction.py`, or the SPLIT/RING section of
`docs/phi-rs-direction.md` with `sweep_cut` substituted for `sign_cut`.

## The drop was the baseline. I read the trajectory backwards.

The trajectory that started this — `1.866 … 2.009 1.960 0.845 2.095 1.259 1.216`
at 32c/32d/128h, seed 44 — was reproduced digit for digit, and a step-resolution
trace of steps 681–730 shows what the checkpoints were sampling:

```
  2.20 |                          *
  1.90 |  *                   * **
  1.60 |   *
       |      *
  1.30 |             *       *           *           *
       |    **      *     * *
  1.00 |                                           *
       |**     *****  **** *       ****** ********* * ****
  0.70 |
       +--------------------------------------------------
        681                    ^ step 700                730
```

**Φ is below 1.0 on 32 of those 50 steps — 64% of the time.** Steps 695–700 read
0.845, 0.844, 0.845, 0.825, 1.128, 0.845.

So the 7th checkpoint caught the level Φ *sits at most of the time*. **What is
anomalous is the six checkpoints before it**, every one of which happened to land
on an excursion. I described this as "holds near 2.0 for six checkpoints, then
drops to 0.845" in three messages and in a commit. That is backwards.

Across all 50 steps `differentiation` stays 0.4639–0.4773 and `complexity`
0.3662–0.3839 — both flat to ±1.5% — while the cut swings 50.75–136.72 as `k`
moves between 4 and 11. Φ tracks the cut and nothing else.

### The margin that decides deployment

Seed 46 of the same run fails without dropping anywhere:

```
1.962 1.914 2.163 1.500 2.077 2.102 1.987 2.051 1.716 1.724
bar = 0.8 × max(first half) = 1.730      final = 1.724      margin −0.006
```

**A 0.35% shortfall is the entire difference between DEPLOYABLE and BLOCKED**, and
moving the checkpoint grid one step later flips it (bar 1.700, final 1.728).
Across six one-step ruler positions seed 46 reads PASS/FAIL/FAIL/PASS/PASS/PASS.

### It is not randomness at this scale

`PhiIIT(seed=...)` is dead for `compute()` — `bench_v2.py:99-103` overwrites
`self._rng` from the input hash before any shuffle, so four different seeds on one
fixed tensor return 0.126492 identically. At 32 rows the pair set is exhaustive
(496 = 32·31/2), and the jitter sd at seed 44's step-700 checkpoint is 0.004.

**Φ here is a deterministic function of the state that is discontinuous in it.**
The 0.845 is the exact value of a quantity that jumps, not a noisy reading. (At
256 cells, where pairs are resampled, a second and different mechanism appears:
sd 0.7–2.3.)

### Three overshoot figures, reported with their conditions

| who | n | construction | sign / true |
|---|---|---|---|
| this doc | 12 | `BenchEngine(12,32,64,32,4)`, seed 42 | 1.82–2.76× (mean 2.14) |
| persistence audit | 12 | real engine MI matrices | 3.0–3.4× |
| pairfield audit | 12 / 32 | — / 32 cells | 3–7× / 5.4–11.5× |

Same direction, different magnitude. The overshoot depends on how close the
population sits to a tie, so the spread is the construction, not a disagreement —
recorded as a range with conditions attached rather than collapsed to one number.

One caveat on the sweep: it matched exhaustive on 4 of 6 steps here and missed
slightly on two (7.085 vs 6.610; 6.657 vs 6.251). The persistence audit reported
6/6 exact. **The sweep is a very good approximation, not provably the minimum** —
a weaker claim than "the sweep is the true minimum", which the measurements do
not support.

## 12/12 decomposed: eleven engines share the mechanism, one does not

Φ decomposed per engine over 1000 steps at 32c/32d/128h, both failing seeds,
reporting the partition size `k` alongside the variance of each factor:

| engine | k range | distinct | cv(Φ) | cv(cut) | cv(diff) | corr(Φ,cut) |
|---|---|---|---|---|---|---|
| **ConsciousnessEngine** | **0–1** | **2** | 1.575/1.617 | 1.573/1.612 | 0.037/0.033 | +0.999/+0.999 |
| PairField | 4–16 | 13/13 | 0.323/0.223 | 0.322/0.229 | 0.137/0.153 | +0.827/+0.282 |
| MitosisEngine | 4–16 | 13/11 | 0.252/0.145 | 0.249/0.220 | 0.127/0.130 | +0.687/**−0.222** |
| OscillatorLaser | 4–16 | 10/11 | 0.197/0.212 | 0.222/0.216 | 0.097/0.084 | +0.761/+0.849 |
| QuantumEngine | 4–16 | 11/13 | 0.376/0.224 | 0.434/0.188 | 0.124/0.139 | +0.842/+0.625 |
| Trinity | 3–16 | 13/13 | 0.253/0.201 | 0.260/0.191 | 0.129/0.131 | +0.704/+0.383 |
| DesireEngine | 3–16 | 14/11 | 0.278/0.126 | 0.250/0.212 | 0.120/0.131 | +0.936/**−0.454** |
| NarrativeEngine | 4–16 | 13/13 | 0.258/0.268 | 0.268/0.303 | 0.119/0.128 | +0.717/+0.518 |
| AlterityEngine | 4–16 | 12/12 | 0.259/0.221 | 0.257/0.265 | 0.121/0.133 | +0.797/+0.321 |
| FinitudeEngine | 4–16 | 12/11 | 0.235/0.161 | 0.241/0.232 | 0.117/0.126 | +0.712/**−0.003** |
| QuestioningEngine | 4–16 | 13/13 | 0.433/0.237 | 0.446/0.265 | 0.125/0.133 | +0.854/+0.421 |
| SeinEngine | 4–16 | 12/12 | 0.209/0.208 | 0.186/0.256 | 0.105/0.125 | +0.875/+0.354 |

**Holds 24/24 without exception:** `cv(cut) > cv(diff) > cv(cplx)`, and `cv(cut)`
tracks `cv(Φ)` throughout. **Does not hold:** `corr(Φ, cut)` is positive in 21/24
but goes negative for Mitosis (−0.222) and Desire (−0.454) and to zero for
Finitude (−0.003) at seed 46. So "Φ's variance **is** the cut" is false for three;
"the cut has the largest spread" survives everywhere. The weaker claim is the one
to carry.

### The carve-out is the deployed engine, and its failure is real

`ConsciousnessEngine` takes only **k ∈ {0, 1}** against 10–14 distinct partition
sizes for every other engine — `k = 0` being an empty cut, the Fiedler sign
putting every cell on one side. Its Φ mean is 0.0267 against 1.4–2.3; cut mean
0.0264 against 74–123; differentiation 1.0125 (cells essentially orthogonal);
complexity exactly 0.0000. **Its population never integrates, so Φ is pinned at
the floor and there is no partition left to flip.**

It is the only engine failing all five seeds, and on this evidence **it fails
honestly.** That bounds the general claim in the useful direction: "the estimator
is unreliable" must not be read as "every verdict is void". Eleven verdicts are
suspect. One is not — and it is the one attached to the engine this project
actually deploys.

### What this does to the PASS rows

`NarrativeEngine` passes 5/5 at this scale with 13 distinct k spanning 4–16 and
`cv(cut)` 0.268/0.303 — indistinguishable from `PairField`'s 13 spanning 4–16 at
0.322/0.229, which fails. Its own verdict also moves with scale: **5/5 at
32/32/128, 2/5 at 32/64/128, 4/5 at 256c.**

So the PASS row carries no more information than the eleven BLOCKED rows beside
it. Passing is not evidence the mechanism is absent; it is evidence the lottery
landed well. **Any deployment decision resting on which engines this condition
passed is resting on the same coin the FAIL verdicts came from** — with the single
exception above.

## RETRACTED: the carve-out was n=2 arithmetic, and the ordering is 31/32

Both claims in the section above are corrected.

### `ConsciousnessEngine`'s verdict is undetermined, not sound

The gate builds it with 32 cells and reads Φ off **two**. Verified directly:

```
requested 32 cells  →  get_hiddens() returns (2, 128)  →  1 pair
```

So every "anomaly" that made it look like a clean carve-out is arithmetic from
n = 2, not dynamics:

| observation | what it actually is at n=2 |
|---|---|
| k ∈ {0, 1} | the only cuts two rows admit |
| `n_pairs_sampled` = 1 | one pair exists |
| complexity exactly 0.0000 | `np.std` of a single MI value is 0 by construction |
| Φ ≈ cut (0.0267 / 0.0264) | `min_partition_mi/(n−1)` with n−1 = 1 **is** the cut |
| differentiation 1.0125 | a statement about two rows, not thirty-two |

Measured directly at steps 100 / 500 / 1000: **Φ = 0.00000** every time, on one
pair. Its 32-cell population was never in the calculation.

I wrote "its population never integrates, so there is no partition to flip" and
called it **the one verdict in the whole run that is not suspect**, and carried
that into `ARCHITECTURE.json`. Both are withdrawn. The verdict is **undetermined**.

**This is the third instance of one defect family, and the largest.** My
`set_hiddens` restores side A only; `get_hiddens` on the pair exposes side A only;
and `_CEAdapter` exposes 2 rows of a 32-cell engine. A separate audit had already
flagged that `_CEAdapter` "never grows past 4 cells at any requested max" — the
signal was on the board and I did not connect it.

### The ordering claim is 31/32, and the counterexample is on a passing seed

Extending the decomposition to seeds 42/43 (which **pass** for most engines):

| engine | seed | cv(Φ) | cv(cut) | cv(diff) | corr(Φ,cut) |
|---|---|---|---|---|---|
| PairField | 42 | 0.388 | 0.373 | 0.136 | +0.928 |
| NarrativeEngine | 42 | 0.263 | 0.282 | 0.116 | +0.869 |
| OscillatorLaser | 43 | 0.167 | 0.163 | 0.089 | +0.731 |
| **QuantumEngine** | **43** | 0.141 | **0.113** | **0.126** | **−0.051** |

`QuantumEngine` at seed 43 has `cv(cut) < cv(diff)`, and `corr(Φ, cut) = −0.051`
against `corr(Φ, diff) = +0.909` — Φ's variance is carried by differentiation
there, not the cut. So **`cv(cut) > cv(diff) > cv(cplx)` is 31/32, not universal.**

The partition still wanders on that run (12 distinct sizes, k = 4–16), so the
mechanism is present; it is simply not the dominant term. And the single
counterexample sits on a **passing** seed — exactly where a claim assembled from
failing seeds would never have looked.
