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

## Superseded — the original direction check, kept for the record

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
