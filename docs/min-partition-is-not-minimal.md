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
