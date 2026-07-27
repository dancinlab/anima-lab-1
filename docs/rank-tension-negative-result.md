# What I would change in the engine, and the measurement that refused it

## The diagnosis holds

Every defect measured this session shares one root: **a raw magnitude compared
against a bare constant, with no shared reference.**

| defect | the comparison |
|---|---|
| `split_threshold=0.3` unreachable | `(output**2).mean()` peaks at 0.037 |
| the same 0.3 on two engines | right in one, 8× too high in the other |
| population runs to the ceiling | nothing links population size to tension |
| child weights inflate on division | +27.9% per generation, a runaway |
| `cell_tension = 0.5` in an `else` | a constant above the bar drove every split |
| ethics gate always opens | two constants compared against a third |

Fix the root and five of the six become impossible to write.

## The proposal — tension as rank within the population

```
tension_i = (cells with a smaller raw response) / (n − 1)
```

Four properties would follow from the definition rather than from tuning:
scale-free (scaling every input changes no ordering), never unreachable (a
quantile always has cells above it), population feedback for free (top 20% of 30
is a harder club than of 3), and order-independent (unlike
`consciousness_engine`'s "mean of the cells so far").

## It does not work. Measured twice.

**First run** — rank plus the population-scaled bar already landed in `0597e24`:

| bar | cells | splits |
|---|---|---|
| 0.5 | 2.9 | 22 |
| 0.7 | 2.0 | **0** |
| 0.8 | 2.0 | **0** |

The scaling multiplies the bar by `n / min_cells` without bound. A rank is
capped at 1.0, so past `n = 2/bar` the effective bar exceeds every possible
value and **division becomes arithmetically impossible**. Two changes I made at
different times are incompatible.

**Second run** — scaling removed, since a rank already carries the feedback:

| bar | cells | splits |
|---|---|---|
| 0.5 | **32.0** | 68 |
| 0.7 | **2.0** | 0 |

A cliff again — the ceiling or the floor, nothing between. And the reason is in
the definition: **at the 2-cell start the only ranks are 0 and 1**, and which
cell holds the 1 flips with the input, so a 3-step mean sits near 0.5 and any
bar above it never fires while any bar below it fires forever.

Scale invariance also failed to arrive: 2.0 → 8.7 cells under a ×10 input, a
333% drift. Better than raw magnitude's 967%, and nowhere near the 0% the
argument promised — once the split trajectory diverges, the populations differ
and the ranks are computed over different sets.

## What survives

- **The diagnosis.** All six defects are the same shape, and that is worth
  designing against.
- **A latent defect in what I already landed.** The population-scaled bar in
  `0597e24` silently assumes tension is unbounded. Any bounded replacement —
  a rank, a probability, a sigmoid — caps the population at
  `max_tension · min_cells / bar`. Annotated at the site.
- **A constraint on any future fix.** A reference-based tension needs a
  population large enough for the reference to mean anything. At `min_cells = 2`
  every population statistic is degenerate: a rank is `{0, 1}`, a z-score is
  `±1`, a percentile is 0 or 100. Whatever replaces the magnitude has to work at
  n = 2 or the engine has to start larger.

## Settling `min_cells = 2` — it answers a different question than I asked of it

`mitosis.py` cites `H297` ("N=2 is optimal starting point") and `CB1`. Only CB1
has a document, `docs/consciousness-threshold-criteria.md:868`:

```
| CB1 Critical cell count | 2.384 | — | 최소 2개 세포 필요 (1개로는 Φ>1 불가) |
```

That is a **floor below which consciousness cannot exist**. It is not a claim
that population statistics mean anything at n = 2, and I had been reading it as
one. The floor and the starting point are separable, and conflating them is what
made the constraint above look fatal.

So: start above the floor and measure. **The population froze at exactly the
starting size** — 4.0, 8.0, 16.0 cells from starts of 4, 8, 16, for every bar
from 0.7 to 0.9. Meaningful statistics did not restore division.

## The real blocker was the axis, not the quantity

`split_patience` asks a cell to stay above the bar for several **consecutive
steps**. With eight stimuli rotating, whichever cell responds hardest rotates
with them, so no cell holds the top for three steps running.
**Persistence-over-steps and variation-across-steps are in direct conflict**,
and no choice of quantity resolves it — QD-6 hit the same wall from the other
side, where a z-score against each cell's own history never fired because a
fixed stimulus makes that history flat.

Aggregate over the **stimulus** instead of over steps, and the criterion becomes
what specialisation actually means:

```
a cell divides when it is consistently extreme FOR SOME PARTICULAR STIMULUS
```

`bench_specialization_split.py`, starting at the floor of 2 so nothing is handed
to it:

| | rank + steps | rank + stimulus |
|---|---|---|
| bar 0.7 → 0.95 | 32.0 ↔ 2.0, a cliff | 9.0 → 13.0, **1.56× over the whole range** |
| ceiling 16 → 64 | — | 11.3 / 14.0 / 14.0, **1.24× against a 4× ceiling** |
| input ×10 | 333% | **79%** (raw magnitude: 967%) |

First configuration measured this session that settles **between** the floor and
the ceiling under varying input, and stays there when the ceiling moves four-fold.

**Scale invariance still did not arrive**, and the promise was 0%. A rank erases
the units but not the network's nonlinear response to scale — at ×10 input, which
cell leads changes, so the ranks are computed over a different ordering. 79% is
an improvement, not the invariance the argument claimed.

## On real language it fails completely — and names the actual defect

The ING record said synthetic vectors could refute a design but not land one,
and that real corpus stimuli were the gate. `bench_specialization_corpus.py`
runs it, on 24 sentences sampled from `data/corpus.txt`:

| bar | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|
| synthetic | 9.0 | 9.3 | 14.0 | 13.0 |
| **real corpus** | **2.0** | **2.0** | **2.0** | **2.0** |

The floor, at every bar. Ownership never forms more than one group, so the
coherence test — is the specialisation meaningful or just a partition? — could
not run at all. Starting at 4, 8 or 16 cells does not help: the population
**collapses back to 2** from every start.

That collapse is the finding. Same code, same settings, 400 steps from 16 cells:

| | splits | merges | final |
|---|---|---|---|
| synthetic randn | 8 | 6 | 18 |
| real corpus | 7 | **21** | **2** |

**Split rates barely differ. The merge side decides everything**, and I had spent
the entire investigation on the split side.

`_check_merges` is `all(t < 0.05 for t in recent)` — a raw magnitude against a
bare constant, the exact shape this document opens by diagnosing, mirrored.
Inter-cell tension measured against that bar:

| | median | p90 | below 0.05 |
|---|---|---|---|
| real corpus | 0.0148 | 0.0205 | **100.0%** |
| synthetic randn | 0.0830 | 0.1336 | 9.3% |

The bar sits 2.4× above the 90th percentile of what real language produces, so
every pair is permanently "redundant" and only `merge_patience = 10` and the
`min_cells` floor prevent total collapse. This is `split_threshold = 0.3`
against a 0.037 peak, in mirror image.

Why real text and not synthetic: 24 corpus sentences under `text_vector` have
mean pairwise cosine **+0.191** against randn's **+0.014** — 14× closer. Cells
fed near-identical inputs stay near-identical. **The per-stimulus design's
success was an artifact of stimuli that were nearly orthogonal**, and nothing in
its argument said so.

## Both sides relative — and the answer is not about the bar at all

Fixing one half of the ledger is what produced the previous result, so
`bench_relative_population.py` makes both halves relative at once: split on a
cell's rank among cells, merge on a pair's rank among pairs. Registered before
running (P1 convergence, P2 the risk that quantiles decide, P3 the separator).

Everything collapses to the floor:

| 분열/병합 quantile | narrow | broad | synthetic |
|---|---|---|---|
| 0.80 / 0.10 | 2.0 | 2.0 | 2.0 |
| 0.90 / 0.15 | 2.0 | 2.0 | 2.0 |
| 0.95 / 0.25 | 2.0 | 2.0 | 2.0 |

**P1 "passed" and the criterion was badly designed** — "real and synthetic within
4 cells" is satisfied by total collapse. Pre-registering P3 as the pass condition
is what caught it: NARROW (internal cosine +0.435) and BROAD (+0.037) both settle
at 2.0, so the population cannot tell a homogeneous stimulus set from a diverse
one. P2 confirmed.

The reason is structural, not tuning. **A rank puts the bottom item at exactly
0.00 and the top at exactly 1.00, by construction.** So "bottom 15%" is
unconditionally true of some pair and "top 10%" unconditionally true of some
cell — both rules fire forever, and the one that is easier to satisfy wins.
Merges outnumbered splits 13.3 to 7.3.

**A rank erases "how much", so it cannot say "nobody is redundant."**

## The signal is blind to what the criterion is supposed to respond to

A z-score keeps magnitude and can say that, so before building it: measured on
the same narrow/broad sets, per step, the lowest pair's z —

| | median min-z | p5 | share below −2 |
|---|---|---|---|
| narrow (+0.435) | −1.90 | −2.13 | **33.3%** |
| broad (+0.037) | −1.62 | −2.33 | **33.3%** |

Identical firing rate. Not worth building, and the check cost one run instead of
an engine.

That pointed one level down — is it normalisation destroying the information, or
was it never there? Raw inter-cell tension, no normalisation of any kind:

| | median | mean | p10 |
|---|---|---|---|
| narrow (+0.435) | 0.01083 | 0.01122 | 0.00421 |
| broad (+0.037) | 0.01070 | 0.01097 | 0.00421 |

**0.99×.** Internal similarity differs 12-fold; inter-cell tension does not move.

So no threshold on inter-cell tension — absolute, calibrated, rank, quantile or
z-score — can regulate population against the diversity of the input, because
the quantity does not carry that information. Six framings of the bar were
tried; the bar was never the thing.

What it does carry: with the population pinned at 8 cells, variation across
stimuli within a pair is 0.00197 against 0.00279 across pairs, a ratio of 1.4×.
That is **not** enough to say weights dominate, and the tempting story that it
measures weight difference rather than input difference is not supported.
The accurate statement is narrower and is what the measurements show: inter-cell
tension responds to **which** stimulus arrived, and not to **how diverse the
stimulus set is**.

## What this leaves

- **The diagnosis holds.** All six defects are the same shape.
- **The latent defect I landed stands.** The population-scaled bar in `0597e24`
  assumes unbounded tension; any bounded replacement caps the population.
  Annotated at the site.
- **`H297` has no evidence document.** CB1 does and says something narrower than
  the citation implies. Flagged, not silently trusted.
- **`merge_threshold = 0.05` is a seventh instance of the root defect**, and the
  one that actually governs population on real data. Annotated at the site.
- **Nothing landed in the engine.** The per-stimulus split does not survive its
  own gate, and the merge bar is a design decision that belongs with the split
  bar rather than applied to half the ledger.
- **A method note worth more than the design.** Synthetic stimuli made a broken
  configuration look regulated across three independent properties. The
  properties were real; the stimuli were not representative. Any future
  population mechanism gets measured on corpus text before it is believed.
