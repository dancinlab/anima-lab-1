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

## What I would not do

Reach for the next variant. Six mechanisms failed in the QD series and two more
here, all measured, and the pattern is that a fix aimed at the symptom finds a
new edge each time. The honest next step is the one this refutation names: decide
what `min_cells = 2` is for. Every reference-based quantity is degenerate there,
and that is upstream of the choice of quantity.
