# Lineage depth conditional on exact cell age

Reproduction:

```bash
.venv/bin/python bench_lineage_age.py
```

Configuration: seeds 42–49, 1,200 steps, collect after step 600 while the grown
population has at least 16 cells, ceiling 32, `cell_dim=64`, `hidden_dim=128`.

The comparison stratum is `(seed, engine step, exact cell age)`. Each higher
depth is compared with each lower depth present in that stratum. Log tension
ratios are divided by the lineage-depth gap, weighted by the smaller group, and
aggregated within seed. Inference uses eight seed summaries rather than treating
correlated cell-step readings as independent.

Result: 150,717 observations, 16,280 comparable strata and 22,155 contrasts.
The weighted geometric ratio per lineage level has seed median 1.0269 and range
0.8516–1.2849. Five of eight seeds are positive; two-sided exact sign `p=0.7266`.
The prior uncontrolled 4.9× depth gradient therefore does not survive fixing
age. Tension tracks recency of division, not accumulated lineage depth.
