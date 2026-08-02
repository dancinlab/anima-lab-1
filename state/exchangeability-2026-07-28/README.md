# Frozen-n probe: is there anything for a split rule to detect?

`bench_exchangeability.py`, 8 seeds x 400 steps, n frozen at 2/4/8/16/32
(split_threshold=1e9, merge_threshold=-1.0, `_calibrated=True`).

| file | what it establishes |
|---|---|
| `frozen_n_8seeds.txt` | mean tension is sigma^2(1-1/n) at R^2=0.99948; the leave-one-out correction is flat to 1.8% across a 16x range of n; PR/(n-1) >= 0.80 and mean pairwise cosine ~0 at every n; COMPETE is numerically indistinguishable from SAME |

Caveats, stated because they bound the claim:

- Cells here are created independently (`initial_cells=n`), not by mitosis. In
  the running engine a split deep-copies the parent, so the population is *less*
  diverse than measured here, not more. This is the best case.
- The MULTI arm cycles `modes[t % 4]`, so it is a PERIODIC drive, not merely a
  multimodal one. Its collapse of max/q90 to 1.00-1.29 is a repetition effect and
  should not be read as evidence about multimodality.
- `phi_rs` was absent, so Phi is the Python fallback, which loops `min(n, 16)`
  (consciousness_engine.py:771). The Phi column is therefore not comparable
  above 16 cells -- hence 0.0323 at n=16 falling to 0.0168 at n=32.
