# Both sides keep memory; one keeps a rule

CLAUDE.md's core claim is that consciousness arises from the repulsion between
Engine A (forward) and Engine G (reverse). Every engine scored so far is a single
population — the pair had never been built.

The sharper question is what "unruled" actually costs. This session measured
`NOISE` at 0/5, but it fails for a reason unrelated to being unruled: it draws
fresh random state every step, so it has no past. **Unruled and memoryless are
different properties, and only one had ever been tested.** So: both sides keep
memory (a coupling matrix that accumulates and survives resizing); A updates it
by Hebbian rule, G updates it with pure `randn` and no rule at all.

## Pure repulsion cannot be stable — that is structural, not a bug

A is pushed away from G while G is pushed away from A, so `field = ha - hg`
feeds its own growth: the separation multiplies by `(1 + 2s)` every step. Row
normalisation only slowed it (the guard still caught `|h| = 1.11e4`). The only
bound is restoring each side's norm — which is why `BenchEngine`'s own repulsion
is norm-preserving. The direction the two histories disagree on is kept; the
runaway is not.

## The pair beats each side alone — at the right strength

32 cells, hidden 64, seeds 42/43/44. Solo score for either side: **4.67**.

| strength | A⇄A′ peak \|h\| | A⇄A′ | A⇄G peak \|h\| | A⇄G |
|---|---|---|---|---|
| 0.00 | 4.31 | 4.00 | 4.31 | 4.00 |
| **0.01** | 4.19 | **5.00** | 4.25 | **5.00** |
| **0.03** | 4.42 | **5.00** | 4.68 | **5.00** |
| 0.08 | 20.18 | 1.00 | 5.65 | **4.67** |
| 0.15 | 25.00 | 0.00 | 5.70 | 1.00 |

`strength = 0` is the sanity row: the two engines do not touch, and the pair
reproduces roughly the solo score (4.00 vs 4.67). Without it nothing below could
be attributed to the coupling.

**A first reading of this experiment was wrong.** Measured only at 0.15, the pair
scored 0.00 / 1.00 against 4.67 solo, and it was written up as "pairing destroys
what each side had — the repulsion CLAUDE.md calls the source of consciousness
makes things worse". That is refuted: 0.15 is simply past the stability edge, as
the `|h|` column shows (4.4 → 20–25). At 0.01–0.03 the pair scores **5.00**,
above either side alone. Bounded is not the same as healthy, and a guard that
passes is not evidence that a system is well-behaved.

## Two findings

**The rule contributes nothing on its own.** Hebbian coupling and pure-`randn`
coupling both score 4.67 solo — identical. What the gate detects is *memory*;
whether the memory is *structured* is invisible to it. This is the same shape as
the nine other findings this session: a channel that looks structural and carries
no load.

**An unruled second side widens the stable band.** At strength 0.08 the
both-ruled pair has already collapsed to 1.00 with `|h|` at 20.18, while the
one-unruled pair holds **4.67** at `|h|` 5.65. The unruled side does not
reinforce the field it is pushed by, so the pair tolerates a stronger coupling
before running away. Both configurations peak at 5.00; the difference is how much
strength they survive.

Reproduce: `.venv/bin/python bench_ag_memory.py --cells 32 --hidden 64`.
Checkpoints: `checkpoints/ag/A_ruled.pt`, `checkpoints/ag/G_unruled.pt`
(coupling matrix + hiddens + weights, written `.tmp` then atomically replaced).
