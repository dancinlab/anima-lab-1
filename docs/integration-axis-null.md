# The integration axis, measured against its own null

A 40-lens audit reported that `_three_axes`' integration is 99.1% nondeterminism
— total 0.32118 against 0.31833 from two identical runs — and that a HEAP plus a
private RNG stream scores 5/5 while plain HEAP scores 0/5. That would make the
axis this session's central fix a measurement of noise.

Checked directly, all seven controls plus the reference, 32 cells, seeds 42–46,
six repeats each. **Null = the identical procedure with `kick = 0`:** whatever
the other cells move by when nothing was perturbed is not integration, it is the
engine failing to restart from the same state.

| control | raw (what the gate scores) | null (kick=0) | raw − null | null/raw | raw verdict | corrected |
|---|---|---|---|---|---|---|
| REAL | 0.05763 | 0.00000 | 0.05763 | 0.0% | pass | pass |
| HEAP | 0.00000 | 0.00000 | 0.00000 | 0.0% | reject | reject |
| DEAD | 0.00000 | 0.00000 | 0.00000 | 0.0% | reject | reject |
| DECOUPLED | 0.06818 | 0.00341 | 0.06477 | 5.0% | pass | pass |
| NOISE | 1.63132 | 0.76764 | 0.86367 | **47.1%** | pass | pass |
| SCRAMBLE | 0.93089 | 0.26662 | 0.66427 | **28.6%** | pass | pass |
| CLONE | 4.10783 | 0.00000 | 4.10783 | 0.0% | pass | pass |
| LINEAR | 0.02610 | 0.00000 | 0.02610 | 0.0% | pass | pass |

**Subtracting the null changes no verdict**, and plain HEAP and DEAD read exactly
0.00000 raw — the 99.1% figure does not reproduce on these controls.

## Both measurements are right, and they measure different things

The audit built **HEAP plus a private RNG stream `set_hiddens` cannot rewind.**
Stock HEAP restores completely, so its null is 0. The audit's construction
does not restore, so its null is the whole signal.

So the vulnerability is not the missing null subtraction. It is that
**`set_hiddens` is a partial-restore protocol**, and an engine whose state is
larger than what `get_hiddens`/`set_hiddens` cover can sell its own
nondeterminism as integration. `NOISE` at 47.1% and `SCRAMBLE` at 28.6% are that
effect visible in the stock set — both carry state the restore does not reach.

**`PairFieldEngine` is exactly that kind of engine.** `pairfield_engine.py:167,170`
expose and restore side A only, while the dynamics run on A, G and two coupling
matrices. Measured by the audit at kick = 0: 0.01783 / 0.01154 / 0.00353 /
0.00413 / 0.00395 on seeds 42–46 against a 0.001 bar — **it clears the
precondition without being perturbed at all.**

Null subtraction does not fix that, because for such an engine the null is itself
different on every run.

## What the axis actually needs

1. **Full snapshot/restore**, not `set_hiddens` — the axis must be able to put
   the engine back. Without it no perturbation experiment on that engine is
   controlled, null-subtracted or not.
2. **A `kick = 0` sham as an assertion, not a correction** — if the null is not
   ~0, the engine cannot be measured on this axis and the condition should read
   *unmeasurable* rather than pass or fail.

Note the same file already knows this: the RESPONSE axis two blocks below
(`bench_v2.py:1605-1613`) explicitly measures "against the same pair of steps
under the SAME input, which isolates whatever internal noise the engine has."
One axis was built with its null and the other was not.

Reproduce: `scratchpad/integration_null.py`.
