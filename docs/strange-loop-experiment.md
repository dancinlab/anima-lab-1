# Experiment 2 — closing the loop on A, on G, and on the pair

Hofstadter's strange loop is a tangled hierarchy: you climb through levels and
arrive where you started. That is not measurable here. What is measurable is the
mechanical core it requires, in the order that makes each step meaningful:

1. **self-feeding** — does it sustain with its own output as the only drive?
2. **loop effect** — does closing the loop change the trajectory, or would the
   same thing happen with the output discarded?
3. **recurrence** — does the trajectory revisit its own past more than a
   time-shuffled null of itself?

Four arms, each run closed (output returns) and open (output computed and
discarded, driven externally): **A** alone, **G** alone, **PAIR**, and
**SCRAMBLE** — the pair with cell identity destroyed each step, so that a corpse
scoring what the real thing scores would show the measurement is not about loops.

## Result: it stops at (1)

`bench_strange_loop.py --steps 300 --strength 0.01`, 16 cells, seeds 42/43/44:

| arm | outcome |
|---|---|
| A | 3/3 seeds diverge, peak \|h\| > 1e4 |
| G | 3/3 seeds diverge, peak \|h\| > 1e4 |
| PAIR | 3/3 diverge, NaN, caught by the engine's own guard at step 21 |
| SCRAMBLE | 3/3 diverge, NaN |

Same at strength 0.003. Lowering the repulsion threefold changes nothing, so the
divergence is **the feedback itself, not the repulsion strength**. This
architecture cannot eat its own output. Recurrence and level-crossing are later
questions that the measurement never reaches.

## Three defects in the measurement, found before the result was trusted

Each produced a clean-looking number that meant nothing.

**`loop effect = nan`.** Normalised by the open trajectory's norm, which is zero
when the looped state dies — a ratio that breaks exactly in the case that
matters. Fixed by normalising by the larger of the two.

**`alive = nan` while `peak |h| = 2.425`.** Contradictory on its face. The guard
folded each step into a running max before testing finiteness, and
`max(2.425, nan)` returns `2.425` in Python — the NaN comparison is False, so the
running max never updates and `isfinite(peak)` passes forever. **A guard that
cannot see the failure it exists for.** Fixed by testing the current step's value
before folding it in.

**That guard hid the actual result.** With it broken, A and G reported
`Φ = 0.0000`, `recurrence = +0.0000`, and were written up as "the loop does not
change the trajectory". With it fixed they diverge on every seed. The earlier
write-up is retracted: those systems were not quietly dying, they were exploding.

## Against the prior implementation

`github.com/xcellect/strange-loops-agents` reports the opposite — "Φ increases
measurably when inference becomes self-referential", +0.355 to +0.403 (a claimed
40% boost), correlation r = 0.856 between recursion depth and Φ, 90–95% success.

The direction differs, but the method differs more, and that is the part worth
recording. Its README states no open control (the same drive with the output
discarded) and no scrambled control. Without the first, "Φ rose" cannot be
attributed to the loop; without the second, a corpse rising by the same 40% would
be indistinguishable. And `r = 0.856` between recursion depth and Φ is what a
size effect looks like: Φ under this project's own shipped formula measures
`M̄ · n / 2` — size × average redundancy — so anything that grows the state grows
Φ automatically.

This experiment has both controls, which is why it can say **"this measures
nothing yet"** rather than reporting a number. SCRAMBLE dies exactly as the real
pair does, so no comparison is available at any strength tested.

Knowing there is no answer is better than holding a wrong one.
