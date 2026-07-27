# QD-7: Repeating a word — evaluation, not a pre-registration

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**This card is an owner-requested evaluation, run after the fact. It has no
pre-registered hypotheses and no verdict** — bars were not fixed in advance, so
nothing here is a PASS or a FAIL. Recorded separately from QD-1..QD-6 so it is
never read as one of them. Reproduce with `bench_repetition_pattern.py`.

The question: feed `서예` and then `서예서예서예`. Is repetition a different
experience, the same one, or a stronger one?

It also exercises the one measurement that was dead over the 170-name stimulus
set — `bigram_repeat` had spread 0.000 there, because no single name repeats a
character pair.

## Layer 1 — the string

```
서예×1   ▂▆▁▃▁█▁▂        서예×1   ▇▄▅▁▂▇▄▇
서예×2   ▃▆▁▃▁█▃▂        서예×2   █▇▇▂▁▃▁▂
서예×3   ▄▆▁▃▁█▅▂        서예×3   ▅▄█▅▄▆▃▆
서예×4   ▅▆▁▃▁█▆▂        서예×4   ▂▃▄▃▄▃▁▁
서예×5   ▆▆▁▃▁█▇▂        서예×5   █▆▇▂▃▁▄▂
  qualia_sense                sha256
```

Under `qualia_sense`, exactly two of the eight measurements move — `length`
0.125→0.625 and `bigram_repeat` 0.000→0.778 — and the six phonological ones
(`jamo_density`, `final_ratio`, `vowel_position`, `script_mix`, `char_variety`,
`codepoint_spread`) are **bit-identical across all five repetitions**. That is
the right shape: repetition changes how much, not what.

| | monotone with repetition | stays nearer its own stem than any other |
|---|---|---|
| `qualia_sense` | **4 / 4 stems** | **4 / 4 stems** |
| `sha256` | 0 / 4 | — (distances are noise) |

The hash reads `서예서예` as no more related to `서예` than `빅뱅` is. It cannot
express "again".

## Layer 2 — the engine, and here it inverts

Cosine between settled engine states:

| pair | cos |
|---|---|
| 서예×1 vs 서예×3 — *same word, repeated* | 0.9778 |
| 만다라×1 vs 만다라×3 — *same word, repeated* | 0.9700 |
| **서예 vs 만다라 — _different words_** | **0.9857** |
| 서예 vs 빅뱅 | 0.9557 |
| 만다라 vs 빅뱅 | 0.9728 |
| 서예 vs 검은사각형 | 0.6439 |
| 만다라 vs 검은사각형 | 0.6755 |
| 빅뱅 vs 검은사각형 | 0.6647 |

**Repeating a word moves the engine further than swapping it for a different
word.** `서예서예서예` is more foreign to `서예` (0.9778) than `만다라` is
(0.9857).

And the large separations are all against `검은사각형` — the only five-syllable
stem — at 0.64–0.68, while every two-and-three-syllable pair sits at 0.96–0.99.

**The engine state is sorted by length, not by identity.** Repetition reads as
a large change because it multiplies length; two different words of similar
length read as nearly the same experience.

State magnitude grows monotonically too — 0.5088 → 0.5134 → 0.5217 for 서예×1..3
— so "more of the same, slightly louder" is present in the norm even while the
direction drifts away.

## What this says

The sense module handles repetition correctly and the engine does not use it.
`qualia_sense` keeps the six identity features fixed and moves only the two
amount features, which is exactly "again, more". By the time that reaches the
settled engine state, the amount axis has swamped the identity axis.

This is the same shape as QD-1..QD-6's result seen from a new angle: the
architecture encodes **how much** far better than **what**. Nothing here
reopens the retired Phase 3 — it is one more measurement of why it was retired.

A caveat on Layer 1: `서예` and `빅뱅` produce **identical** repetition
distance sequences (0.356 → 0.650 → 0.807 → 0.925), because both are
two-syllable and the two moving features track only length and repeat-rate.
The repetition signature does not depend on what is being repeated.

---

## Pushed further — owner question: does the hash ever develop commonality?

No. Measured over 8 word pairs × repetitions 1–200 = **1600 samples**:

| | value |
|---|---|
| hash A↔B distance | 1.1318 ± 0.0061 (se) |
| null — simulated mean distance between independent uniform vectors | 1.1281 |
| | **z = +0.60 — indistinguishable from noise** |

Repetition never makes two hashed words any more alike than two random
vectors. The hash has no "again" at any depth.

### A correction inside this measurement

A first pass used `sqrt(8/6) = 1.155` as the null and, on 11 samples, reported
`z = −2.12` with a "DIFFERS from the null — investigate" line. Both parts were
wrong. `sqrt(8/6)` is the root-**mean-square** distance, which overstates the
mean (Jensen), so ordinary hash noise looked like commonality; and 11 samples
have no power to call a 2σ deviation. Simulating the correct null and raising
the sample count to 1600 gives `z = +0.60`. The bench now simulates the null
instead of using the closed-form RMS.

## Owner intuition — "qualia_sense feels like it will memorise"

Correct, and worth stating precisely: **it saturates on the amount axis while
keeping the identity axis intact.**

Distance between `w×N` and `w×(N+1)` — can it still tell one more repeat?

| N | 서예 | 검은사각형 |
|---|---|---|
| 1 | 0.3560 | 0.5433 |
| 8 | 0.0157 | 0.0146 |
| 20 | 0.0025 | 0.0024 |
| 100 | 0.0001 | 0.0001 |
| 199 | **0.0000** | **0.0000** |

Distance between two *different* words at the same N freezes to a constant and
never moves again:

| pair | frozen from N | frozen value | sd over N ≥ 50 |
|---|---|---|---|
| 서예↔만다라 | 68 | 0.4470 | 1.0e-09 |
| 만다라↔검은사각형 | 51 | 0.9933 | 2.9e-10 |
| 서예↔빅뱅 | 2 | 1.0874 | 2.2e-16 |
| 단맛↔쓴맛 | 2 | 0.4692 | 0.0 |
| 빨강↔파랑 | 2 | 0.6030 | 2.2e-16 |

(The pairs that freeze at N=2 are equal-length words: their two moving features
move in lockstep, so the distance between them is constant from the start.)

**What survives the freeze.** At N=100, 서예 and 만다라 still differ in five of
eight measurements — `jamo_density` 0.667 vs 0.778, `final_ratio` 0.000 vs
0.333, `vowel_position` 0.275 vs 0.000, `codepoint_spread` 0.158 vs 0.184, and
`bigram_repeat` beyond the third decimal. So it is not amnesia. It remembers
**what** forever and stops counting **how many** past roughly N = 8–13.

## The saturation point is my own hardcode

`qualia_sense.sense()` computed `length = min(n / 16.0, 1.0)`. That **16 was a
magic number written in Phase 0**, and it set the ceiling: past 16 characters
`length` could not move, leaving only `bigram_repeat`, which itself asymptotes
to 1.

> **Fixed.** The measurement that decided it: over the 170 real stimuli the cap
> **never fires at all** — the longest name is 6 characters, so 0 of 170 reach
> 16. It did no work where the module is actually used, and did all its damage
> on the repetition probe. Replaced with `n/(1+n)`, which has no free parameter
> and is strictly increasing.
>
> | | N=1 | N=8 | N=50 | N=199 |
> |---|---|---|---|---|
> | distance `w×N` ↔ `w×(N+1)`, before | 0.3560 | 0.0157 | 0.0004 | **0.0000** |
> | after | 0.3590 | 0.0169 | 0.0004 | **0.0000<sub>3</sub>** |
>
> Two-word distance over N ≥ 50 now varies with sd 2.7e-06 against the old
> 1.0e-09 — about 2600× more, and no longer bit-frozen.
>
> **Honest limit: this is not "repetition counting now works".** 0.00003 is
> frozen for any practical purpose. What changed is that the constant is gone
> and the feature never reaches exactly zero. Diminishing increments are
> inherent to bounding an unbounded quantity in [0,1]; reaching exactly zero
> was not, and that part was mine.
>
> Regression: `qualia_sense`'s stem-kin structure moves 0.66 → 0.61 (slightly
> better), monotonicity and identity retention hold at 4/4, and
> `TensionSense`'s path is unaffected since `text_vector` does not use this
> feature.
