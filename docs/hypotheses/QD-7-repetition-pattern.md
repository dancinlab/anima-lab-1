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
