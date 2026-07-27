# QD-10: What the system can and cannot understand about a combination

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**Owner-requested evaluation, run after the fact. No pre-registered hypotheses
and no verdict.** Reproduce with `bench_combination_understanding.py`.

Goal: does the system understand a combined concept? Answered in two parts,
because the two senses of "understand" have different answers.

## Part 1 — semantic understanding is blocked, and the blocker is the corpus

Understanding what `얼음+불` *means* requires a semantic input. This repo has
exactly one candidate, `data/corpus_v2.txt` at 70MB, and it cannot serve.

| section | lines | unique lines | note |
|---|---|---|---|
| arithmetic drills | 428,711 | 426,547 | **41.7% of the corpus** |
| Korean dialogue | 253,025 | **133** | 99.95% duplicate |
| English dialogue | 107,373 | **57** | 99.95% duplicate |
| Korean prose | 115,610 | 51,606 | |
| English prose | 108,731 | 48,452 | |

The dialogue is 360,398 lines and **190 unique ones** — one block repeated 2538
times. And the prose that survives deduplication is not language either:

| | tokens | word types | type/token | natural range |
|---|---|---|---|---|
| Korean prose | 870,211 | **2,074** | 0.24% | — |
| English prose | 1,223,950 | **1,728** | **0.14%** | 1–3% |

An order of magnitude too few word types. The tells are everywhere: the top
Korean bigrams each occur **exactly 3000 times** (`둘이+합치면`, `합치면+개이고`),
the English has `together_they(3000)`, and one prose line is
`느낌 느낌 느낌 느낌 느낌 느낌 느낌 느낌`.

**No part of this 70MB is natural language.** Semantic understanding of a
combination is therefore not merely unmeasured here — there is nothing for it
to be made of. This is a data defect, not an architecture one, and it sits
upstream of ConsciousLM and AnimaLM as much as of anything tested in this series.

## Part 2 — structural understanding, which is testable and partly there

Without semantics, one real sense of understanding survives:

> a representation understands a combination if the combination alone is
> enough to say what it was combined from.

Fourteen words, 182 ordered pairs:

| encoding | ① recover (A,B) | chance | ② locate the seam | chance | ③ order gap |
|---|---|---|---|---|---|
| bag | **50.0%** | 0.55% | 2.7% | 20.0% | 0.0000 |
| bigram | **22.5%** | 0.55% | 15.4% | 20.0% | **0.8131** |
| qualia_sense | 18.7% | 0.55% | 50.0% | 20.0% | 0.0000 |

**① The combination does say what it was made of** — every encoding is far
above the 0.55% chance level. `bag`'s 50.0% is exact and interpretable: pooling
characters makes `AB` and `BA` identical, so the unordered pair is recovered
perfectly and the order is a coin flip. Precisely half.

`bigram` scores lower (22.5%) despite being the only encoding that *can* tell
`AB` from `BA` (gap 0.8131). Its reconstruction `enc(a) + enc(b)` differs from
`enc(a+b)` by the junction and boundary corrections (QD-9), so the true pair is
no longer an exact match and sometimes loses to a neighbour. **Carrying order
and reconstructing cleanly are in tension here.**

**② Nothing locates the seam.** `bigram` reaches 15.4% against a 20% chance
level — below it. The system can say *what* was combined and not *where the
parts met*.

### Two caveats on this table

- **`qualia_sense`'s 50% on ② is not interpretable.** The seam test composes
  candidates as a plain sum `enc(x) + enc(y)`, which is the right model for the
  pooled encodings and the wrong one for `qualia_sense`, whose features are
  ratios and compose as a length-weighted mean (see
  `bench_concept_combination.predict_concat`). Its number reflects a mismatched
  composition rule, not an ability. Left in the table with this warning rather
  than removed.
- **The first version of ① was invalid.** It compared `enc(a+b)` against a
  table also built from `enc(a+b)`, which is an identity lookup with distance 0
  — `bag` and `bigram` both scored 100% on no evidence. The table is now built
  from the parts.

## Where this leaves the goal

| sense of "understand a combination" | status |
|---|---|
| says what it was made of | **yes** — 50.0% / 22.5% against 0.55% chance |
| tells `AB` from `BA` | **yes, for `bigram` only** — 0.8131 |
| locates where the parts met | **no** — below chance for every encoding |
| knows what the combination *means* | **blocked** — no natural language in the repo |

The structural half is partly achieved and measured. The semantic half is not
achieved, and the reason is now a specific, fixable, upstream fact: **the
training corpus contains no natural language.** Until that changes, no
architecture in this repo can understand a combined concept, because none of
them has ever been shown what a concept is.
