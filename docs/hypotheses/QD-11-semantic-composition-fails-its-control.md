# QD-11: Two results that did not survive their controls

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**Owner-requested evaluation. No pre-registered hypotheses, no verdict.**
Reproduce with `bench_semantic_composition.py`.

Goal: does the system understand a combined concept, semantically. QD-10 said
the corpus could not supply meaning. This card found a second source, tested it
properly, and reports that the result failed.

## The second semantic source

`ALL_DATA_TYPES` gives every one of 170 concepts a human-written one-line
Korean description and a human-assigned category — `서예 → ("✒️", "호흡과 획")`
in 예술, `매운맛 → ("🌶️", "통증의 쾌락")` in 미각. Small, but authored by a
person about what the thing means. Real ground truth, unlike the corpus.

## Result 1 — the apparent category structure is a form confound

| encoding | same-category | cross-category | gap z |
|---|---|---|---|
| meaning (description bigrams) | 1.3591 | 1.3866 | 2.67 |
| form (name bigrams) | 1.3809 | 1.3876 | 0.66 |
| **form (`qualia_sense`)** | 0.4281 | 0.5228 | **11.12** |

`qualia_sense` measures length, jamo density and character dispersion — it has
no access to meaning — yet it separates categories four times more sharply than
the description-based encoder. The reason is that the categories are built from
form-similar words:

```
미각   매운맛 단맛 쓴맛 감칠맛 신맛      ← all end in 맛
색깔   빨강 파랑 노랑 초록 보라 검정     ← all exactly 2 characters
```

**26% of concepts share their category's most common final character, against
~1% by chance.** The z = 11.12 is spelling, not semantics. Any claim of
semantic structure from a form encoder on this stimulus set is confounded.

## Result 2 — semantic composition fails its shuffled-label control

Combining two concepts' descriptions and classifying the result by nearest
category centroid:

| | same-category pairs stay | cross-category pairs hit a parent |
|---|---|---|
| real labels | 28.9% | 27.4% |
| naive chance (1/17, 2/17) | 5.9% | 11.8% |
| **shuffled labels — the real control** | **26.4% ± 2.4%** | **25.3% ± 1.3%** |
| | **z = +1.0** | **z = +1.6** |

Against the naive chance level this looked like 5× performance. Against
shuffled labels — the same geometry with the semantics removed — it is
**not significant**. The score comes from how centroids partition the space,
not from the labels being correct.

**The naive baseline was the wrong baseline.** 20 shuffles cost seconds and
overturned the conclusion; reporting "28.9% against 5.9% chance" as semantic
composition would have been wrong.

## Where the goal stands

| sense of "understand a combination" | status |
|---|---|
| says what it was made of | **yes** — 50.0% / 22.5% against 0.55% (QD-10) |
| tells `AB` from `BA` | **yes, `bigram` only** — 0.8131 (QD-10) |
| locates where the parts met | **question retracted** — not in the input |
| knows what the combination means | **no** — fails its control, twice over |

The semantic half is not achieved, and it is now blocked for two independently
measured reasons rather than one: the corpus contains no natural language
(QD-10), and the only other semantic source in the repo is too small and too
form-confounded to carry a composition result past a shuffled-label control.

**The concrete blocker is data.** Nothing in this repository has ever shown any
of these systems what a concept means. That is not fixable by architecture, by
a further encoding, or by another experiment on the present inputs — it needs a
real corpus, which is an external dependency this session cannot satisfy.
