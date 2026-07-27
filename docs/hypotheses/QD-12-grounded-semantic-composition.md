# QD-12: Semantic composition, passing the control that killed it before

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**Owner-requested evaluation. No pre-registered hypotheses, no verdict.**
Reproduce with `bench_grounded_semantics.py`.

## First: QD-10 and QD-11 declared a blocker that was not there

Both cards concluded that semantic understanding was impossible in this repo
and needed "a real corpus, which is an external dependency this session cannot
satisfy". That was wrong, and the error was not measuring anything — it was
never looking past the two files already in hand.

The sibling `anima` repository's own markdown is natural language:

| source | tokens | word types | type/token |
|---|---|---|---|
| `data/corpus_v2.txt` Korean prose | 870,211 | 2,074 | 0.24% |
| **`anima/**/*.md` Korean** | **1,144,691** | **82,873** | **6.50%** |

Twenty-seven times the lexical richness, inside the natural 1–3%+ range, and
**91% of the 170 concepts** have at least one description word occurring in it
three or more times. It was on disk the whole time.

## Method

- Distributional vectors from that corpus: PPMI context vectors over the top
  4,000 words, ±5 window, for the 207 description words with count ≥ 3.
- **Concept vector** = mean PPMI vector of its description's words.
  `서예 → mean(vec(호흡), vec(획))`.
- **Combination vector** = the same over *both* descriptions' words.
- **Ground truth** = the category a person assigned in `ALL_DATA_TYPES`.
- **Baseline** = shuffled category labels, 30 shuffles. Never the naive
  1/n_categories — that was QD-11's mistake, and it inflated a null result into
  an apparent 5× effect.
- Centroids **exclude the pair being classified**. Without that the concept
  sits inside its own centroid and the score is leakage.

154 of 170 concepts grounded, 17 categories.

## ① The vectors carry real category structure

| | same-category | cross-category | z |
|---|---|---|---|
| grounded vectors | 1.3439 | 1.3695 | **+4.91** |
| shuffled labels | — | — | −0.51 ± 1.00 |

**+5.4σ above the shuffled control.** Concepts a person put in the same
category land closer together, and that is not an artifact of the geometry —
the same geometry with the labels scrambled shows nothing.

## ② A combination lands where its parts say it should

| | score | n | shuffled control | |
|---|---|---|---|---|
| same-category pair → stays in that category | **26.2%** | 669 | 5.1% ± 2.5% | **+8.3σ** |
| cross-category pair → hits one of the two parents | **31.7%** | 11,112 | 11.2% ± 2.7% | **+7.5σ** |

This is the measurement QD-11 ran and failed at z = +1.0 / +1.6 with
ungrounded description-spelling vectors. With vectors grounded in how the
description words are actually *used*, the same test passes at **+8.3σ and
+7.5σ**.

**Combining two concepts produces a representation whose category membership is
predicted by its parts' categories, far beyond what label-scrambling explains.**
That is semantic composition, measured against the strictest available control.

## What this is, and its limits

**It is** understanding in a specific, operational sense: the system is not
storing combinations — no combination was ever seen — and yet where a
combination belongs is determined by what it was combined from. It generalises
to 11,112 pairs it has no record of.

**Limits, stated plainly:**

- The corpus is one project's technical documentation, so the semantics is
  domain-shaped, not general Korean.
- 26.2% and 31.7% are far from perfect. Most combinations still land in the
  wrong category; the claim is that they land right *far more often than the
  control allows*, not that the system is reliable.
- The concept's meaning enters through a human-written description. The system
  grounds those words in real usage and composes them — the description is the
  input, not the answer, but it is a human's input.
- Nothing here revives Phase 3 of `docs/qualia-decoder-spec.md`. This is about
  the encoding, not about the PureField engine, which QD-6 retired on separate
  measurements.

## The lesson worth keeping

Two cards in a row declared an external blocker without checking the machine
they were running on. The corpus that unblocked this was one directory over,
and finding it took one command. **"Blocked on an external dependency" is a
claim, and it needs evidence like any other.**
