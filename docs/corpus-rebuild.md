# Training corpus — what was wrong and what replaced it

`data/corpus_v2.txt` is 70MB and measurably not language (QD-10). This records
the defect, the rebuild, and the end-to-end check that the result is usable.
`corpus.toml` is the build, split, and evaluation SSOT; regenerate with
`python3 build_corpus.py`. A configured source missing on a host is a hard
error, so the builder cannot silently produce a different corpus there.

## The defect

| | tokens | word types | type/token | |
|---|---|---|---|---|
| v2 Korean (non-arithmetic) | 2,702,000 | 2,427 | **0.09%** | |
| v2 English (non-arithmetic) | 3,424,951 | 2,109 | **0.06%** | |

Natural language sits at 1–3% or above. Alongside that: **41.7% of the file is
arithmetic drills**, and the dialogue section is 360,398 lines containing **190
unique ones** — a single block repeated 2538 times.

## What the rebuild does

1. **Splits arithmetic out** into `data/corpus_v3_arith.txt` (428,711 lines).
   Arithmetic may well be wanted for arithmetic training; at 41.7% of a
   *language* corpus it is 41.7% of the gradient spent on something else.
2. **Deduplicates** every line across all sources.
3. **Caps template families.** Digit masking was not enough — the 3000 copies
   are word problems whose names, objects and verbs all vary
   (`예린은(는) 블록을(를) 76개 가지고 있고 … 둘이 합치면 97개이고`), so each line
   has a unique digit-masked skeleton while `합치면 개이고` still occurred 3000
   times. Families are found by over-represented 6-grams instead, which knows
   nothing about content: no pattern is hardcoded, and the template survives at
   20 copies rather than 3000.
4. **Adds the natural language already on disk.** Two sources were sitting
   unused:
   - `anima/**/*.md` — the sibling repo's own documentation, markdown stripped
     of code fences, tables and links so identifiers do not enter the vocabulary;
   - `data/.corpus_cache/ko_wiki.txt` — real Korean Wikipedia, 180,789 tokens
     over 60,383 types at 33.40% type/token, **already tracked in git** in a
     hidden cache directory.

## Canonical held-out split

The original training scripts cut the final 10% of one ordered file and then
measured one random validation batch. That is not a stable held-out benchmark:
source order determines the distribution, template families can cross the
boundary, and validation advances the same RNG used to draw training batches.

The builder now assigns connected template families as one deterministic hash
group and writes separate files:

| partition | lines | bytes |
|---|---:|---:|
| train | 333,512 | 39,888,552 |
| validation | 36,982 | 4,399,360 |

There are **zero exact lines** shared between the partitions. A deterministic
seeded audit found **20/400 exact 64-byte windows (5.00%)**, below the maximum in
`corpus.toml`. Rebuilding twice produces identical SHA-256 hashes. For
comparison, the NF9 interleaved v2 split measured 82.5% overlap with the same
window width and 400 samples; a sequential final-10% split avoided that exact
reuse but held out a different source distribution. The family-isolated split
addresses both failure modes.

Training now defaults to this pair, rejects a held-out corpus above the overlap
limit, stores both content hashes in every checkpoint, and rejects a silent
corpus change on resume. Validation averages a fixed 256KB of evenly spaced
contexts and does not consume training RNG.

## Result

| | tokens | word types | type/token | top bigram |
|---|---|---|---|---|
| v2 Korean | 2,702,000 | 2,427 | 0.09% | — |
| **v3 Korean** | 1,020,650 | **122,093** | **11.96%** | 557 |
| v2 English | 3,424,951 | 2,109 | 0.06% | — |
| **v3 English** | 3,524,796 | **60,970** | **1.73%** | 6,832 |

Korean vocabulary is **50× larger** at 100× the type/token ratio; English is 29×
larger. The template signature — every top bigram sharing one count — is gone.

## NF9 generalization audit

The step-40,000 NF9 checkpoint was trained on v2. On `summer` RTX 5070, the
same checkpoint and the same fixed 256KB span measured:

| corpus | partition | CE | BPC |
|---|---|---:|---:|
| v2 | train | 0.2796 | 0.4034 |
| v2 | sequential final 10% | 1.6493 | 2.3794 |
| v3 | train | 4.9422 | 7.1302 |
| v3 | family-isolated validation | 4.9749 | 7.1773 |

The historical v2 validation best of 0.3652 came from one stochastic batch and
is not comparable to the fixed-context result. NF9 has learned the repetitive
v2 distribution but does not generalize to the natural-language corpus. Its
low in-distribution CE therefore cannot support a language-quality claim.

### First v3 transition checkpoint

Training resumed from the step-40,000 NF9 weights on the canonical v3 pair. At
step 41,000, the first fixed 256KB validation measured CE **1.8331**, BPC
**2.6446**. This is a 63.2% BPC reduction from the same weights' pre-transition
v3 result (7.1773), while remaining above their v2 held-out BPC (2.3794).

The run also exposed two runtime-control defects before that checkpoint:

- the SOC sandpile stabilized one legal toppling per site per NumPy round,
  producing long critical-regime stalls;
- mean step latency let short Φ-only steps hide costly full steps, so cell
  growth repeatedly crossed the throughput budget.

The runtime now uses abelian bulk toppling, which preserves the stable state and
avalanche size, plus a checkpointed p90 latency governor with hysteresis. All
cell-growth values come from `training.toml`; resumed and canonical runs share
the same policy rather than target-specific constants.

## Verified end to end

Lexical metrics alone would not show the corpus is *usable*. Rebuilding QD-12's
grounded semantic composition on it, with the same shuffled-label control:

| | v3 corpus | shuffled control | |
|---|---|---|---|
| same-category concepts land closer | z = +4.54 | −0.02 ± 0.93 | **+4.9σ** |
| same-category pair stays in category | 26.0% (n=566) | 5.8% ± 3.3% | **+6.2σ** |
| cross-category pair hits a parent | 31.1% (n=9,445) | 11.5% ± 3.4% | **+5.8σ** |

Comparable to running against the raw `anima` markdown (+5.4σ / +8.3σ / +7.5σ);
slightly lower because markdown stripping drops some context, grounding 142 of
170 concepts rather than 154.

## The scripts now say what they are training on

`corpus_quality.warn_if_degenerate` runs at load time in `train_v9.py`,
`train_v11.py` and `train_v12.py`. It changes nothing about what is loaded —
switching a training default is the owner's call, being told what you are
training on is not. A run on a degenerate corpus otherwise looks exactly like a
run on real text from the console: same progress, same loss-curve shape.

```
  [corpus] ⚠️  data/corpus_v2.txt does not look like natural language:
    ko: 247,104 tokens over 1,519 types = 0.61% type/token  ← natural is ≥1%
    en: 306,700 tokens over 1,349 types = 0.44% type/token  ← natural is ≥1%
    73% of lines are duplicates
    Build a clean corpus with `python3 build_corpus.py` → data/corpus_v3.txt
```

`data/corpus_v3.txt` passes silently — ko 15.96%, en 4.86% on the same sample.

**The first version of this check was wrong and its own corpus caught it.** It
read the first 4MB of the file and reported v3 at 0.66% type/token, calling the
clean corpus degenerate. A concatenated corpus is ordered by source, so its head
is one source. The sample is now spread over eight positions across the file.

## Not done here

`corpus_v2.txt` is left in place and untouched — replacing a training input is
the owner's call, not a side effect of a corpus audit. The two v3 files are
gitignored and rebuilt from `build_corpus.py`, which is what is tracked.
