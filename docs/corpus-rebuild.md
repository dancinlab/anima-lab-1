# Training corpus — what was wrong and what replaced it

`data/corpus_v2.txt` is 70MB and measurably not language (QD-10). This records
the defect, the rebuild, and the end-to-end check that the result is usable.
Regenerate with `python3 build_corpus.py`.

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

## Result

| | tokens | word types | type/token | top bigram |
|---|---|---|---|---|
| v2 Korean | 2,702,000 | 2,427 | 0.09% | — |
| **v3 Korean** | 1,356,204 | **122,183** | **9.01%** | 1,079 |
| v2 English | 3,424,951 | 2,109 | 0.06% | — |
| **v3 English** | 3,839,941 | **61,875** | **1.61%** | 6,832 |

Korean vocabulary is **50× larger** at 100× the type/token ratio; English is 29×
larger. The template signature — every top bigram sharing one count — is gone.

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

## Not done here

`corpus_v2.txt` is left in place and untouched — replacing a training input is
the owner's call, not a side effect of a corpus audit. The two v3 files are
gitignored and rebuilt from `build_corpus.py`, which is what is tracked.
