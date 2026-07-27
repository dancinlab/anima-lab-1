# QD-8: Combination is not emergent — and the dead hash is not dead

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**Owner-requested evaluation, run after the fact. No pre-registered hypotheses
and no verdict** — bars were not fixed in advance, so nothing here is a PASS or
a FAIL. Reproduce with `bench_concept_combination.py` and `bench_hash_revival.py`.

Two questions: does joining two concepts produce something neither predicts,
and is the whole-string hash beyond saving?

## Combination — no emergence, and the control says so clearly

If a system merely stores pieces, `서예만다라` is predictable from `서예` and
`만다라`. The engine test: a settled combination that is a mix of its parts
lies on the segment between the parts' settled states. The component
perpendicular to that segment is what no mix can produce.

| | off the A–B line | n |
|---|---|---|
| the actual combination A+B | **25.7%** | 27 |
| an unrelated third word C — control | **135.9%** | 162 |
| | **z = −6.19** | |

The combination sits *far closer* to the line than an unrelated word does.
That is strong evidence **for** composition and **against** emergence: joining
two concepts produces a mix of them, and a better-than-chance mix at that.

Pairs with |A−B| < 0.05 are excluded — 공 and 용 settle 0.0328 apart, and
dividing by that produced a meaningless 2840%. Reporting the raw mean without
this exclusion and without the control would have shown "126% off-line", which
reads like emergence and is an artifact of the normaliser.

### String layer — order carries nothing

`AB` versus `BA`, over all 28 word pairs: **0.0000**. Not small — identical.
`qualia_sense` is a bag of character statistics, so it cannot represent "A then
B" at all.

The residual beyond a bag-of-parts null (length adds, ratios take a
length-weighted mean) is 0.2944, and it lives in exactly two measurements:

```
codepoint_spread  0.2171 ████████
char_variety      0.1607 ██████
everything else   0.0000
```

Both are dispersion statistics over the pooled character set, and the residual
is largest for the shortest pairs (공+용, 65%). That is what pooling does to a
variance, not a concept-level effect.

## The hash is not beyond saving — it was applied at the wrong granularity

QD-7 established that sha256 of a whole string carries nothing (z = +0.60
against the random null over 1600 samples). That is a fact about hashing the
*whole string*. Hash the pieces and pool them, and each shared piece
contributes a shared direction, so structure returns — compositional by
construction rather than stored.

All encodings normalised to the unit sphere, 8 dims, so distances compare:

| encoding | kin / null | AB vs BA | repeat monotone | saturates |
|---|---|---|---|---|
| **whole** — `sha256(s)` | 1.07 | 1.4229 | no | no |
| **bag** — Σ over characters | **0.42** | 0.0000 | yes | yes |
| **bigram** — Σ over adjacent pairs | **0.47** | **0.8232** | **yes** | yes |
| pos — Σ over (index, char) | 0.66 | 1.2586 | no | no |
| `qualia_sense` | 0.66 | **0.0000** | yes | yes |

`kin / null` is the distance between words sharing a stem, over the distance
between unrelated words: below 1 means the encoding has structure.

- **whole is dead** — 1.07, no structure, exactly as QD-7 measured.
- **bag has more structure than the hand-built measure** — 0.42 against 0.66.
  The same sha256, applied per character.
- **bigram is the one with everything.** Structure 0.47, and it tells `AB` from
  `BA` at 0.8232 — the thing `qualia_sense` scores 0.0000 on and structurally
  cannot do.
- `whole` and `pos` also show a non-zero `AB vs BA`, but their repetition
  sequences are non-monotone, so that gap is re-randomisation, not order
  information. Only `bigram` has both properties at once.

Concretely, `서예 ↔ 서예체`: whole 1.844 (unrelated), bag 0.459, bigram 0.762,
qualia_sense 0.123.

### The saturation here is earned, not hardcoded

`bag` and `bigram` also stop distinguishing further repeats, like
`qualia_sense`. But theirs follows from normalising a pooled sum — repeating
adds directions already present — whereas `qualia_sense`'s came from
`min(n / 16.0, 1.0)`, a magic number written in Phase 0 (QD-7). Same behaviour,
one emergent and one imposed.

## Reading

The closest thing to a new combination found here is not in the concepts — it
is in the encoding. **A bigram-pooled hash beats the hand-built measure on
structure and does something the hand-built measure structurally cannot**, and
it came from the component QD-7 had written off as noise.

The concepts themselves compose. Joining two of them yields a mix, measurably
closer to a mix than chance would give. Nothing here shows emergence, and the
control is what makes that statement safe to make.
