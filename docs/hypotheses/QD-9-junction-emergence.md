# QD-9: A piece that exists only in the whole — and reaches the engine

<!-- @hypothesis-ok — this repo's canonical hypothesis dir is docs/hypotheses/ (91 pre-existing cards, referenced across docs/). Not migrating to HYPOTHESES/ as part of this experiment. -->

**Owner-requested evaluation, run after the fact. No pre-registered hypotheses
and no verdict.** Reproduce with `bench_emergence_junction.py`.

This is the first result in the QD series where something new arises from
combination *and* survives the pipeline. Stated carefully below, including what
it is not.

## Why QD-8's null was foreordained

QD-8 concluded that combination is pure composition. It measured with
`qualia_sense`, which scores `AB` against `BA` at exactly **0.0000** — the
encoding is a bag of character statistics and the join is invisible to it. An
order-blind encoding cannot show that joining creates anything, because to it
nothing was joined.

## The mechanism, proved before it was measured

Pooling is a sum of one unit direction per token, so the algebra is exact:

```
bag      tokens(AB) = chars(A) + chars(B)
         pool(AB) = pool(A) + pool(B)                    ← additive, always
         residual = 0 by construction

bigram   tokens(AB) = bigrams(A) + [junction] + bigrams(B)   (boundary-padded)
         pool(AB) = pool(A) + pool(B)
                    + unit(a_last·b_first)                ← in NEITHER part
                    − unit(a_last·$) − unit(^·b_first)    ← in the parts, gone
```

The join **creates one token and destroys two**. The whole is not a superset of
its parts.

Verified numerically rather than assumed — the residual matches that expression
to **6.6e-16 across all 153 pairs**, and the junction bigram is absent from both
parts in **every** pair.

At the string layer it is not a small correction: the residual is **0.732** of
the mean distance between the two parts.

## It reaches the settled engine state

The engine has swallowed every signal this session. The test is ablation:
delete that one novel token from the pool, change nothing else, and see whether
the settled state moves.

| | off the A–B line | n |
|---|---|---|
| combination A+B | **49.6%** | 153 |
| same, junction ablated | **44.0%** | 153 |
| unrelated word C — control | 88.0% | 2448 |

**Junction contributes +5.62 ± 0.83 percentage points · paired z = +6.74**,
with 106 of 153 pairs moving in the predicted direction.

Removing a single token — one that existed in neither part — measurably shifts
where the combination settles. **The novel element is carried, not swallowed.**

### Two corrections made while measuring this

- **One-character words broke the algebra.** `bigrams()` fell back to
  `list(s)` when a string was too short for any bigram, mixing character tokens
  into a bigram pool; 서예+공 and 서예+용 showed an error of exactly 1.0.
  Boundary padding (`^s$`) fixes it — every string then has bigrams, and the
  fix also makes the destroyed-token half of the algebra visible.
- **The first run used an unpaired test and read z = +1.74.** Each pair
  contributes a real *and* an ablated measurement of the same combination, so
  the per-pair difference is the statistic; the unpaired standard error throws
  the pairing away. The design was paired all along. The word set also went
  from 8 (28 pairs) to 18 (153 pairs).

## What this is, and what it is not

**It is** a demonstrated case of a new element arising from combination and
propagating: provably absent from both parts, provably present in the whole,
large at the representation layer, and statistically carried through to the
engine's settled state.

**It is not** semantic emergence. `예` + `만` making `예만` is a mechanical
consequence of concatenating strings — n-gram algebra, not a leap of meaning.
Nothing here shows the system understanding a combined concept.

**And composition still dominates.** At 49.6% off-line against a control of
88.0%, the combination is far closer to a mix of its parts than an unrelated
word is. The honest picture is *mostly composition, with a small novel
component that is real and survives*.

## Why it matters here

Every previous attempt in this series put the new information into a **value** —
a coordinate, a coupling strength, an accumulated trace, a stored anchor — and
the attractor absorbed all of them. This one puts it into a **token**: a
discrete element that either is or is not in the pool. Tokens do not average
away, and this one did not.

That is the first structural opening the series has found. It says nothing yet
about whether meaning can ride on it.
