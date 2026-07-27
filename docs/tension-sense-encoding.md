# TensionSense — replacing `ord(c) / 256.0`

The live S-engine path turned text into numbers by writing each character's
codepoint into the matching slot. `docs/qualia-decoder-spec.md` listed it as an
honesty defect at the start of this series; it is now fixed, with the
replacement chosen by measurement rather than taste.

## Three defects, each demonstrated against the replacement

`dim = 128`, distances between the two encodings of the given pair.

| | `ord(c)/256` | bigram pooling |
|---|---|---|
| two strings differing only **past** character 128 | **0.000000** | 0.149585 |
| the same string shifted by one leading space | **266.74** | 0.6822 |
| kin closer than stranger (`서예`↔`서예체` vs `서예`↔`빅뱅`) | 204.70 vs 10.17 → **FAIL** | 1.0301 vs 1.3784 → ok |

1. **Truncation.** Anything past `dim` characters did not exist. Two documents
   agreeing for 128 characters and diverging after were bit-identical to the
   engine — distance exactly 0.
2. **Position lock.** Character *i* went to slot *i*, so inserting one space at
   the front moved every character into the next slot and changed the vector
   completely: distance 266.74, larger than between two unrelated strings.
3. **No similarity.** Codepoints are arbitrary. `서예`/`서예체` came out **20×
   further apart** than `서예`/`빅뱅` — the encoding actively inverted kinship.

## The replacement, and why this one

`qualia_sense.text_vector` pools boundary-padded character bigrams: each bigram
contributes a fixed pseudorandom unit direction derived from its hash, and the
sum is normalised. Length is unbounded, shared substrings give shared
directions, and `^`/`$` padding makes `AB` differ from `BA`.

It was not picked for elegance. Four encodings were measured in QD-8 on
stem-kin distance over unrelated distance — below 1 means the encoding has
structure — and on whether `AB` can be told from `BA`:

| encoding | kin / unrelated | AB vs BA |
|---|---|---|
| whole-string `sha256` | 1.07 — none | 1.42 (noise; repetition non-monotone) |
| `qualia_sense` features | 0.66 | **0.0000** — structurally order-blind |
| bag of characters | 0.42 | 0.0000 |
| **bigram pooling** | **0.47** | **0.8231** |

Bag-of-characters has slightly better kin structure but cannot represent order
at all. Bigram pooling is the only one with both, which is why it is the one
wired in.

## Scope

This changes how `TensionSense` encodes **strings**. Tensor inputs are
untouched. Nothing else in the S-engine path changed, and the habituation
baseline it feeds is unaltered.
