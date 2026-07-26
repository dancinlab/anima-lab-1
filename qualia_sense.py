#!/usr/bin/env python3
"""qualia_sense.py — stimulus → feature vector from actual string content.

Replaces the SHA-256 path in bench_consciousness_universe.py, where a
stimulus's "characteristics" were bits of `sha256(name)`. A hash is a valid
way to get eight numbers and an invalid way to get eight *measurements*: it
destroys similarity structure, so `서예` and `서예체` land as far apart as
`서예` and `빅뱅`.

Every feature here is measured from the string itself and is named for what
it measures. None of them is named for a psychological state — this module
does not know what `emotionality` or `transcendence` would be, and does not
pretend to.

What this is NOT:
  - It reads FORM, not MEANING. `서예` and `붓글씨` are unrelated here.
    Semantic proximity needs an embedding model (torch is absent on this host).
  - The stimuli are names OF artworks, not the artworks. Nothing is seen.

Honest limitation, stated plainly: which feature drives which state variable
downstream is still a choice. The difference from the hash is that the choice
is now made over real measurements and is written down, instead of being made
over noise and hidden.

Korean syllables decompose per Unicode:
  code = ord(ch) - 0xAC00;  cho = code//588,  jung = (code%588)//28,  jong = code%28

Usage:
    f = sense("서예")
    f.complexity, f.structure, ...     # named measurements in [0, 1]
    feature_distance(sense("서예"), sense("서예체"))
"""

import hashlib
import math
import unicodedata
from dataclasses import dataclass, fields
from typing import Dict, List

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3
N_JUNG = 21
N_JONG = 28

# Feature order is fixed — vectors are compared positionally.
FEATURE_NAMES = (
    "length",           # how long the token is
    "jamo_density",     # letters packed per character (Korean syllables hold 2-3)
    "final_ratio",      # share of Korean syllables carrying a final consonant
    "vowel_position",   # mean vowel index in the 21-vowel table
    "script_mix",       # entropy over script classes (hangul/latin/han/digit/symbol)
    "char_variety",     # entropy of the character distribution
    "bigram_repeat",    # share of character bigrams that recur
    "codepoint_spread", # dispersion of codepoints
)


@dataclass(frozen=True)
class Qualia:
    """Eight measurements of a stimulus string, each in [0, 1]."""

    name: str
    length: float
    jamo_density: float
    final_ratio: float
    vowel_position: float
    script_mix: float
    char_variety: float
    bigram_repeat: float
    codepoint_spread: float

    def vector(self) -> List[float]:
        return [getattr(self, n) for n in FEATURE_NAMES]

    def as_dict(self) -> Dict[str, float]:
        return {n: getattr(self, n) for n in FEATURE_NAMES}


def _entropy(counts) -> float:
    """Shannon entropy of a count distribution, normalised to [0, 1]."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(len(probs)) if len(probs) > 1 else 0.0


def _script_class(ch: str) -> str:
    cp = ord(ch)
    if HANGUL_BASE <= cp <= HANGUL_LAST or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "han"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "latin"
    return "symbol"


def _decompose(ch: str):
    """Korean syllable → (cho, jung, jong), or None for non-syllables."""
    cp = ord(ch)
    if not (HANGUL_BASE <= cp <= HANGUL_LAST):
        return None
    code = cp - HANGUL_BASE
    return code // (N_JUNG * N_JONG), (code % (N_JUNG * N_JONG)) // N_JONG, code % N_JONG


def sense(name: str) -> Qualia:
    """Measure a stimulus string. Deterministic, similarity-preserving."""
    s = unicodedata.normalize("NFC", name)
    if not s:
        return Qualia(name=name, **{n: 0.0 for n in FEATURE_NAMES})

    chars = list(s)
    n = len(chars)

    # length — saturating, so a long title does not dominate every other feature
    length = min(n / 16.0, 1.0)

    # jamo_density / final_ratio / vowel_position — Korean phonological structure
    jamo_total = 0
    syllables = 0
    finals = 0
    jung_sum = 0
    for ch in chars:
        d = _decompose(ch)
        if d is None:
            jamo_total += 1
            continue
        cho, jung, jong = d
        syllables += 1
        jamo_total += 3 if jong else 2
        finals += 1 if jong else 0
        jung_sum += jung
    jamo_density = min((jamo_total / n) / 3.0, 1.0)
    final_ratio = finals / syllables if syllables else 0.0
    vowel_position = (jung_sum / syllables) / (N_JUNG - 1) if syllables else 0.0

    # script_mix — how many writing systems the token draws on
    klasses = ["hangul", "han", "latin", "digit", "symbol"]
    counts = [sum(1 for ch in chars if _script_class(ch) == k) for k in klasses]
    script_mix = _entropy(counts)

    # char_variety — repetition inside the token
    freq: Dict[str, int] = {}
    for ch in chars:
        freq[ch] = freq.get(ch, 0) + 1
    char_variety = _entropy(list(freq.values())) if n > 1 else 0.0

    # bigram_repeat — recurring character pairs
    if n > 1:
        bigrams = [s[i:i + 2] for i in range(n - 1)]
        bigram_repeat = 1.0 - (len(set(bigrams)) / len(bigrams))
    else:
        bigram_repeat = 0.0

    # codepoint_spread — dispersion of the codepoints, scaled by a full Hangul block
    cps = [ord(ch) for ch in chars]
    if n > 1:
        mean = sum(cps) / n
        var = sum((c - mean) ** 2 for c in cps) / n
        codepoint_spread = min(math.sqrt(var) / 4000.0, 1.0)
    else:
        codepoint_spread = 0.0

    return Qualia(
        name=name,
        length=length,
        jamo_density=jamo_density,
        final_ratio=final_ratio,
        vowel_position=vowel_position,
        script_mix=script_mix,
        char_variety=char_variety,
        bigram_repeat=bigram_repeat,
        codepoint_spread=codepoint_spread,
    )


def hash_sense(name: str) -> Qualia:
    """The original SHA-256 path, kept for comparison.

    bench_consciousness_universe.data_characteristics() in Qualia form, so the
    two feature sources can be tested against each other rather than argued about.
    """
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    vals = [((h >> (8 * i)) & 0xFF) / 255.0 for i in range(len(FEATURE_NAMES))]
    return Qualia(name=name, **dict(zip(FEATURE_NAMES, vals)))


def feature_distance(a: Qualia, b: Qualia) -> float:
    """Euclidean distance between two feature vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.vector(), b.vector())))


def sim_inputs(q: Qualia) -> Dict[str, float]:
    """Map measurements onto the two state variables the CA simulation needs.

    Assignment rule: the four features with the highest measured spread over
    the stimulus set drive the state, most-discriminating first. Measured over
    the 170 names in bench_consciousness_universe.ALL_DATA_TYPES:

      final_ratio      0.367  → init_p
      codepoint_spread 0.328  → init_g
      script_mix       0.279  → bias_p
      char_variety     0.266  → bias_g
      vowel_position   0.237     unused here
      jamo_density     0.156     unused here
      length           0.073     unused here
      bigram_repeat    0.000     degenerate on this set — no name repeats a bigram

    The rule is stated so it can be checked, and it is a rule about spread, not
    about meaning. Nothing here claims these are complexity, emotion, or
    transcendence. Re-run the spread measurement if the stimulus set changes.
    """
    return {
        "init_p": q.final_ratio,
        "init_g": q.codepoint_spread,
        "bias_p": q.script_mix,
        "bias_g": q.char_variety,
    }


if __name__ == "__main__":
    probes = ["서예", "서예체", "서예가", "만다라", "검은사각형", "빅뱅"]
    print(f"{'stimulus':<12} " + " ".join(f"{n[:6]:>7}" for n in FEATURE_NAMES))
    for p in probes:
        q = sense(p)
        print(f"{p:<12} " + " ".join(f"{v:>7.3f}" for v in q.vector()))

    print("\ndistance from 서예:")
    base = sense("서예")
    hbase = hash_sense("서예")
    for p in probes[1:]:
        print(f"  {p:<12} content={feature_distance(base, sense(p)):.3f}  "
              f"hash={feature_distance(hbase, hash_sense(p)):.3f}")
