#!/usr/bin/env python3
"""emotion_grounding.py — emit an emotion value only when one can be justified.

`bench_consciousness_universe.simulate_meta_ca` produced 18 confident emotion
numbers per stimulus from arithmetic over `sha256(name)` bits — `joy = 0.3 +
0.4·emotionality·(1 − complexity·0.3)` where `emotionality` is byte 3 of a hash.
The `░▒▓█` heatmap built from them is a picture of a hash. That is what QD-1
flagged and left, and it is the block that started this whole series: it was
mistaken for a measurement, which is exactly what a fabrication that looks like
data does.

Grounding them properly needs a corpus where the emotion words actually occur.
Measured over both available sources:

    corpus_v3      11/18 words at ≥3 occurrences, 4/18 at ≥20, 3 at zero
    ko_wiki         6/18 at ≥3,  1/18 at ≥20, 7 at zero

Neither can carry eighteen emotions. Building a model on the four words with
real support would be the same manipulation in a new coat.

So this module does what CLAUDE.md #1 says to do — *의식이 말 못하면 침묵* — and
returns `None` for every emotion it cannot ground. A `None` renders as a blank
in the heatmap. A mostly blank heatmap is the honest picture; the full one was
never real.

    from emotion_grounding import ground_emotions
    ground_emotions("호흡과 획")   # {'meaning': 0.31, 'trust': None, ...}
"""

import collections
import math
import pathlib
import re

EMOTION_WORDS = {
    "joy": "기쁨", "sadness": "슬픔", "anger": "분노", "fear": "공포",
    "surprise": "놀람", "curiosity": "호기심", "awe": "경외", "love": "사랑",
    "trust": "신뢰", "flow": "몰입", "meaning": "의미", "creativity": "창조",
    "hope": "희망", "ecstasy": "황홀", "peace": "평화", "rage": "격분",
    "despair": "절망", "longing": "그리움",
}

CORPUS = pathlib.Path("data/corpus_v3.txt")
KO = re.compile(r"[가-힣]{2,}")
MIN_COUNT = 20          # below this a "context vector" is a handful of sentences
CONTEXT_VOCAB = 4000
WINDOW = 5

_cache = {}


def _load():
    if "vectors" in _cache:
        return _cache["vectors"], _cache["freq"]
    if not CORPUS.exists():
        _cache["vectors"], _cache["freq"] = {}, collections.Counter()
        return _cache["vectors"], _cache["freq"]

    sents, freq = [], collections.Counter()
    with CORPUS.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            w = KO.findall(line)
            if len(w) >= 3:
                sents.append(w)
                freq.update(w)

    ctx = {w: i for i, (w, _) in enumerate(freq.most_common(CONTEXT_VOCAB))}
    targets = {w for w in freq if freq[w] >= MIN_COUNT}
    counts = collections.defaultdict(lambda: collections.Counter())
    for s in sents:
        for i, w in enumerate(s):
            if w not in targets:
                continue
            lo, hi = max(0, i - WINDOW), min(len(s), i + WINDOW + 1)
            for j in range(lo, hi):
                if j != i and s[j] in ctx:
                    counts[w][ctx[s[j]]] += 1

    total = sum(sum(c.values()) for c in counts.values()) or 1
    col = collections.Counter()
    for c in counts.values():
        col.update(c)

    vectors = {}
    for w, c in counts.items():
        row = sum(c.values())
        v = {}
        for j, n in c.items():
            pmi = math.log((n * total) / max(row * col[j], 1e-12))
            if pmi > 0:
                v[j] = pmi
        norm = math.sqrt(sum(x * x for x in v.values()))
        if norm > 1e-12:
            vectors[w] = {j: x / norm for j, x in v.items()}

    _cache["vectors"], _cache["freq"] = vectors, freq
    return vectors, freq


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(j, 0.0) for j, x in a.items())


def grounded_emotions():
    """The subset of the 18 that this corpus can actually support."""
    vectors, freq = _load()
    return {e: w for e, w in EMOTION_WORDS.items() if w in vectors}


def ground_emotions(description):
    """description → {emotion: value in [0,1]} with None where ungrounded.

    A value is the cosine between the description's pooled context vector and
    the emotion word's, rescaled from [-1,1] to [0,1] so it drops into the same
    range the old hash arithmetic used. `None` means the corpus does not contain
    the emotion word often enough to say anything, and nothing is invented to
    fill the gap.
    """
    vectors, _ = _load()
    words = [w for w in KO.findall(description) if w in vectors]

    pooled = collections.Counter()
    for w in words:
        for j, x in vectors[w].items():
            pooled[j] += x
    norm = math.sqrt(sum(x * x for x in pooled.values()))
    pooled = {j: x / norm for j, x in pooled.items()} if norm > 1e-12 else {}

    out = {}
    for emotion, word in EMOTION_WORDS.items():
        ev = vectors.get(word)
        out[emotion] = ((_cos(pooled, ev) + 1.0) / 2.0
                        if ev and pooled else None)
    return out


if __name__ == "__main__":
    g = grounded_emotions()
    print(f"\n  근거를 얻은 감정 {len(g)}/{len(EMOTION_WORDS)}: "
          f"{' '.join(f'{e}({w})' for e, w in g.items())}")
    print(f"  나머지 {len(EMOTION_WORDS) - len(g)}개는 코퍼스에 "
          f"{MIN_COUNT}회 미만 — 값을 만들지 않고 None 을 낸다\n")
    for desc in ("호흡과 획", "우주의 설계도", "통증의 쾌락"):
        v = ground_emotions(desc)
        shown = " ".join(f"{e}={x:.2f}" for e, x in v.items() if x is not None)
        print(f"  {desc:<12} {shown or '(근거 없음)'}")
    print()
