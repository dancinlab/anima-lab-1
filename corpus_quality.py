#!/usr/bin/env python3
"""corpus_quality.py — say out loud when a training corpus is not language.

`data/corpus_v2.txt` is 70MB and measures 0.09% type/token in Korean and 0.06%
in English, against a natural 1–3%. 41.7% of it is arithmetic drills, and its
dialogue section is 360,398 lines containing 190 unique ones. Three training
scripts default to it (QD-10, docs/corpus-rebuild.md).

Nothing warned. A run on it looks exactly like a run on real text — same
progress bars, same loss curve shape — so the defect is invisible from the
console. This makes it audible.

It does NOT change what anything trains on. Switching a training default is the
owner's call; being told what you are training on is not.

    from corpus_quality import warn_if_degenerate
    warn_if_degenerate(path)
"""

import collections
import re

KO = re.compile(r"[가-힣]{2,}")
EN = re.compile(r"[A-Za-z']{3,}")
ARITH = re.compile(r"^[\d\s+\-*/=÷×.RxX]+$")

# Natural running text sits at 1–3% type/token or above; this is a property of
# language, not a knob. Below it, the "vocabulary" is a template inventory.
NATURAL_FLOOR = 0.01
SAMPLE_CHARS = 4_000_000       # total sampled, spread across the file
SAMPLE_CHUNKS = 8              # ... in this many places, not all from the head


def measure(path, sample_chars=SAMPLE_CHARS, chunks=SAMPLE_CHUNKS):
    """Returns {lang: (tokens, types, ratio)} plus the arithmetic-line share.

    The sample is spread over the whole file, not taken from the head. A
    concatenated corpus is ordered by source, so its first megabytes are one
    source: reading only the head reported data/corpus_v3.txt at 0.66%
    type/token when the whole file measures 9.01%, and the diagnostic called
    its own clean corpus degenerate.
    """
    import os
    size = os.path.getsize(path)
    per = max(sample_chars // max(chunks, 1), 1)
    parts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if size <= sample_chars:
            parts.append(f.read())
        else:
            for i in range(chunks):
                f.seek(int(size * i / chunks))
                f.readline()          # drop the partial line the seek landed in
                parts.append(f.read(per))
    text = "\n".join(parts)
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    arith = sum(1 for s in lines if ARITH.match(s)) / max(len(lines), 1)
    dupe = 1.0 - len(set(lines)) / max(len(lines), 1)

    out = {}
    for name, pat in (("ko", KO), ("en", EN)):
        toks = pat.findall(text)
        if len(toks) < 1000:
            continue
        c = collections.Counter(t.lower() for t in toks)
        out[name] = (len(toks), len(c), len(c) / len(toks))
    return out, arith, dupe


def warn_if_degenerate(path, sample_chars=SAMPLE_CHARS):
    """Print a warning when the corpus is measurably not natural language.

    Returns True when a warning was printed. Never raises, never changes what
    the caller does — a diagnostic that stops a run would be worse than the
    silence it replaces.
    """
    try:
        stats, arith, dupe = measure(path, sample_chars)
    except Exception:
        return False

    bad = [n for n, (_, _, r) in stats.items() if r < NATURAL_FLOOR]
    if not bad and arith < 0.2 and dupe < 0.3:
        return False

    print(f"  [corpus] ⚠️  {path} does not look like natural language:")
    for n, (toks, types, r) in stats.items():
        flag = "  ← natural is ≥1%" if r < NATURAL_FLOOR else ""
        print(f"    {n}: {toks:,} tokens over {types:,} types = "
              f"{r * 100:.2f}% type/token{flag}")
    if arith >= 0.2:
        print(f"    {arith * 100:.0f}% of lines are arithmetic drills")
    if dupe >= 0.3:
        print(f"    {dupe * 100:.0f}% of lines are duplicates")
    print("    Build a clean corpus with `python3 build_corpus.py` "
          "→ data/corpus_v3.txt")
    print("    Measured in docs/corpus-rebuild.md · the run continues unchanged.")
    return True


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:] or ["data/corpus_v2.txt", "data/corpus_v3.txt"]:
        print(f"\n  {p}")
        if not warn_if_degenerate(p):
            stats, arith, dupe = measure(p)
            for n, (toks, types, r) in stats.items():
                print(f"    {n}: {toks:,} tokens over {types:,} types = "
                      f"{r * 100:.2f}% type/token  ok")
    print()
