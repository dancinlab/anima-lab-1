#!/usr/bin/env python3
"""build_corpus.py — assemble a training corpus that is actually language.

`data/corpus_v2.txt` is 70MB and measurably degenerate (QD-10):

    41.7% arithmetic drills
    the dialogue is 360,398 lines and 190 UNIQUE ones — one block × 2538
    deduplicated prose: 2,074 Korean word types over 870,211 tokens (0.24%)
                        1,728 English types over 1,223,950 tokens (0.14%)
    top bigrams occur at exactly 3000 each — templates, not language

QD-12 found real language one directory over: `anima/**/*.md` is 1,144,691
Korean tokens over 82,873 types, type/token 6.50%.

This writes NEW files and never touches the old one:

    data/corpus_v3.txt        natural language, deduplicated
    data/corpus_v3_arith.txt  the arithmetic drills, kept separate

Splitting them is the point. Arithmetic may well be wanted for arithmetic
training, but mixed into a language corpus at 41.7% it is 41.7% of the
gradient spent on something that is not language.

Markdown is stripped of fenced code, tables, links and headings markup before
its prose is taken — otherwise the "vocabulary" fills with identifiers.

Usage:
    python3 build_corpus.py            # build + report
    python3 build_corpus.py --report   # report on existing files only
"""

import argparse
import collections
import math
import pathlib
import re

ARITH = re.compile(r"^[\d\s+\-*/=÷×.RxX]+$")
HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
KO_TOK = re.compile(r"[가-힣]{2,}")
EN_TOK = re.compile(r"[A-Za-z']{3,}")

OUT_TEXT = pathlib.Path("data/corpus_v3.txt")
OUT_ARITH = pathlib.Path("data/corpus_v3_arith.txt")
SOURCES_MD = (pathlib.Path("/Users/mini/dancinlab/anima"), pathlib.Path("."))
# Plain-text sources, listed explicitly. data/.corpus_cache/ko_wiki.txt is real
# Korean Wikipedia prose — 180,789 tokens over 60,383 types, type/token 33.40%,
# no template signature — and it was already tracked in git, sitting unused in a
# hidden cache directory. A glob over data/*.txt would pull in corpus_v2 (handled
# separately) and corpus_v3 itself, so the list is explicit.
SOURCES_TXT = (pathlib.Path("data/.corpus_cache/ko_wiki.txt"),)
MIN_TOKENS = 4              # a line needs this many word tokens to be prose
TEMPLATE_CAP = 20           # max lines per template family
NGRAM = 6                   # token window used to detect a shared skeleton


def strip_markdown(text):
    """Drop fenced code, tables, link targets and markup so prose is left."""
    out, in_fence = [], False
    for line in text.splitlines():
        s = line.rstrip()
        if s.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if s.lstrip().startswith(("|", ">", "    ", "\t")):
            continue
        s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)     # links → their text
        s = re.sub(r"`[^`]*`", " ", s)                        # inline code
        s = re.sub(r"https?://\S+", " ", s)
        s = re.sub(r"^#{1,6}\s*", "", s)
        s = re.sub(r"[*_~]{1,3}", "", s)
        s = re.sub(r"<!--.*?-->", " ", s)
        out.append(s.strip())
    return out


def token_count(line):
    return len(KO_TOK.findall(line)) + len(EN_TOK.findall(line))


def ngrams(line):
    t = KO_TOK.findall(line) + [w.lower() for w in EN_TOK.findall(line)]
    return [tuple(t[i:i + NGRAM]) for i in range(len(t) - NGRAM + 1)]


def drop_templates(lines):
    """Cap families of lines that share an over-represented n-gram.

    Digit masking is not enough. The 3000 copies here are word problems whose
    names, objects and comparison verbs all vary —
    `예린은(는) 블록을(를) 76개 가지고 있고 … 둘이 합치면 97개이고` — so every
    line has a distinct digit-masked skeleton while `합치면 개이고` still occurs
    3000 times. Frequency of a shared n-gram finds the family without knowing
    anything about its content, which is the point: no pattern is hardcoded.
    """
    freq = collections.Counter()
    for s in lines:
        freq.update(set(ngrams(s)))
    hot = {g for g, n in freq.items() if n > TEMPLATE_CAP}

    kept, used = [], collections.Counter()
    dropped = 0
    for s in lines:
        gs = [g for g in ngrams(s) if g in hot]
        if gs:
            key = min(gs)
            used[key] += 1
            if used[key] > TEMPLATE_CAP:
                dropped += 1
                continue
        kept.append(s)
    print(f"  template families: {len(hot):,} over-represented {NGRAM}-grams, "
          f"{dropped:,} lines dropped")
    return kept


def collect():
    text_lines, arith_lines, seen = [], [], set()

    def keep(s):
        return True

    old = pathlib.Path("data/corpus_v2.txt")
    if old.exists():
        for raw in old.open(encoding="utf-8", errors="replace"):
            s = raw.strip()
            if not s:
                continue
            if ARITH.match(s):
                arith_lines.append(s)
                continue
            if s in seen:
                continue
            seen.add(s)
            if token_count(s) >= MIN_TOKENS and keep(s):
                text_lines.append(s)

    for root in SOURCES_MD:
        if not root.exists():
            continue
        for f in root.rglob("*.md"):
            sp = str(f)
            if ".git" in sp or "node_modules" in sp or ".venv" in sp:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for s in strip_markdown(content):
                if not s or s in seen:
                    continue
                if not (HANGUL.search(s) or LATIN.search(s)):
                    continue
                seen.add(s)
                if token_count(s) >= MIN_TOKENS and keep(s):
                    text_lines.append(s)

    for f in SOURCES_TXT:
        if not f.exists():
            continue
        for raw in f.open(encoding="utf-8", errors="replace"):
            s = raw.strip()
            if not s or s in seen or s.startswith("#"):
                continue
            seen.add(s)
            if token_count(s) >= MIN_TOKENS:
                text_lines.append(s)

    return drop_templates(text_lines), arith_lines


def metrics(lines, label):
    ko = [w for s in lines for w in KO_TOK.findall(s)]
    en = [w.lower() for s in lines for w in EN_TOK.findall(s)]
    rows = []
    for name, toks in (("ko", ko), ("en", en)):
        if not toks:
            continue
        c = collections.Counter(toks)
        big = collections.Counter()
        pat = KO_TOK if name == "ko" else EN_TOK
        for s in lines:
            t = pat.findall(s)
            for i in range(len(t) - 1):
                big[(t[i], t[i + 1])] += 1
        top = [n for _, n in big.most_common(5)]
        rows.append((label, name, len(toks), len(c), len(c) / len(toks) * 100,
                     max(top) if top else 0, len(set(top)) == 1 if top else False))
    return rows


def report(rows):
    print(f"  {'corpus':<16} {'':<3} {'tokens':>11} {'types':>9} {'type/token':>11} "
          f"{'top bigram':>11} {'templated?':>11}")
    for label, name, ntok, ntyp, tt, topn, flat in rows:
        flag = "yes ⚠" if flat else "no"
        print(f"  {label:<16} {name:<3} {ntok:>11,} {ntyp:>9,} {tt:>10.2f}% "
              f"{topn:>11,} {flag:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if not args.report:
        text_lines, arith_lines = collect()
        OUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
        OUT_TEXT.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        OUT_ARITH.write_text("\n".join(arith_lines) + "\n", encoding="utf-8")
        print(f"\n  wrote {OUT_TEXT}  {len(text_lines):,} lines "
              f"({OUT_TEXT.stat().st_size / 1e6:.1f} MB)")
        print(f"  wrote {OUT_ARITH}  {len(arith_lines):,} lines "
              f"({OUT_ARITH.stat().st_size / 1e6:.1f} MB)")

    print("\n  ── lexical quality, before and after " + "─" * 30)
    rows = []
    old = pathlib.Path("data/corpus_v2.txt")
    if old.exists():
        old_lines = [s.strip() for s in old.open(encoding="utf-8", errors="replace")
                     if s.strip() and not ARITH.match(s.strip())]
        rows += metrics(old_lines, "v2 (non-arith)")
    rows += metrics(OUT_TEXT.read_text(encoding="utf-8").splitlines(), "v3")
    report(rows)
    print("\n  natural language sits at 1–3% type/token or above;")
    print("  a top-bigram count shared by every top pair is a template signature.")
    print()


if __name__ == "__main__":
    main()
