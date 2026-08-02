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

This writes NEW files and never touches the old one. Paths and split policy
come from the repository SSOT, ``corpus.toml``:

    data/corpus_v3.txt             natural language, deduplicated
    data/corpus_v3_train.txt       canonical training partition
    data/corpus_v3_validation.txt  held-out, family-isolated validation
    data/corpus_v3_arith.txt       the arithmetic drills, kept separate

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
import dataclasses
import hashlib
import pathlib
import random
import re
import tomllib

ARITH = re.compile(r"^[\d\s+\-*/=÷×.RxX]+$")
HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
KO_TOK = re.compile(r"[가-힣]{2,}")
EN_TOK = re.compile(r"[A-Za-z']{3,}")

CONFIG_PATH = pathlib.Path(__file__).with_name("corpus.toml")


@dataclasses.dataclass(frozen=True)
class CorpusConfig:
    """Canonical corpus build and split settings loaded from ``corpus.toml``."""

    legacy_source: pathlib.Path
    markdown_sources: tuple[pathlib.Path, ...]
    text_sources: tuple[pathlib.Path, ...]
    excluded_sources: tuple[pathlib.Path, ...]
    full_output: pathlib.Path
    train_output: pathlib.Path
    validation_output: pathlib.Path
    arithmetic_output: pathlib.Path
    minimum_tokens: int
    template_family_cap: int
    template_ngram_tokens: int
    validation_fraction: float
    split_seed: str
    audit_window_bytes: int
    audit_samples: int
    maximum_overlap: float
    evaluation_bytes: int


def load_config(path: pathlib.Path = CONFIG_PATH) -> CorpusConfig:
    """Load the single source of truth for corpus construction."""
    path = pathlib.Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))["corpus"]
    root = path.parent
    outputs = raw["outputs"]
    filters = raw["filters"]
    split = raw["split"]
    evaluation = raw["evaluation"]

    def local(value: str) -> pathlib.Path:
        candidate = pathlib.Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = CorpusConfig(
        legacy_source=local(raw["legacy_source"]),
        markdown_sources=tuple(local(value) for value in raw["markdown_sources"]),
        text_sources=tuple(local(value) for value in raw["text_sources"]),
        excluded_sources=tuple(local(value) for value in raw["excluded_sources"]),
        full_output=local(outputs["full"]),
        train_output=local(outputs["train"]),
        validation_output=local(outputs["validation"]),
        arithmetic_output=local(outputs["arithmetic"]),
        minimum_tokens=int(filters["minimum_tokens"]),
        template_family_cap=int(filters["template_family_cap"]),
        template_ngram_tokens=int(filters["template_ngram_tokens"]),
        validation_fraction=float(split["validation_fraction"]),
        split_seed=str(split["seed"]),
        audit_window_bytes=int(split["audit_window_bytes"]),
        audit_samples=int(split["audit_samples"]),
        maximum_overlap=float(split["maximum_overlap"]),
        evaluation_bytes=int(evaluation["bytes"]),
    )
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("corpus.split.validation_fraction must be between 0 and 1")
    if config.minimum_tokens < 1 or config.template_ngram_tokens < 1:
        raise ValueError("corpus filter sizes must be positive")
    if config.audit_window_bytes < 1 or config.audit_samples < 1:
        raise ValueError("corpus audit sizes must be positive")
    if config.evaluation_bytes < 1:
        raise ValueError("corpus.evaluation.bytes must be positive")
    return config


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


def ngrams(line, size):
    t = KO_TOK.findall(line) + [w.lower() for w in EN_TOK.findall(line)]
    return [tuple(t[i:i + size]) for i in range(len(t) - size + 1)]


def group_and_cap_templates(lines, config):
    """Cap template families and return ``(line, split_family)`` records.

    Digit masking is not enough. The 3000 copies here are word problems whose
    names, objects and comparison verbs all vary —
    `예린은(는) 블록을(를) 76개 가지고 있고 … 둘이 합치면 97개이고` — so every
    line has a distinct digit-masked skeleton while `합치면 개이고` still occurs
    3000 times. Frequency of a shared n-gram finds the family without knowing
    anything about its content, which is the point: no pattern is hardcoded.
    """
    line_grams = [set(ngrams(s, config.template_ngram_tokens)) for s in lines]
    freq = collections.Counter(g for grams in line_grams for g in grams)
    hot = {g for g, count in freq.items()
           if count > config.template_family_cap}

    # A line may contain several hot n-grams. Unioning co-occurring signatures
    # makes the whole template family one split unit, so a shared skeleton can
    # never leak between training and validation under a different signature.
    parent = {g: g for g in hot}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        first, second = sorted((left, right))
        parent[second] = first

    for grams in line_grams:
        shared = sorted(grams & hot)
        for other in shared[1:]:
            union(shared[0], other)

    kept, used = [], collections.Counter()
    dropped = 0
    families = set()
    for s, grams in zip(lines, line_grams):
        shared = sorted(grams & hot)
        if shared:
            key = find(shared[0])
            families.add(key)
            used[key] += 1
            if used[key] > config.template_family_cap:
                dropped += 1
                continue
            family = "\x1f".join(key)
        else:
            family = s
        kept.append((s, family))
    print(f"  template families: {len(families):,} connected components, "
          f"{dropped:,} lines dropped")
    return kept


def collect(config):
    text_lines, arith_lines, seen = [], [], set()
    excluded = {path.resolve() for path in config.excluded_sources}

    def keep(s):
        return True

    old = config.legacy_source
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
            if token_count(s) >= config.minimum_tokens and keep(s):
                text_lines.append(s)

    for root in config.markdown_sources:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.md")):
            if f.resolve() in excluded:
                continue
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
                if token_count(s) >= config.minimum_tokens and keep(s):
                    text_lines.append(s)

    for f in config.text_sources:
        if not f.exists():
            continue
        for raw in f.open(encoding="utf-8", errors="replace"):
            s = raw.strip()
            if not s or s in seen or s.startswith("#"):
                continue
            seen.add(s)
            if token_count(s) >= config.minimum_tokens:
                text_lines.append(s)

    return group_and_cap_templates(text_lines, config), arith_lines


def split_records(records, config):
    """Deterministically split family groups without train/validation leakage."""
    train, validation = [], []
    threshold = int(config.validation_fraction * (1 << 64))
    seed = config.split_seed.encode("utf-8")
    assignments = {}
    for line, family in records:
        if family not in assignments:
            digest = hashlib.sha256(seed + b"\0" + family.encode("utf-8")).digest()
            assignments[family] = (
                validation if int.from_bytes(digest[:8], "big") < threshold else train
            )
        assignments[family].append(line)
    return train, validation


def sampled_byte_overlap(train_path, validation_path, window, samples, seed):
    """Measure exact held-out byte-window reuse with an isolated seeded sampler."""
    train = train_path.read_bytes()
    validation = validation_path.read_bytes()
    if len(train) < window or len(validation) < window:
        raise ValueError("corpus split is smaller than the configured audit window")
    population = len(validation) - window + 1
    count = min(samples, population)
    sampler = random.Random(seed)
    positions = sampler.sample(range(population), count)
    hits = sum(validation[pos:pos + window] in train for pos in positions)
    return hits / count, hits, count


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG_PATH)
    args = ap.parse_args()
    config = load_config(args.config)

    if not args.report:
        records, arith_lines = collect(config)
        train_lines, validation_lines = split_records(records, config)
        text_lines = [line for line, _ in records]
        if set(train_lines) & set(validation_lines):
            raise RuntimeError("exact lines leaked across the corpus split")

        for path, lines in (
            (config.full_output, text_lines),
            (config.train_output, train_lines),
            (config.validation_output, validation_lines),
            (config.arithmetic_output, arith_lines),
        ):
            write_lines(path, lines)
            print(f"  wrote {path}  {len(lines):,} lines "
                  f"({path.stat().st_size / 1e6:.1f} MB)")

        overlap, hits, count = sampled_byte_overlap(
            config.train_output,
            config.validation_output,
            config.audit_window_bytes,
            config.audit_samples,
            config.split_seed,
        )
        print(f"  split audit: exact lines=0, "
              f"{config.audit_window_bytes}-byte overlap={hits}/{count} "
              f"({overlap:.2%})")
        if overlap > config.maximum_overlap:
            raise RuntimeError(
                f"validation overlap {overlap:.2%} exceeds configured maximum "
                f"{config.maximum_overlap:.2%}"
            )

    print("\n  ── lexical quality, before and after " + "─" * 30)
    rows = []
    old = config.legacy_source
    if old.exists():
        old_lines = [s.strip() for s in old.open(encoding="utf-8", errors="replace")
                     if s.strip() and not ARITH.match(s.strip())]
        rows += metrics(old_lines, "v2 (non-arith)")
    rows += metrics(config.full_output.read_text(encoding="utf-8").splitlines(), "v3")
    report(rows)
    print("\n  natural language sits at 1–3% type/token or above;")
    print("  a top-bigram count shared by every top pair is a template signature.")
    print()


if __name__ == "__main__":
    main()
