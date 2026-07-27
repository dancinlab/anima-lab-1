#!/usr/bin/env python3
"""audit_fake_measurements.py — find constants wearing the name of a measurement.

Every defect this session turned up was one shape: a value that reads like it
was measured and was not. They were all found by accident, one at a time:

    sha256(name) bits           → `complexity`, `emotionality`, `transcendent`
    a rule selected and dropped → `best_rule`, counted and never applied
    a variable nothing reads    → gate `g`, provably unable to change any argmax
    a bar on the wrong scale    → split_threshold 0.3 against a 0.037 peak
    a literal in an else branch → `cell_tension = 0.5`, the sole cause of every
                                  split in the engine anima_unified.py runs

The last one is mechanically detectable, so this looks for the rest of its kind
rather than waiting to trip over them. It reports, it does not judge — a
constant may be a legitimate initial value, and the point is to put every one in
front of a human instead of leaving it to be discovered by a failing experiment.

Three checks, all static:

  A. CONSTANT-AS-MEASUREMENT   a numeric literal assigned to a name that reads
                               like a measurement, ranked by whether it sits in
                               a fallback branch (else / except) — where a
                               "default" quietly becomes the operating value
  B. UNREAD ASSIGNMENT         a measurement-named local assigned and never read
                               again in its function
  C. THRESHOLD LITERALS        comparisons against bare numbers, which is where
                               a scale mismatch hides

    python3 audit_fake_measurements.py           # live code
    python3 audit_fake_measurements.py --all     # include archive/ and LEGACY
"""

import argparse
import ast
import pathlib
import re

MEASURED = re.compile(
    r"tension|phi|score|entropy|complexity|coherence|energy|confidence|"
    r"emotion|arousal|valence|novelty|curiosity|salience|attention|"
    r"activation|density|similarity|distance|error|loss|reward|fitness",
    re.I,
)
SKIP_DIRS = ("archive", "benchmarks", ".venv", "node_modules", "__pycache__", "tests")


def _is_num(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


class Auditor(ast.NodeVisitor):
    def __init__(self, path, src):
        self.path = path
        self.lines = src.splitlines()
        self.const_hits = []
        self.unread_hits = []
        self.threshold_hits = []
        self._fallback_depth = 0

    # --- A: constant assigned to a measurement-named target -----------------
    def visit_Assign(self, node):
        if _is_num(node.value):
            for t in node.targets:
                name = (t.id if isinstance(t, ast.Name)
                        else t.attr if isinstance(t, ast.Attribute) else None)
                if name and MEASURED.search(name):
                    self.const_hits.append(
                        (node.lineno, name, node.value.value, self._fallback_depth > 0))
        self.generic_visit(node)

    def visit_If(self, node):
        for n in node.body:
            self.visit(n)
        self._fallback_depth += 1        # orelse is the fallback path
        for n in node.orelse:
            self.visit(n)
        self._fallback_depth -= 1
        self.visit(node.test)

    def visit_ExceptHandler(self, node):
        self._fallback_depth += 1
        self.generic_visit(node)
        self._fallback_depth -= 1

    # --- C: comparison against a bare number -------------------------------
    def visit_Compare(self, node):
        left = node.left
        lname = (left.id if isinstance(left, ast.Name)
                 else left.attr if isinstance(left, ast.Attribute) else "")
        if MEASURED.search(lname or ""):
            for cmp_ in node.comparators:
                if _is_num(cmp_):
                    self.threshold_hits.append((node.lineno, lname, cmp_.value))
        self.generic_visit(node)

    # --- B: measurement-named local assigned and never read afterwards -----
    def visit_FunctionDef(self, node):
        assigned, read = {}, set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store) and MEASURED.search(n.id):
                    assigned.setdefault(n.id, n.lineno)
                elif isinstance(n.ctx, ast.Load):
                    read.add(n.id)
        for name, ln in assigned.items():
            if name not in read:
                self.unread_hits.append((ln, name, node.name))
        self.generic_visit(node)


def audit(path):
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return None
    a = Auditor(path, src)
    a.visit(tree)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--path", default=".")
    args = ap.parse_args()

    files = []
    for f in sorted(pathlib.Path(args.path).rglob("*.py")):
        s = str(f)
        if not args.all and (any(f"/{d}/" in s or s.startswith(f"{d}/") for d in SKIP_DIRS)
                             or "LEGACY" in s):
            continue
        files.append(f)

    print(f"\n  가짜 측정 감사 — {len(files)}개 파일\n")

    fallback, plain, unread, thresholds = [], [], [], []
    for f in files:
        a = audit(f)
        if a is None:
            continue
        for ln, name, val, in_fallback in a.const_hits:
            (fallback if in_fallback else plain).append((f, ln, name, val))
        for ln, name, fn in a.unread_hits:
            unread.append((f, ln, name, fn))
        for ln, name, val in a.threshold_hits:
            thresholds.append((f, ln, name, val))

    print(f"  ── A. 대비책 분기 안의 상수 — 여기가 위험한 자리 ({len(fallback)}건)")
    print("     else/except 안의 기본값은 조용히 '평상시 값'이 되곤 한다\n")
    for f, ln, name, val in fallback[:25]:
        print(f"    {f.as_posix():<38}:{ln:<5} {name} = {val}")
    if len(fallback) > 25:
        print(f"    … 외 {len(fallback) - 25}건")

    print(f"\n  ── B. 계산해놓고 아무도 안 읽는 값 ({len(unread)}건)")
    for f, ln, name, fn in unread[:15]:
        print(f"    {f.as_posix():<38}:{ln:<5} {name}  (in {fn})")
    if len(unread) > 15:
        print(f"    … 외 {len(unread) - 15}건")

    print(f"\n  ── C. 맨숫자와의 비교 — 척도 불일치가 숨는 자리 ({len(thresholds)}건)")
    for f, ln, name, val in thresholds[:15]:
        print(f"    {f.as_posix():<38}:{ln:<5} {name} vs {val}")
    if len(thresholds) > 15:
        print(f"    … 외 {len(thresholds) - 15}건")

    print(f"\n  A 중 대비책 아닌 평범한 상수 대입은 {len(plain)}건 (초기값일 수 있어 별도 집계)")
    print("  이 도구는 신고만 한다 — 상수가 정당한 초기값일 수 있고,")
    print("  판단은 각 자리를 사람이 보고 해야 한다.\n")


if __name__ == "__main__":
    main()
