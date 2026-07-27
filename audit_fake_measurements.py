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
  D. CONSTANT COMPARISON       a name that only ever takes literal values,
                               compared against a literal — decidable, and it is
                               how an ethics gate ends up unable to refuse
  E. HASH TAINT                a value derived from hashlib flowing into a
                               measurement-named target — a hash read as data
An F was attempted and REMOVED: "inside a loop over `r`, an additive term not
mentioning `r` cannot change an argmax over the loop's candidates". That is the
shape of the inert gate `g`, but the check as written produced **1137 hits**
across the live root files and mis-attributed nested loops to the outermost
variable — it never verified the part that makes the pattern a defect, that the
loop's result feeds a selection. A check that buries real findings under a
thousand false ones is worse than no check. The gate `g` therefore remains
undetectable by this tool, and that is stated rather than papered over.

Recall against the six defects this session found by hand, measured on the code
as it was BEFORE they were fixed (git 710c0ec):

    A  cell_tension = 0.5              consciousness_engine.py:332   caught
    D  phi_preservation ∈ {0.5, 1.0} vs 0.3 → always True            caught
    E  sha256 → complexity/emotionality/entropy_input                caught
    C  split_threshold vs 0.3          flags the comparison, cannot judge scale
    -  best_rule computed and discarded                              MISSED
    -  gate g inert                                                  MISSED

    3 of 6 clean, 1 partial. The two misses are both "a value that IS read but
    cannot affect the outcome", which needs dataflow this tool does not do.

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
        self.const_cmp_hits = []
        self.hash_hits = []
        self._fallback_depth = 0
        self._hash_names = set()

    # --- A: constant assigned to a measurement-named target -----------------
    def _mentions_hash(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
               and n.value.id == "hashlib":
                return True
            if isinstance(n, ast.Name) and n.id in self._hash_names:
                return True
        return False

    def visit_Assign(self, node):
        # E: hash taint — track names carrying hash-derived values, and report
        # when one reaches a measurement-named target.
        if self._mentions_hash(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._hash_names.add(t.id)
                    if MEASURED.search(t.id):
                        self.hash_hits.append((node.lineno, t.id))
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
        # D: names whose every assignment in this scope is a literal
        lit = {}
        for n in ast.walk(node):
            # `x += 1` makes x non-literal even if every plain assignment to it
            # is a number. Without this, any counter initialised to 0 and then
            # incremented reads as "only ever 0" and every comparison against it
            # is reported as decided — 40 of the first 55 hits were exactly that.
            if isinstance(n, (ast.AugAssign, ast.For)):
                tgt = n.target
                nm = (tgt.id if isinstance(tgt, ast.Name)
                      else tgt.attr if isinstance(tgt, ast.Attribute) else None)
                if nm:
                    lit[nm] = None
                continue
            if isinstance(n, ast.Assign):
                # Tuple unpacking (`chi2, p = stats.chisquare(...)`) assigns a
                # computed value, so every name in the target must count as
                # non-literal. Missing this reported two p-values as "only ever
                # 1.0" because the literal was in the skipped-data else branch
                # while the real assignment came through a tuple.
                flat = []
                for t in n.targets:
                    flat.extend(t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])
                unpacked = any(isinstance(t, (ast.Tuple, ast.List)) for t in n.targets)
                for t in flat:
                    nm = (t.id if isinstance(t, ast.Name)
                          else t.attr if isinstance(t, ast.Attribute) else None)
                    if nm is None:
                        continue
                    if unpacked:
                        lit[nm] = None
                        continue
                    if not _is_num(n.value):
                        lit[nm] = None            # not literal-only, permanently
                    elif nm in lit and lit[nm] is None:
                        pass                      # already disqualified
                    else:
                        lit.setdefault(nm, set()).add(n.value.value)
        for n in ast.walk(node):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 \
               and _is_num(n.comparators[0]):
                left = n.left
                nm = (left.id if isinstance(left, ast.Name)
                      else left.attr if isinstance(left, ast.Attribute) else None)
                vals = lit.get(nm)
                if not vals:
                    continue
                bar = n.comparators[0].value
                op = n.ops[0]
                def holds(v):
                    if isinstance(op, ast.Gt):  return v > bar
                    if isinstance(op, ast.GtE): return v >= bar
                    if isinstance(op, ast.Lt):  return v < bar
                    if isinstance(op, ast.LtE): return v <= bar
                    return None
                res = {holds(v) for v in vals}
                if len(res) == 1 and None not in res:
                    self.const_cmp_hits.append(
                        (n.lineno, nm, sorted(vals), bar, res.pop(), node.name))

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
    const_cmp, hash_taint = [], []
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
        for hit in a.const_cmp_hits:
            const_cmp.append((f,) + hit)
        for ln, name in a.hash_hits:
            hash_taint.append((f, ln, name))

    print(f"  ── A. 대비책 분기 안의 상수 — 여기가 위험한 자리 ({len(fallback)}건)")
    print("     else/except 안의 기본값은 조용히 '평상시 값'이 되곤 한다\n")
    for f, ln, name, val in fallback[:25]:
        print(f"    {f.as_posix():<38}:{ln:<5} {name} = {val}")
    if len(fallback) > 25:
        print(f"    … 외 {len(fallback) - 25}건")

    print(f"\n  ── D. 항상 참/거짓인 비교 — 판정 가능한 결함 ({len(const_cmp)}건)")
    print("     값이 리터럴만 갖는데 리터럴과 비교하면 결과는 이미 정해져 있다\n")
    for f, ln, nm, vals, bar, res, fn in const_cmp[:15]:
        print(f"    {f.as_posix():<34}:{ln:<5} {nm} in {vals} vs {bar} -> always {res}  ({fn})")
    if len(const_cmp) > 15:
        print(f"    … 외 {len(const_cmp) - 15}건")

    print(f"\n  ── E. 해시에서 흘러든 '측정값' ({len(hash_taint)}건)")
    for f, ln, nm in hash_taint[:15]:
        print(f"    {f.as_posix():<34}:{ln:<5} {nm}")
    if len(hash_taint) > 15:
        print(f"    … 외 {len(hash_taint) - 15}건")

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
