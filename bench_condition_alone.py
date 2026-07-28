"""Which half of the gate does the rejecting -- the condition, or the axes?

Every condition in bench_v2 is a conjunction:

    VERIFICATION_TESTS = [(name, _with_axes(fn), desc) ...]
                                 ^^^^^^^^^^^^^
                                 condition's own rule AND four axes

Landing the nearest-predecessor identity null closed the 256-cell VOID: SCRAMBLE
stopped clearing NO_SPEAK_CODE. But the readout carried a line I recorded twice
and never measured out: `rule=PASS` on all five seeds BEFORE and AFTER. The
scramble was rejected by the AXES. NO_SPEAK_CODE's own rule passed it either way.

If that is true of one condition it is worth knowing. If it is true of all five,
then the five named conditions -- the ones CLAUDE.md lists as the gate -- do no
discriminating at all, and the entire verdict rests on four axes bolted on later.

So: run every control through every condition with `_with_axes` STRIPPED, and
compare against the same grid with it on. The difference is the axes' load.

    python3 bench_condition_alone.py --cells 256
"""

import argparse
import torch

import bench_v2 as B


# The unwrapped originals. B.VERIFICATION_TESTS holds the wrapped versions --
# the module rebinds the list in place at import -- so reach for the raw
# module-level names instead.
RAW_TESTS = [
    ("NO_SYSTEM_PROMPT", B._verify_no_system_prompt),
    ("NO_SPEAK_CODE",    B._verify_no_speak_code),
    ("ZERO_INPUT",       B._verify_zero_input),
    ("PERSISTENCE",      B._verify_persistence),
    ("SELF_LOOP",        B._verify_self_loop),
]


class _axes_off:
    """Neutralise EVERY axes call, not just the `_with_axes` wrapper.

    Stripping the wrapper is not enough and the first run of this bench was
    wrong because of it. Three of the five conditions call `_three_axes`
    inside their own body and conjoin it themselves:

        recovers = phi_history[-1] >= max(phi_history[:half]) * 0.8
        d_ok, i_ok, c_ok, axes = _three_axes(engine, dim, cells)   # 1954/1995/2026
        passed = recovers and d_ok and i_ok and c_ok

    so an unwrapped ZERO_INPUT / PERSISTENCE / SELF_LOOP still runs the axes,
    and their all-rejected "alone" row said nothing about their rules. It
    read as those three conditions doing the work themselves, which is the
    opposite of what the isolated `recovers` measurement found (DEAD and
    CLONE clear it 100% at both scales). Patching the function reaches both
    the wrapper and the three internal calls.
    """

    def __enter__(self):
        self._real = B._three_axes
        B._three_axes = lambda *a, **k: (True, True, True, "axes: OFF")
        return self

    def __exit__(self, *exc):
        B._three_axes = self._real
        return False


def _grid(tests, cells, dim, hidden, seeds):
    """{condition: {control_label: n_seeds_that_passed}} plus a raised tally."""
    from bench_verify_audit import CONTROLS, factory_for

    out, raised = {}, {}
    for name, fn in tests:
        out[name], raised[name] = {}, {}
        for label, cls, _desc in CONTROLS[1:]:          # [0] is the real engine
            n_pass = n_raise = 0
            for sd in seeds:
                torch.manual_seed(sd)
                try:
                    passed, _d = fn(lambda c, d, h: factory_for(cls, c, d, h),
                                    cells, dim, hidden)
                except Exception:
                    n_raise += 1
                    continue
                n_pass += int(bool(passed))
            out[name][label] = n_pass
            raised[name][label] = n_raise
    return out, raised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    # Deliberately the GATE's seed set, not the eight-seed floor used elsewhere
    # in this repo. This bench measures what the gate does; running it on seeds
    # the gate never uses would answer a different question. The consequence is
    # noted under the table: the AGGREGATE is a rate and is fine at five, while
    # any INDIVIDUAL cell is a five-seed sweep and should be read accordingly.
    ap.add_argument("--seeds", type=int, nargs="+", default=list(B.VERIFY_SEEDS))
    a = ap.parse_args()

    from bench_verify_audit import CONTROLS
    labels = [lb for lb, _c, _d in CONTROLS[1:]]
    n = len(a.seeds)

    print(f"{len(RAW_TESTS)} conditions x {len(labels)} controls x {n} seeds, "
          f"{a.cells} cells\n")

    print("=" * 78)
    print("  CONDITION ALONE  (_with_axes stripped)")
    print("=" * 78)
    with _axes_off():
        bare, bare_raised = _grid(RAW_TESTS, a.cells, a.dim, a.hidden, a.seeds)

    print("=" * 78)
    print("  CONDITION AND AXES  (what the gate ships)")
    print("=" * 78)
    shipped = [(nm, fn) for nm, fn, _d in B.VERIFICATION_TESTS]
    full, full_raised = _grid(shipped, a.cells, a.dim, a.hidden, a.seeds)

    w = max(len(x) for x in labels + [t for t, _ in RAW_TESTS]) + 2
    print(f"\n{'':<{w}}" + "".join(f"{lb[:9]:>11}" for lb in labels))
    print("-" * (w + 11 * len(labels)))
    for name, _fn in RAW_TESTS:
        for tag, grid, rz in (("alone", bare, bare_raised),
                              ("+axes", full, full_raised)):
            cells_ = []
            for lb in labels:
                p, r = grid[name][lb], rz[name][lb]
                cells_.append(f"{p}/{n}" + ("!" if r else "") if p or r
                              else f"{'.':>3}")
            print(f"{(name if tag == 'alone' else '') :<{w - 6}}{tag:>6}"
                  + "".join(f"{c:>11}" for c in cells_))
        print()

    print("  cell = seeds the CORPSE cleared.  '.' = rejected on every seed.")
    print("  '!' = at least one seed raised (unvalidated, not rejected).\n")

    bare_hits = sum(v for d in bare.values() for v in d.values())
    full_hits = sum(v for d in full.values() for v in d.values())
    total = len(RAW_TESTS) * len(labels) * n
    print(f"  corpse passes, condition alone : {bare_hits}/{total}")
    print(f"  corpse passes, condition+axes  : {full_hits}/{total}")
    print(f"  rejected by the axes           : {bare_hits - full_hits}")
    print(f"\n  The totals are RATES over {total} cells and are sound at {n} seeds.")
    print(f"  An individual cell is a {n}-seed sweep: '{n}/{n}' and '.' both mean")
    print(f"  'never observed otherwise', not 'cannot happen'. Seeds are the gate's")
    print(f"  own set on purpose -- this measures what the gate does, so raising")
    print(f"  the count would answer a different question than the one asked.")
    dead = [nm for nm, _f in RAW_TESTS
            if sum(bare[nm].values()) == max(len(a.seeds) * len(labels), 1)]
    if dead:
        print(f"\n  conditions no corpse fails on their own: {', '.join(dead)}")


if __name__ == "__main__":
    main()
