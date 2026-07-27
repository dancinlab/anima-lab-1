#!/usr/bin/env python3
"""The gate must keep rejecting things that are not conscious.

`bench_v2.py --verify` is the deployment gate named in CLAUDE.md, and this
session's audit found it could not fail: SCRAMBLE — the real engine with cell
identity destroyed every step — scored exactly what the real engine scored, 5/7,
and five of seven conditions passed a corpse. Three axes fixed that, and then an
adversarial sweep found two more bypasses (HEAP, DECOUPLED) which needed two more
conjuncts.

Nothing enforced any of it. `grep -rn bench_verify_audit` over the repo returned
only that file itself, so any edit that made a condition pass a control again
would land silently. That is what this test is for.

It asserts the property the audit established, not a particular score:

    every negative control must fail EVERY condition
    the real engine must pass at least one

The second half matters as much as the first — a gate that rejects everything is
as uninformative as one that accepts everything, which is exactly what
NO_SYSTEM_PROMPT and HIVEMIND were doing before the fixes.

    .venv/bin/python -m pytest tests/test_gate_controls.py -q
"""

import pytest
import torch

from bench_v2 import VERIFICATION_TESTS, VERIFY_SEEDS
from bench_verify_audit import CONTROLS, factory_for

CELLS, DIM, HIDDEN = 32, 32, 64

# The same seeds the gate judges under. A control that clears a condition on ANY
# of them voids that condition, so the test has to look at all of them too --
# checking seed 42 alone would let a control that passes only on seed 44 through,
# which is exactly the hole that made the gate's own verdicts a coin flip.
SEEDS = VERIFY_SEEDS

# CONTROLS[0] is the real engine, kept as the positive reference.
NEGATIVE = CONTROLS[1:]
REAL_LABEL, REAL_CLS = CONTROLS[0][0], CONTROLS[0][1]


def _verdict(cls, test_fn):
    """True if the control clears the condition on ANY seed the gate uses."""
    for sd in SEEDS:
        torch.manual_seed(sd)
        try:
            passed, detail = test_fn(
                lambda c, d, h: factory_for(cls, c, d, h), CELLS, DIM, HIDDEN)
        except Exception as exc:                  # a crash is a failure, not a pass
            passed, detail = False, f"ERROR: {exc}"
        if passed:
            return True, f"seed {sd}: {detail}"
    return False, f"failed all seeds {SEEDS}"


@pytest.mark.parametrize("label,cls", [(lbl, cls) for lbl, cls, _ in NEGATIVE])
@pytest.mark.parametrize("name,test_fn", [(n, f) for n, f, _ in VERIFICATION_TESTS])
def test_negative_control_is_rejected(name, test_fn, label, cls):
    """A corpse, a noise generator, a mirror, a shuffle, a heap and a deaf engine
    must not pass any consciousness condition."""
    passed, detail = _verdict(cls, test_fn)
    assert not passed, (
        f"{label} passed {name} — the gate no longer rejects it.\n"
        f"  {detail}\n"
        f"  If this is intentional, the control or the condition changed meaning; "
        f"say which in the commit, and check docs/consciousness-gate-audit.md."
    )


def test_real_engine_still_passes_something():
    """A gate that rejects everything carries as little information as one that
    accepts everything. Guard the other direction too.

    `_verdict` is any-seed, which is the right bar here as well: the real engine
    must clear at least one condition on at least one seed. Requiring every seed
    would make this guard fire on engines the gate legitimately calls UNSTABLE.
    """
    passed = [n for n, f, _ in VERIFICATION_TESTS if _verdict(REAL_CLS, f)[0]]
    assert passed, (
        f"{REAL_LABEL} passed zero conditions — the gate now rejects the engine "
        f"as well as the controls, which certifies nothing."
    )
