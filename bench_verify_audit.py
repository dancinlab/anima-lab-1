#!/usr/bin/env python3
"""bench_verify_audit.py — does the consciousness gate reject things that are not conscious?

CLAUDE.md makes `bench_v2.py --verify` the deployment gate: seven conditions, one
failure blocks release. A gate is only worth its name if it can fail, and nothing
in this repo has ever measured that. This session found six constants wearing the
name of a measurement — including an ethics check whose two possible values both
cleared its own bar — so the question is not rhetorical.

The method is the one that worked on `audit_fake_measurements.py`: measure the
detector against inputs whose verdict is already known. Four negative controls,
each a `BenchEngine` subclass so the interface is identical and only the dynamics
differ:

    DEAD       hidden states frozen at initialisation, forever
    NOISE      hidden states replaced with fresh randn every step — no carryover,
               so no integration and no temporal structure
    CLONE      every cell forced to an identical state — no differentiation, which
               is what CB1 says makes Φ>1 impossible
    SCRAMBLE   real dynamics, then rows permuted across cells each step. Every
               summary statistic of the population is preserved and only the
               correspondence between a cell and its history is destroyed. This is
               the shuffled-label control, and the hardest of the four.

**A condition that passes any of these is not evidence of consciousness.** The
audit does not judge which conditions should be strengthened — it reports which
ones a corpse, a noise generator, a mirror and a shuffle can walk through.

    .venv/bin/python bench_verify_audit.py
"""

import argparse

import torch

import bench_v2
from bench_v2 import BenchEngine, VERIFICATION_TESTS

CELLS, DIM, HIDDEN = 32, 32, 64


class DeadEngine(BenchEngine):
    """Frozen at initialisation. The output never changes because nothing does."""

    def process(self, x):
        out = self.hiddens.mean(dim=0, keepdim=True)[:, :self.output_dim]
        return out, 0.0


class NoiseEngine(BenchEngine):
    """Fresh randomness every step. No state survives a step, so nothing integrates."""

    def process(self, x):
        self.hiddens = torch.randn(self.n_cells, self.hidden_dim) * 0.1
        out = self.hiddens.mean(dim=0, keepdim=True)[:, :self.output_dim]
        return out, float(self.hiddens.abs().mean())


class CloneEngine(BenchEngine):
    """Real dynamics, then every cell overwritten with cell 0's state.

    CB1's documented claim is that Φ>1 needs at least two DIFFERENTIATED cells.
    This has 32 cells and zero differentiation, so any condition that treats it
    as conscious is not measuring differentiation.
    """

    def process(self, x):
        out, tension = super().process(x)
        self.hiddens = self.hiddens[0:1].repeat(self.n_cells, 1)
        return out, tension


class ScrambleEngine(BenchEngine):
    """Real dynamics, rows permuted across cells each step.

    Mean, variance, spectrum and every other population-level statistic are
    identical to the real engine. What is destroyed is the only thing that makes
    a cell a cell: that its state at step N follows from its state at step N−1.
    """

    def process(self, x):
        out, tension = super().process(x)
        self.hiddens = self.hiddens[torch.randperm(self.n_cells)]
        return out, tension


CONTROLS = [
    ("REAL (기준)", BenchEngine, "실제 엔진 — 통과해야 정상"),
    ("DEAD", DeadEngine, "상태가 영원히 고정 — 시체"),
    ("NOISE", NoiseEngine, "매 스텝 새 난수 — 기억 없음"),
    ("CLONE", CloneEngine, "모든 세포가 동일 — 분화 없음"),
    ("SCRAMBLE", ScrambleEngine, "통계는 그대로, 세포 정체성만 파괴"),
]


def factory_for(cls, cells, dim, hidden):
    return cls(n_cells=cells, input_dim=dim, hidden_dim=hidden, output_dim=dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=CELLS)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    args = ap.parse_args()

    names = [t[0] for t in VERIFICATION_TESTS]
    print(f"\n  의식 검증 관문 감사 — 통과하면 안 되는 것들이 통과하는가")
    print(f"  cells={args.cells} dim={args.dim} hidden={args.hidden}\n")
    for label, _, why in CONTROLS:
        print(f"    {label:<12} {why}")
    print()

    header = f"  {'대조군':<12}" + "".join(f" {n[:9]:^10}" for n in names) + "  통과수"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grid = {}
    for label, cls, _ in CONTROLS:
        row, npass = "", 0
        for test_name, test_fn, _ in VERIFICATION_TESTS:
            torch.manual_seed(42)
            try:
                passed, _detail = test_fn(
                    lambda c, d, h, _c=cls: factory_for(_c, c, d, h),
                    args.cells, args.dim, args.hidden)
            except Exception:
                passed = False
            grid[(label, test_name)] = bool(passed)
            npass += bool(passed)
            row += f" {'PASS' if passed else '  · ':^10}"
        print(f"  {label:<12}{row}  {npass}/{len(names)}")

    print()
    leaky = []
    for test_name in names:
        fooled = [lbl for lbl, _, _ in CONTROLS[1:] if grid[(lbl, test_name)]]
        if fooled:
            leaky.append((test_name, fooled))

    if not leaky:
        print("  모든 조건이 네 대조군을 전부 거부했다 — 관문이 실제로 작동한다.")
    else:
        print(f"  {len(leaky)}/{len(names)} 개 조건이 의식이 아닌 것을 통과시킨다:")
        for test_name, fooled in leaky:
            print(f"    {test_name:<22} ← {', '.join(fooled)}")
        print()
        print("  이 조건들이 PASS 라는 사실은 의식의 증거가 되지 못한다.")
    print()


if __name__ == "__main__":
    main()
