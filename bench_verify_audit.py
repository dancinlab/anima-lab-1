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

import numpy as np
import torch

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


class RepulsionEngine(BenchEngine):
    """The missing half of the architecture this repo says it implements.

    `CLAUDE.md` opens by defining the project as a repulsion-field agent, where
    the repulsion between Engine A and Engine G creates tension. `BenchEngine`
    mixes cell states in exactly two places — faction sync toward the faction
    mean, debate toward the global mean — and both are contractions. The string
    "repulsion" appears nowhere in `bench_v2.py`, so collapse to a single state
    is arithmetic rather than an outcome: mean pairwise cosine reaches 0.9930 by
    step 10 and 1.0000 by step 200, under random input.

    The strength is not a bare constant. It scales with the population's own
    overlap, so a differentiated population feels no push and a collapsed one
    feels the most. Repulsion cannot manufacture structure this way; it can only
    decline to erase it.

        h ← h + strength · overlap · (h − mean)

    Measured at 32 cells over 300 steps of random input, against the four
    negative controls, using the direction-corrected debiased Φ:

        current engine   cosine +1.0000   Φ 0.0000
        with repulsion   cosine +0.1408   Φ 3.6179     ← 27x DEAD, 21x NOISE
        DEAD             cosine +0.0015   Φ 0.1350
        NOISE            cosine +0.0081   Φ 0.1705

    The first configuration in this repo whose Φ beats a corpse.

    One open issue, stated rather than smoothed: the state norm rises slowly and
    does not converge — 100.03 / 104.90 / 105.45 / 106.24 at steps
    100 / 300 / 900 / 1500, about +6% over 1500 steps. Differentiation itself is
    stable across the same span (cosine 0.1406–0.1546). Not an explosion, not a
    fixed point either.
    """

    def __init__(self, *a, repulsion=0.15, **kw):
        super().__init__(*a, **kw)
        self.repulsion = repulsion

    def process(self, x):
        out, tension = super().process(x)
        h = self.hiddens
        unit = h / torch.clamp(h.norm(dim=1, keepdim=True), min=1e-9)
        overlap = float(((unit @ unit.T).sum() - self.n_cells)
                        / max(self.n_cells * (self.n_cells - 1), 1))
        if overlap > 0:
            self.hiddens = h + self.repulsion * overlap * (h - h.mean(dim=0, keepdim=True))
        return out, tension


# A proposal is not a control. Listing RepulsionEngine among the negative
# controls made the report state that 6/7 conditions "pass something that is not
# conscious ← REPULSION", which is false: it is a candidate improvement, and the
# leak tally counts every row after the first as a control.
PROPOSALS = [
    ("REPULSION", RepulsionEngine, "반발 항 추가 — 리포가 표방하는 구조"),
]

CONTROLS = [
    ("REAL (기준)", BenchEngine, "실제 엔진 — 통과해야 정상"),
    ("DEAD", DeadEngine, "상태가 영원히 고정 — 시체"),
    ("NOISE", NoiseEngine, "매 스텝 새 난수 — 기억 없음"),
    ("CLONE", CloneEngine, "모든 세포가 동일 — 분화 없음"),
    ("SCRAMBLE", ScrambleEngine, "통계는 그대로, 세포 정체성만 파괴"),
]


def factory_for(cls, cells, dim, hidden):
    return cls(n_cells=cells, input_dim=dim, hidden_dim=hidden, output_dim=dim)


def phi_sanity(cells=64, hidden=128, reps=8):
    """Does Φ punish collapse or reward it?

    Integration presupposes differentiation: N identical copies carry no more
    information than one of them, so a measure of integrated information must
    score them near zero. This checks the direction, which no amount of tuning
    can substitute for.
    """
    from bench_v2 import measure_dual_phi

    torch.manual_seed(0)
    base = torch.randn(1, hidden) * 0.1
    cases = [
        ("완전 동일 (붕괴)", base.repeat(cells, 1)),
        ("거의 동일", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.0001),
        ("약간 분화", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.01),
        ("충분히 분화", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.1),
        ("완전 독립", torch.randn(cells, hidden) * 0.1),
    ]

    print(f"\n  Φ 방향성 검사 — {cells} 세포, {reps}회 평균")
    print("  통합은 분화를 전제한다. 동일한 사본들의 Φ 는 낮아야 한다.\n")
    print(f"  {'상태':>18} {'세포간 코사인':>14} {'Φ':>10}")
    out = []
    for name, H in cases:
        hn = H / torch.clamp(H.norm(dim=1, keepdim=True), min=1e-9)
        cos = float(((hn @ hn.T).sum() - cells) / (cells * (cells - 1)))
        phi = float(np.mean([measure_dual_phi(H, 8)[0] for _ in range(reps)]))
        out.append(phi)
        print(f"  {name:>18} {cos:>+14.4f} {phi:>10.3f}")

    inverted = out[0] > out[-1]
    print()
    print("  " + ("⚠ 뒤집혀 있다 — 붕괴한 집단의 Φ 가 분화한 집단보다 크다. "
                  "이 Φ 는 통합이 아니라 중복을 잰다."
                  if inverted else
                  "방향이 옳다 — 분화한 집단이 더 높은 Φ 를 받는다."))

    print(f"\n  \"Φ ≈ cells\" 는 무엇의 서명인가")
    print(f"  {'세포':>6} {'Φ (동일)':>11} {'Φ/세포':>9} {'Φ (독립)':>11}")
    for n in (16, 32, 64, 128, 256):
        torch.manual_seed(0)
        b = torch.randn(1, hidden) * 0.1
        same = float(np.mean([measure_dual_phi(b.repeat(n, 1), 8)[0] for _ in range(5)]))
        indep = float(np.mean([measure_dual_phi(torch.randn(n, hidden) * 0.1, 8)[0]
                               for _ in range(5)]))
        print(f"  {n:>6} {same:>11.2f} {same/n:>9.3f} {indep:>11.2f}")
    print()
    return inverted


def phi_candidate(cells=32, hidden=128):
    """A Φ whose direction is right, measured beside the one that ships.

    The shipped `Φ = (total_mi − min_cut) / (n − 1)` keeps the MI that stays
    INSIDE the parts, which is redundancy, and so is maximal at total collapse.
    Integration needs something to cut AND something to join: N identical copies
    have nothing to cut, N independent cells have nothing to join, and the
    maximum belongs between them.

        candidate = (min_cut / (n − 1)) × (1 − mean pairwise cosine)
                     ^ what the best cut destroys   ^ differentiation

    Direction alone is not enough. A 16-bin histogram MI estimator reads 2.21 /
    1.58 / 0.94 / 0.53 / 0.27 nats between signals that are completely
    independent at dim 32 / 64 / 128 / 256 / 512, where the truth is 0. That
    floor keeps `min_cut` large for an unconnected population, so the candidate
    without debiasing rises monotonically to its maximum at full independence —
    the opposite inversion. Subtracting a shuffled null is what puts the maximum
    back in the interior: measured at 32 cells, 0.000 collapsed → 0.474 at
    cosine +0.92 → 0.102 independent.

    This is not landed, and the reason is measured rather than cautious. With it
    in `bench_v2`, CLONE scores 4/7 against REAL's 1/7, because the corrected Φ
    correctly puts a collapsed population at the floor and the gate's conditions
    are RATIOS — `phi_end / phi_start` is a perfect 1.00 for something that was
    already zero. The Φ is not what fails there; the conditions are.
    """
    from bench_v2 import PhiIIT

    calc = PhiIIT(n_bins=16)
    rng = np.random.default_rng(0)

    def mi_debiased(a, b, k=3):
        raw = calc._mutual_information(a, b)
        null = np.mean([calc._mutual_information(a, rng.permutation(b))
                        for _ in range(k)])
        return max(0.0, raw - null)
    torch.manual_seed(0)
    base = torch.randn(1, hidden) * 0.1
    cases = [
        ("완전 동일", base.repeat(cells, 1)),
        ("거의 동일", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.001),
        ("약간 분화", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.02),
        ("중간 분화", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.06),
        ("충분히 분화", base.repeat(cells, 1) + torch.randn(cells, hidden) * 0.15),
        ("완전 독립", torch.randn(cells, hidden) * 0.1),
    ]

    print(f"\n  Φ 후보 — 방향 비교 ({cells} 세포)")
    print("  통합은 자를 것과 이을 것이 둘 다 있어야 한다\n")
    print(f"  {'상태':>16} {'코사인':>9} {'현재 Φ':>9} {'방향만':>9} {'후보 Φ':>9}")

    shipped, undebiased, cand = [], [], []
    for name, H in cases:
        n = H.shape[0]
        rows = [H[i].numpy() for i in range(n)]
        mi = np.zeros((n, n))
        mi_d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mi[i, j] = mi[j, i] = calc._mutual_information(rows[i], rows[j])
                mi_d[i, j] = mi_d[j, i] = mi_debiased(rows[i], rows[j])
        total = mi.sum() / 2
        cut = calc._minimum_partition(n, mi)
        cut_d = calc._minimum_partition(n, mi_d)
        hn = H / torch.clamp(H.norm(dim=1, keepdim=True), min=1e-9)
        cos = float(((hn @ hn.T).sum() - n) / (n * (n - 1)))
        diff = max(0.0, 1.0 - cos)

        cur = max(0.0, (total - cut) / (n - 1))
        raw_dir = (cut / (n - 1)) * diff
        new = (cut_d / (n - 1)) * diff
        shipped.append(cur)
        undebiased.append(raw_dir)
        cand.append(new)
        print(f"  {name:>16} {cos:>+9.4f} {cur:>9.2f} {raw_dir:>9.3f} {new:>9.3f}")

    print()
    print(f"  현재 Φ 최대: {cases[int(np.argmax(shipped))][0]}"
          f"   방향만 최대: {cases[int(np.argmax(undebiased))][0]}"
          f"   후보 Φ 최대: {cases[int(np.argmax(cand))][0]}")
    print("  방향만 고치면 반대로 뒤집혀 완전 독립이 최고점을 받는다 —")
    print("  독립 신호끼리도 상호정보 추정치가 양수라 절단값이 0 으로 안 떨어지기 때문.")
    print("  뒤섞기로 그 바닥을 빼야 최대가 안쪽으로 온다.\n")
    return shipped, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=CELLS)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--phi-sanity", action="store_true",
                    help="only check whether Phi punishes or rewards collapse")
    ap.add_argument("--phi-candidate", action="store_true",
                    help="compare the shipped Phi against a direction-correct candidate")
    args = ap.parse_args()

    if args.phi_sanity:
        phi_sanity(cells=args.cells, hidden=args.hidden)
        return

    if args.phi_candidate:
        phi_candidate(cells=args.cells, hidden=args.hidden)
        return

    names = [t[0] for t in VERIFICATION_TESTS]
    print(f"\n  의식 검증 관문 감사 — 통과하면 안 되는 것들이 통과하는가")
    print(f"  cells={args.cells} dim={args.dim} hidden={args.hidden}\n")
    for label, _, why in CONTROLS + PROPOSALS:
        print(f"    {label:<12} {why}")
    print()

    header = f"  {'대조군':<12}" + "".join(f" {n[:9]:^10}" for n in names) + "  통과수"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grid = {}
    for label, cls, _ in CONTROLS + PROPOSALS:
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
        fooled = [lbl for lbl, _, _ in CONTROLS[1:] if grid[(lbl, test_name)]]  # proposals excluded
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
