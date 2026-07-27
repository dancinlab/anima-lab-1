#!/usr/bin/env python3
"""bench_ratchet_law31.py — does the Φ ratchet prevent collapse, or cause it?

CLAUDE.md records the ratchet as key #1 of "영속성의 3가지 열쇠" (Law 31), on
`PERSIST3`'s evidence: 1000 steps at 512 cells, no collapse, Φ growing ×62. The
ratchet restores prior state whenever Φ drops, and `PERSISTENCE` scores whether Φ
drops — so that benchmark may have confirmed the device rather than the property.
The two have never been run apart. `phi_ratchet` is a constructor flag, so they
can be.

There is a sharper reason to look. `ConsciousnessEngine._measure_phi_iit` — the
quantity the ratchet maximises — computes

    (total_mi − min_partition) / (n − 1)

which is the INVERTED direction this session corrected in `bench_v2`: it rises as
cells become more alike. So the ratchet does not restore the healthiest state. It
restores whichever state had the most correlated cells, and mean pairwise cosine
→ 1 is the collapse signature.

PRE-DECLARED PREDICTION, so this can come out wrong:

    ratchet ON ends with HIGHER mean pairwise cosine than ratchet OFF.

FALSIFIER: if cosine(on) ≤ cosine(off) across seeds, the reading above is wrong
and the ratchet is not driving collapse — say so and drop the claim.

RESULT: THE FALSIFIER FIRED. 1000 steps, 3 seeds, max 64 cells — cosine is
+0.4683 with the ratchet ON against +0.5526 OFF. LOWER, not higher. The reading
above is dropped.

The ratchet restored 29–64 times per run, so the device runs; it just does not
move the population. Φ trajectories overlap (ON 0.232/0.325/0.275 final vs OFF
0.359/0.228/0.244), cell growth is identical at 62–64 either way, and neither arm
satisfies the persistence rule. The ratchet shows neither the harm predicted here
nor the benefit Law 31 claims for it — so Law 31's key #1 is unsupported, but not
for the reason predicted. A quantity can point the wrong way and still fail to
steer the system, if what it gates is weak enough.

Reported per arm: corrected Φ(IIT) from `bench_v2.PhiIIT` (not the engine's own,
which is the thing under suspicion), the engine's internal Φ, mean pairwise
cosine, restore count, and whether `PERSISTENCE`'s own rule holds.

    .venv/bin/python bench_ratchet_law31.py --steps 1000 --seeds 3

SECOND QUESTION (`--gate`): would a ratchet have manufactured a pass?

`_verify_persistence` scores `(monotonic or recovers) AND d_ok AND i_ok AND c_ok`
— the Φ rule AND the three axes this session added. A ratcheted engine's Φ trace
freezes (0.351 → 0.351 → 0.406 → 0.406), and a frozen trace is monotonic
non-decreasing, so it satisfies the Φ rule outright. `--gate` reports the two
conjuncts separately: the Φ rule alone is the gate as it stood BEFORE this
session, and the axes are what catches a ratchet.

    .venv/bin/python bench_ratchet_law31.py --steps 1000 --seeds 3
    .venv/bin/python bench_ratchet_law31.py --gate
"""

import argparse

import torch

from bench_v2 import (PhiIIT, BenchEngine, measure_dual_phi, _three_axes)
from bench_verify_audit import NoiseEngine, factory_for
from consciousness_engine import ConsciousnessEngine


class CountingEngine(ConsciousnessEngine):
    """Identical, but counts how often the ratchet actually restores."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.restores = 0

    def _phi_ratchet_check(self):
        before = None
        if self._best_hiddens is not None:
            before = self.cell_states[0].hidden.clone()
        super()._phi_ratchet_check()
        if before is not None and not torch.equal(before, self.cell_states[0].hidden):
            self.restores += 1


def _cosine(engine):
    """Mean off-diagonal pairwise cosine of the cell hiddens. → 1 is collapse."""
    h = engine._get_hiddens_tensor().detach()
    n = h.shape[0]
    if n < 2:
        return float('nan')
    u = h / torch.clamp(h.norm(dim=1, keepdim=True), min=1e-9)
    return float(((u @ u.T).sum() - n) / (n * (n - 1)))


def run(ratchet, steps, seed, cells, dim):
    torch.manual_seed(seed)
    eng = CountingEngine(cell_dim=dim, hidden_dim=dim * 2, initial_cells=2,
                         max_cells=cells, phi_ratchet=ratchet)
    calc = PhiIIT()
    quarter, trace, cos_trace = max(steps // 4, 1), [], []
    for t in range(steps):
        eng.step(torch.randn(dim) * 0.1)
        if (t + 1) % quarter == 0:
            h = eng._get_hiddens_tensor().detach()
            # compute() returns (phi, components) — taking the tuple whole cost a
            # 50-minute run that died on the format string.
            trace.append(calc.compute(h)[0] if h.shape[0] >= 2 else 0.0)
            cos_trace.append(_cosine(eng))
    return {
        'phi': trace,
        'cos': cos_trace,
        'cos_final': cos_trace[-1] if cos_trace else float('nan'),
        'internal_phi': eng._measure_phi_iit(),
        'restores': eng.restores,
        'cells': eng.n_cells,
        # PERSISTENCE's own rule: grows, or recovers after dropping.
        'holds': bool(trace and (all(b >= a for a, b in zip(trace, trace[1:]))
                                 or trace[-1] >= max(trace) * 0.9)),
    }


def ratcheted(base_cls):
    """Wrap any gate-compatible engine with the ratchet, so the device can be
    tested apart from the engine that happens to carry it."""

    class Ratcheted(base_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._best_phi, self._best_h, self._restores = 0.0, None, 0
            # Its OWN counter. Reading the base class's `step_count` via getattr
            # silently gave NoiseEngine (which has no such attribute) a default
            # of 0, so `0 % 10 == 0` held every step and the noise arm was
            # ratcheted 1000 times against the real arm's 100 — 876 restores vs
            # 9, an artefact of the probe rather than of the engines.
            self._rt_step = 0

        def process(self, x):
            out = super().process(x)
            self._rt_step += 1
            if self._rt_step % 10 == 0:
                h = self.get_hiddens()
                phi, _ = measure_dual_phi(h, min(8, h.shape[0] // 2))
                if phi > self._best_phi:
                    self._best_phi, self._best_h = phi, h.clone()
                elif self._best_h is not None and phi < self._best_phi * 0.8:
                    self.hiddens = self._best_h.clone()
                    self._restores += 1
            return out

    Ratcheted.__name__ = f"Ratcheted{base_cls.__name__}"
    return Ratcheted


def gate_mode(cells, dim, hidden, seed=42):
    """Split PERSISTENCE into its conjuncts: the Φ rule (the gate as it stood
    before this session) and the three axes (what this session added)."""
    print(f"\n  래칫이 통과를 만들어내는가 — PERSISTENCE 를 두 조각으로\n")
    print(f"  {'엔진':>22} {'Φ 궤적':>36} {'단조':>5} {'회복':>5} "
          f"{'축':>5} {'구(舊)관문':>10} {'현(現)관문':>10}")

    arms = [("난수", NoiseEngine), ("난수+래칫", ratcheted(NoiseEngine)),
            ("실제", BenchEngine), ("실제+래칫", ratcheted(BenchEngine))]
    for label, cls in arms:
        torch.manual_seed(seed)
        eng = factory_for(cls, cells, dim, hidden)
        hist = []
        for step in range(1000):
            eng.process(torch.randn(1, dim) * 0.1)
            if step % 100 == 99:
                p, _ = measure_dual_phi(eng.get_hiddens(), min(8, cells // 2))
                hist.append(p)
        monotonic = all(hist[i] >= hist[i - 1] - 0.01 for i in range(1, len(hist)))
        recovers = hist[-1] >= max(hist[:len(hist) // 2]) * 0.8
        d_ok, i_ok, c_ok, _axes = _three_axes(eng, dim, cells)
        old = monotonic or recovers                       # pre-session gate
        new = old and d_ok and i_ok and c_ok              # current gate
        traj = " → ".join(f"{p:.2f}" for p in hist[:6]) + " …"
        print(f"  {label:>22} {traj:>36} {'O' if monotonic else 'X':>5} "
              f"{'O' if recovers else 'X':>5} "
              f"{('O' if d_ok and i_ok and c_ok else 'X'):>5} "
              f"{'PASS' if old else 'FAIL':>10} {'PASS' if new else 'FAIL':>10}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="Φ 규칙 vs 3축을 분리해 래칫이 통과를 만드는지 본다")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    args = ap.parse_args()

    if args.gate:
        gate_mode(args.cells, args.dim, args.dim * 2)
        return

    print(f"\n  Law 31 열쇠 #1 검증 — 래칫 없이도 붕괴가 없는가")
    print(f"  {args.steps} step · {args.seeds} seed · 최대 {args.cells} 세포\n")
    print(f"  {'래칫':>6} {'seed':>5} {'Φ(IIT) 궤적':>34} "
          f"{'코사인':>8} {'내부Φ':>8} {'복원':>5} {'세포':>5} {'유지':>5}")

    agg = {}
    for ratchet in (True, False):
        rows = []
        for seed in range(args.seeds):
            r = run(ratchet, args.steps, seed, args.cells, args.dim)
            rows.append(r)
            traj = " → ".join(f"{p:.3f}" for p in r['phi'])
            print(f"  {'ON' if ratchet else 'OFF':>6} {seed:>5} {traj:>34} "
                  f"{r['cos_final']:>+8.4f} {r['internal_phi']:>8.3f} "
                  f"{r['restores']:>5} {r['cells']:>5} "
                  f"{'O' if r['holds'] else 'X':>5}")
        agg[ratchet] = rows

    on = sum(r['cos_final'] for r in agg[True]) / args.seeds
    off = sum(r['cos_final'] for r in agg[False]) / args.seeds
    print(f"\n  평균 코사인   래칫 ON {on:+.4f}   OFF {off:+.4f}")
    print(f"  예측(래칫이 붕괴를 유발): ON > OFF  →  "
          f"{'적중' if on > off else '빗나감 — 주장 철회'}")
    held_off = sum(r['holds'] for r in agg[False])
    print(f"  래칫 OFF에서 붕괴 없음: {held_off}/{args.seeds} seed"
          f"  →  {'열쇠 #1은 불필요' if held_off == args.seeds else '래칫이 실제로 기여'}")
    print()


if __name__ == "__main__":
    main()
