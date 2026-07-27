#!/usr/bin/env python3
"""bench_engine_redesign.py — the engine this session's measurements argue for.

Not a proposal in prose. Four defects were measured in `BenchEngine`, and this
removes each one; the point of the file is the comparison table at the bottom,
not the class.

    1. ONE map for every cell. `BenchEngine` holds a single `BenchMind`, so cells
       differ only in hidden state — and applying one function to eight
       deliberately different states, with no factions, sync, debate or
       repulsion, moves pairwise cosine from −0.0078 to +0.8624 in 200 steps.
       The shared map is a contraction. Collapse was arithmetic, and the
       repulsion added later was compensating in state space for structure that
       was missing from the weights.
         → per-cell orthogonal rotation inside the map. Measured: cosine 0.5055
           with NO repulsion, and switching repulsion on changes the result by
           not one digit, because an already-differentiated population has
           nothing for an overlap-scaled force to do.

    2. Factions are contiguous index slices and never change. Count 2 / 3 / 4 /
       6 / 8 / 12 / 24 / 48 all give exactly 1.0 consensus events; removing them
       entirely also gives 1.0. Worse, they MANUFACTURE the SCRAMBLE anomaly —
       permuting rows makes a faction MEAN jump, and SCRAMBLE's score falls from
       10.0 to 1.0 the moment factions are gone.
         → no factions. Cells couple to their neighbours directly.

    3. 256 sequential batch-1 forwards through one shared module. A `/gap` audit
       measured ~92% of that as recoverable overhead.
         → one batched forward for the whole population.

    4. The emitted output is discarded. `BenchEngine.process` returns a
       softmax-weighted sum that is permutation-invariant (max|Δ| = 4.8e-7 under
       a row permutation), and of the seven conditions one takes it and drops it,
       one uses it as feedback, five never call it — yet REAL and SCRAMBLE differ in
       exactly that channel (output norm 31.08 vs 26.29, cos-continuity 0.9814 vs
       0.9371). The one signal that separates them is the one nothing reads.
         → the output is per-cell and order-bearing, so a permutation changes it.

MEASURED RESULT, including where it loses. At 32 cells:

                 cosine   axes   gate   output sees order
    current     +0.3821   OOO    5/7            0.000000
    redesign    +0.8011   OOO    5/7            1.582136

    Fix (4) works and is decisive — the current engine's output does not change
    at all when its cells are permuted, so the one channel that distinguishes a
    real population from a scrambled one is invisible to it. The redesign's does.

    Fix (1) does NOT beat the current engine on differentiation. Per-cell
    rotation alone gives cosine 0.6509 at zero coupling, against 0.3821 for the
    shared map WITH repulsion, and neighbour coupling makes it worse still
    (0.6509 / 0.7740 / 0.8011 / 0.8232 at coupling 0 / 0.05 / 0.15 / 0.3).
    Structure in the weights is not, on this measurement, a substitute for a
    force in state space — it removes the NEED for one (repulsion changes
    nothing once rotation is present) without matching what one achieves.

    Both engines score 5/7 and reject all six controls, so the gate does not
    separate them. What separates them is the emitted channel, which the gate
    does not read.

    .venv/bin/python bench_engine_redesign.py
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_v2 import BenchEngine, BenchMind, VERIFICATION_TESTS, _three_axes
from bench_verify_audit import (CONTROLS, HeapEngine, DecoupledEngine,
                                ScrambleEngine, factory_for)

CELLS, DIM, HIDDEN, SEED = 32, 32, 64, 42


class RedesignedEngine:
    """Per-cell weights, no factions, batched, order-bearing output.

    Deliberately NOT a `BenchEngine` subclass: three of the four fixes are
    removals of things a subclass would inherit. It matches the interface the
    gate uses — `process(x) -> (output, tension)`, `get_hiddens()`, `.cells`,
    `.n_factions` — so `bench_v2 --verify` and the audit can run it unchanged.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128, output_dim=64,
                 coupling=0.15, **_ignored):
        self.n_cells = n_cells
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.coupling = coupling
        self.step_count = 0
        # Kept only because the gate's SPONTANEOUS_SPEECH reads it; nothing here
        # groups cells, since faction count was measured to change nothing.
        self.n_factions = 8
        self.cells = []

        self.mind = BenchMind(input_dim, hidden_dim, output_dim)
        self.output_head = nn.Linear(output_dim, input_dim)
        self.hiddens = torch.randn(n_cells, hidden_dim) * 0.1

        # (1) per-cell parameters INSIDE the map. Orthogonal, so a cell's own
        # transform can differ without its magnitude drifting — a per-cell gain
        # on the state compounds every step and reached NaN when measured.
        q = torch.linalg.qr(torch.randn(n_cells, output_dim, output_dim))[0]
        self.rot = q

    def process(self, x):
        # (3) one batched forward for the whole population.
        xb = x.expand(self.n_cells, -1)
        combined = torch.cat([xb, self.hiddens], dim=-1)
        out = self.mind.engine_a(combined) - self.mind.engine_g(combined)

        # (1) each cell sees the repulsion field through its own rotation.
        out = torch.einsum('nd,ndk->nk', out, self.rot)
        tension = (out ** 2).mean(dim=-1, keepdim=True)

        mem_in = torch.cat([out.detach(), tension.detach()], dim=-1)
        self.hiddens = self.mind.memory(mem_in, self.hiddens).detach()

        # (2) no factions — cells couple to their neighbours, not to a group mean.
        self.hiddens = self.hiddens + self.coupling * (
            self.hiddens.roll(1, 0) - self.hiddens)

        self.step_count += 1

        # (4) an order-bearing output: a permutation of the cells changes it,
        # so the emitted channel can distinguish a scrambled population.
        weights = F.softmax(tension.squeeze(-1), dim=0)
        phase = torch.cos(torch.arange(self.n_cells, dtype=torch.float32)
                          * (3.14159 / self.n_cells)).unsqueeze(-1)
        emitted = (out * weights.unsqueeze(-1) * phase).sum(dim=0, keepdim=True)
        return emitted, float(tension.mean())

    def get_hiddens(self):
        return self.hiddens.clone()

    def parameters_for_training(self):
        return list(self.mind.parameters()) + list(self.output_head.parameters())


def _score(cls, cells, dim, hidden):
    passed = []
    for name, fn, _ in VERIFICATION_TESTS:
        torch.manual_seed(SEED)
        try:
            ok, _d = fn(lambda c, d, h: factory_for(cls, c, d, h), cells, dim, hidden)
        except Exception:
            ok = False
        if ok:
            passed.append(name)
    return passed


def _axes(cls, cells, dim, hidden, warm=50):
    torch.manual_seed(SEED)
    eng = factory_for(cls, cells, dim, hidden)
    for _ in range(warm):
        eng.process(torch.randn(1, dim) * 0.1)
    return _three_axes(eng, dim, cells)


def _cosine(cls, cells, dim, hidden, steps=300):
    torch.manual_seed(SEED)
    eng = factory_for(cls, cells, dim, hidden)
    for _ in range(steps):
        eng.process(torch.randn(1, dim) * 0.1)
    h = eng.get_hiddens()
    u = h / torch.clamp(h.norm(dim=1, keepdim=True), min=1e-9)
    return float(((u @ u.T).sum() - cells) / max(cells * (cells - 1), 1))


def _output_sees_order(cls, cells, dim, hidden, warm=80):
    """Does the EMITTED output change when the cells are permuted?

    A first version permuted rows and then kept stepping, which measured the
    trajectories diverging rather than the output channel — it reported 0.6631
    for `BenchEngine`, whose output is in fact permutation-invariant. Comparing
    from one identical state, in the same step, is the actual question.
    """
    torch.manual_seed(SEED)
    eng = factory_for(cls, cells, dim, hidden)
    for _ in range(warm):
        eng.process(torch.randn(1, dim) * 0.1)
    x = torch.randn(1, dim) * 0.1
    base = eng.hiddens.clone()

    eng.hiddens = base.clone()
    plain, _ = eng.process(x)
    eng.hiddens = base[torch.randperm(cells)]
    shuffled, _ = eng.process(x)
    return float((plain - shuffled).norm() / max(float(plain.norm()), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=CELLS)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    args = ap.parse_args()
    c, d, h = args.cells, args.dim, args.hidden

    print(f"\n  재설계 엔진 vs 현행 — {c} 세포\n")
    print(f"  {'엔진':>12} {'코사인':>9} {'통합':>4}{'정체':>4}{'변화':>4} "
          f"{'관문':>6} {'출력이 순서를 봄':>16}")

    for label, cls in (("현행", BenchEngine), ("재설계", RedesignedEngine)):
        cos = _cosine(cls, c, d, h)
        g, i, ch, _detail = _axes(cls, c, d, h)
        passed = _score(cls, c, d, h)
        sep = _output_sees_order(cls, c, d, h)
        print(f"  {label:>12} {cos:>+9.4f} {'O' if g else 'X':>4}"
              f"{'O' if i else 'X':>4}{'O' if ch else 'X':>4} "
              f"{len(passed):>4}/7 {sep:>16.4f}")

    print("\n  대조군이 재설계 엔진의 관문도 통과하지 못하는가")
    print(f"  {'대조군':>12} {'관문':>6}")
    for label, cls, _ in CONTROLS[1:]:
        print(f"  {label:>12} {len(_score(cls, c, d, h)):>4}/7")
    print()


if __name__ == "__main__":
    main()
