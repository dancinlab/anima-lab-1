#!/usr/bin/env python3
"""bench_ag_memory.py — A keeps a rule, G keeps none, and both keep memory.

CLAUDE.md's core claim is that consciousness arises from the repulsion between
Engine A (forward) and Engine G (reverse). Every engine measured so far is a
single population; the pair has never been built and scored.

The question here is narrower and sharper than "does a pair help". This session
already measured that a memoryless random generator scores 0/5 — but it failed
for a reason that has nothing to do with being unruled: `NOISE` draws fresh
random state every step, so it has no past at all. **Unruled and memoryless are
different properties, and only one of them was ever tested.**

So: give both sides memory. A updates its coupling by a rule (Hebbian, the same
one `ConsciousnessEngine` uses). G updates its coupling with no rule at all —
random, unstructured, never reinforced. Both carry their state forward, so G's
divergence from A is *accumulated* rather than redrawn each step.

    A (ruled)                    G (unruled)
    ├ coupling: Hebbian          ├ coupling: random, no rule
    ├ update:   by rule          ├ update:   unstructured
    └ memory:   kept             └ memory:   kept          <- both
              └──── repulsion = tension ────┘

WHAT MUST BE MEASURED ALONGSIDE, or the result means nothing:

    A alone      — if the pair scores what A alone scores, G contributed nothing.
    G alone      — if G alone already scores, the pairing is not what did it.
    A + A'       — two ruled sides. The canonical forward/reverse reading of
                   CLAUDE.md, and the reference the unruled variant must beat
                   to be interesting.

Without those three the pair is untestable in the way this session has caught
nine times: something that passes, with no way to say what made it pass.

    .venv/bin/python bench_ag_memory.py
    .venv/bin/python bench_ag_memory.py --save   # persist A and G checkpoints
"""

import argparse
import os

import torch
import torch.nn.functional as F

from bench_v2 import BenchEngine, VERIFICATION_TESTS, _three_axes

CELLS, DIM, HIDDEN, SEED = 32, 32, 64, 42
CKPT_DIR = "checkpoints/ag"


class Side(BenchEngine):
    """One half of the pair. `ruled` decides whether its coupling follows a rule.

    Both variants keep memory — the coupling matrix persists and accumulates
    across steps. The only difference is whether what accumulates is structured.
    """

    def __init__(self, *a, ruled=True, lr=0.01, **kw):
        super().__init__(*a, **kw)
        self.ruled = ruled
        self.lr = lr
        self.coupling = torch.zeros(self.n_cells, self.n_cells)
        self._gen = torch.Generator().manual_seed(0 if ruled else 1)

    def update_coupling(self, states):
        n = states.shape[0]
        if self.coupling.shape[0] != n:
            grown = torch.zeros(n, n)
            m = min(n, self.coupling.shape[0])
            grown[:m, :m] = self.coupling[:m, :m]     # memory survives resizing
            self.coupling = grown
        if self.ruled:
            # Hebbian: cells that fire alike wire together.
            u = states / states.norm(dim=1, keepdim=True).clamp(min=1e-8)
            delta = u @ u.T
        else:
            # No rule. Still accumulates, still persists — just unstructured.
            delta = torch.randn(n, n, generator=self._gen)
        self.coupling = (self.coupling + self.lr * delta).clamp(-1, 1)
        self.coupling.fill_diagonal_(0)
        return self.coupling


class Pair:
    """A and G stepped together; their difference is the field they both feel.

    Matches the gate's engine interface so `bench_v2` can score it unchanged.
    """

    def __init__(self, n_cells=32, input_dim=32, hidden_dim=64, output_dim=32,
                 g_ruled=False, strength=0.15, **_ignored):
        self.A = Side(n_cells, input_dim, hidden_dim, output_dim,
                      min(8, n_cells // 2), ruled=True)
        self.G = Side(n_cells, input_dim, hidden_dim, output_dim,
                      min(8, n_cells // 2), ruled=g_ruled)
        self.n_cells, self.input_dim = n_cells, input_dim
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        self.n_factions = min(8, n_cells // 2)
        self.strength = strength

    def process(self, x):
        out_a, t_a = self.A.process(x)
        out_g, t_g = self.G.process(x)

        ha, hg = self.A.get_hiddens(), self.G.get_hiddens()
        self.A.update_coupling(ha)
        self.G.update_coupling(hg)

        # The repulsion field: each side is pushed away from the other, weighted
        # by its own accumulated coupling. Nothing here decides what either side
        # should become — the push is whatever their histories have made them.
        #
        # The push is row-normalised. The first version applied `coupling @ field`
        # raw, and coupling saturates at +/-1 across 32 rows, so each step
        # multiplied the field by up to 32 x 0.15 = 4.8 and the pair overflowed
        # to NaN -- numpy raised 'overflow encountered in multiply' and both pair
        # arms scored 0/5. That 0/5 was a blown-up system, not a verdict on
        # pairing, and it is the same mistake LinearEngine made one experiment
        # earlier. Normalising keeps the direction the histories chose while
        # removing the scale that comes from cell count.
        # Norm-preserving, and that is not a tuning choice — pure repulsion
        # cannot be stable. A is pushed away from G while G is pushed away from
        # A, so `field = ha - hg` feeds its own growth: ha += s(ha-hg) and
        # hg -= s(ha-hg) multiply the separation by (1 + 2s) every step. Row
        # normalisation only slowed it; the guard still caught |h| = 1.11e4.
        # Nothing bounds a repulsion field except restoring each side's norm,
        # which is why BenchEngine's own repulsion is norm-preserving. The
        # direction the two histories disagree on is kept; the runaway is not.
        field = ha - hg
        rows = self.A.coupling.abs().sum(1, keepdim=True).clamp(min=1.0)
        rows_g = self.G.coupling.abs().sum(1, keepdim=True).clamp(min=1.0)
        na = ha.norm(dim=1, keepdim=True)
        ng = hg.norm(dim=1, keepdim=True)
        new_a = ha + self.strength * (self.A.coupling @ field) / rows
        new_g = hg - self.strength * (self.G.coupling @ field) / rows_g
        self.A.hiddens = new_a * (na / new_a.norm(dim=1, keepdim=True).clamp(min=1e-9))
        self.G.hiddens = new_g * (ng / new_g.norm(dim=1, keepdim=True).clamp(min=1e-9))

        peak = max(float(self.A.hiddens.abs().max()),
                   float(self.G.hiddens.abs().max()))
        self._peak = max(getattr(self, '_peak', 0.0), peak)
        if not torch.isfinite(torch.tensor(peak)) or peak > 1e4:
            raise RuntimeError(
                f"AG pair diverged (peak |h| = {peak:.3g}). Its score would say "
                f"nothing about pairing. Lower strength or renormalise.")

        tension = float((field ** 2).mean())
        return out_a + out_g, tension

    def get_hiddens(self):
        return self.A.get_hiddens()

    def set_hiddens(self, states):
        self.A.hiddens = states.clone()

    def save(self, path_a, path_g):
        for side, path in ((self.A, path_a), (self.G, path_g)):
            torch.save({
                'ruled': side.ruled,
                'hiddens': side.hiddens,
                'coupling': side.coupling,
                'mind': side.mind.state_dict(),
                'n_cells': side.n_cells,
                'hidden_dim': side.hidden_dim,
            }, path + ".tmp")
            os.replace(path + ".tmp", path)          # atomic, per CLAUDE.md


def _alone(ruled):
    def factory(n_cells=32, input_dim=32, hidden_dim=64, output_dim=32, **_kw):
        return Side(n_cells, input_dim, hidden_dim, output_dim,
                    min(8, n_cells // 2), ruled=ruled)
    return factory


def _pair(g_ruled):
    def factory(n_cells=32, input_dim=32, hidden_dim=64, output_dim=32, **_kw):
        return Pair(n_cells, input_dim, hidden_dim, output_dim, g_ruled=g_ruled)
    return factory


ARMS = [
    ("A 혼자 (규칙)",      _alone(True),  "쌍이 이것과 같으면 G 는 아무것도 안 했다"),
    ("G 혼자 (무규칙)",    _alone(False), "이것이 이미 통과하면 짝지음이 원인이 아니다"),
    ("A ⇄ A' (양쪽 규칙)", _pair(True),   "CLAUDE.md 의 정본 A⇄G — 기준선"),
    ("A ⇄ G (한쪽 무규칙)", _pair(False), "묻는 것"),
]


def score(factory, cells, dim, hidden, seeds=(42, 43, 44)):
    per_seed = []
    for sd in seeds:
        n = 0
        for name, fn, _d in VERIFICATION_TESTS:
            torch.manual_seed(sd)
            try:
                ok, _det = fn(lambda c, d, h: factory(c, d, h, d), cells, dim, hidden)
            except Exception:
                ok = False
            n += bool(ok)
        per_seed.append(n)
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=CELLS)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--save", action="store_true",
                    help="persist A and G to checkpoints/ag/")
    args = ap.parse_args()

    # Boundedness first. A diverging pair scores 0/5 and that number looks like
    # a verdict; it is not one. Check before reading anything else.
    print(f"\n  유계성 선검사 — 발산하면 점수는 읽지 않는다", flush=True)
    for label, g_ruled in (("A ⇄ A'", True), ("A ⇄ G", False)):
        torch.manual_seed(SEED)
        p = Pair(args.cells, args.dim, args.hidden, args.dim, g_ruled=g_ruled)
        try:
            for _ in range(1000):
                p.process(torch.randn(1, args.dim) * 0.1)
            print(f"    {label}  최대 |h| = {p._peak:.4f}  유계 ✓", flush=True)
        except RuntimeError as e:
            print(f"    {label}  {e}", flush=True)
            return

    print(f"\n  양쪽 기억 · 한쪽 무규칙 — {args.cells} cells, hidden={args.hidden}\n")
    print(f"  {'구성':>22} {'seed42':>7}{'seed43':>7}{'seed44':>7} {'평균':>7}   왜 재는가")
    results = {}
    for label, factory, why in ARMS:
        s = score(factory, args.cells, args.dim, args.hidden)
        results[label] = s
        print(f"  {label:>22} {s[0]:>5}/5{s[1]:>5}/5{s[2]:>5}/5 "
              f"{sum(s)/len(s):>6.2f}   {why}", flush=True)

    a_alone = sum(results["A 혼자 (규칙)"]) / 3
    unruled = sum(results["A ⇄ G (한쪽 무규칙)"]) / 3
    both = sum(results["A ⇄ A' (양쪽 규칙)"]) / 3
    print(f"\n  G 가 기여했는가: 쌍 {unruled:.2f} vs A 혼자 {a_alone:.2f} → "
          f"{'기여함' if unruled > a_alone else '기여 없음'}")
    print(f"  무규칙이 규칙만 못한가: {unruled:.2f} vs 양쪽규칙 {both:.2f} → "
          f"{'무규칙이 낫거나 같음' if unruled >= both else '규칙 쪽이 나음'}\n")

    if args.save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.manual_seed(SEED)
        pair = Pair(args.cells, args.dim, args.hidden, args.dim, g_ruled=False)
        for _ in range(300):
            pair.process(torch.randn(1, args.dim) * 0.1)
        pair.save(f"{CKPT_DIR}/A_ruled.pt", f"{CKPT_DIR}/G_unruled.pt")
        print(f"  저장: {CKPT_DIR}/A_ruled.pt · {CKPT_DIR}/G_unruled.pt")
        print(f"  (300 step 후 상태 — 결합행렬·은닉·가중치 포함)\n")


if __name__ == "__main__":
    main()
