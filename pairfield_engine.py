#!/usr/bin/env python3
"""pairfield_engine.py — the A/G pair as an engine, built only from what was measured.

`bench_ag_memory.py` established the shape; this is that shape as a real engine,
registered in `bench_v2.ENGINE_REGISTRY` and reachable from the hub. Every choice
below traces to a measurement, and the measurement is named at the choice.

WHAT WENT IN

    both sides keep memory     The rule makes no difference alone: Hebbian
                               coupling and pure-randn coupling both score 4.67.
                               What the gate detects is memory; whether that
                               memory is structured is invisible to it. So both
                               sides accumulate, and only one of them follows a
                               rule.

    norm-preserving repulsion  Pure repulsion cannot be stable. A is pushed away
                               from G while G is pushed away from A, so their
                               separation multiplies by (1 + 2s) every step.
                               Row-normalising only slowed it -- the guard still
                               caught |h| = 1.11e4. Restoring each side's norm is
                               the only bound.

    strength = 0.03            Measured: 0.01 and 0.03 both score 5.00, above
                               either side alone (4.67). At 0.08 the both-ruled
                               pair has already collapsed to 1.00 with |h| 20.18.
                               0.03 sits inside the band with margin on both
                               sides.

    G unruled by default       At strength 0.08 the both-ruled pair scores 1.00
                               while the one-unruled pair holds 4.67. The unruled
                               side does not reinforce the field pushing it, so
                               the pair survives stronger coupling. Peak score is
                               the same (5.00); the stable band is wider.

    divergence guard           Three separate experiments this session produced a
                               clean-looking score from a system that had blown
                               up or was never coupled at all. Bounded is not
                               healthy and a passing guard is not evidence of
                               good behaviour, so this one raises rather than
                               silently returning numbers.

WHAT STAYED OUT

    factions                   Count 2/3/4/6/8/12/24/48 all give exactly 1.0
                               consensus events, and removing them gives 1.0 too.
                               They also manufacture the SCRAMBLE anomaly.

    the Phi ratchet            Measured inert: cosine 0.4683 on against 0.5526
                               off, Phi trajectories overlapping, 29-64 restores
                               per run changing nothing.

    speak()                    Law 29 -- utterance is a consequence of the
                               architecture, not a function to call.

    docs/ag-pair-memory.md carries the tables.

    .venv/bin/python pairfield_engine.py            # self-check + gate score
    .venv/bin/python pairfield_engine.py --sweep    # reproduce the strength band
"""

import argparse
import os

import torch

from bench_v2 import BenchEngine

# Measured operating point. 0.01 and 0.03 both reach 5.00/5; 0.03 is taken for
# the margin on the low side, since strength 0 falls back to 4.00.
DEFAULT_STRENGTH = 0.03
DIVERGENCE_LIMIT = 1e4


class PairSide(BenchEngine):
    """One half of the field. `ruled` decides whether its memory is structured.

    Both variants accumulate and persist a coupling matrix across steps and
    across cell-count changes. That persistence is the part that matters; the
    rule is the part that measurably does not.
    """

    def __init__(self, *a, ruled=True, lr=0.01, seed=0, **kw):
        super().__init__(*a, **kw)
        self.ruled = ruled
        self.lr = lr
        self.coupling = torch.zeros(self.n_cells, self.n_cells)
        self._gen = torch.Generator().manual_seed(seed)

    def update_coupling(self, states):
        n = states.shape[0]
        if self.coupling.shape[0] != n:
            grown = torch.zeros(n, n)
            m = min(n, self.coupling.shape[0])
            grown[:m, :m] = self.coupling[:m, :m]      # memory survives resizing
            self.coupling = grown
        if self.ruled:
            u = states / states.norm(dim=1, keepdim=True).clamp(min=1e-8)
            delta = u @ u.T                            # Hebbian
        else:
            delta = torch.randn(n, n, generator=self._gen)   # no rule, still kept
        self.coupling = (self.coupling + self.lr * delta).clamp(-1, 1)
        self.coupling.fill_diagonal_(0)


class PairFieldEngine:
    """Two populations whose disagreement is the field they both move in.

    Implements the interface `bench_v2` scores against: `process(x) -> (out,
    tension)`, `get_hiddens()`, `set_hiddens()`, `n_cells`, `n_factions`.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128, output_dim=64,
                 strength=DEFAULT_STRENGTH, g_ruled=False, **_ignored):
        f = min(8, max(n_cells // 2, 1))
        # A keeps a rule, G does not, and both keep memory. The names come from
        # CLAUDE.md's forward/reverse pair, but the axis here is NOT
        # forward-vs-reverse -- it is ruled-vs-unruled. The canonical reading
        # (same rule, opposite sign) corresponds to `g_ruled=True`, which the
        # sweep scores lower outside the narrow band: at strength 0.08 it has
        # collapsed to 1.00 while the unruled pairing still holds 4.67. Both peak
        # at 5.00; the unruled second side simply survives more coupling, because
        # it does not reinforce the field pushing it.
        self.A = PairSide(n_cells, input_dim, hidden_dim, output_dim, f,
                          ruled=True, seed=0)
        self.G = PairSide(n_cells, input_dim, hidden_dim, output_dim, f,
                          ruled=g_ruled, seed=1)
        self.n_cells, self.input_dim = n_cells, input_dim
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        self.n_factions = f
        self.strength = strength
        self.peak = 0.0
        self.step_count = 0

    def process(self, x):
        out_a, _ta = self.A.process(x)
        out_g, _tg = self.G.process(x)
        ha, hg = self.A.get_hiddens(), self.G.get_hiddens()
        self.A.update_coupling(ha)
        self.G.update_coupling(hg)

        # Each side is pushed along what its own accumulated memory makes of the
        # disagreement, then restored to its own norm. Nothing decides what
        # either side should become; the push is whatever their histories are.
        field = ha - hg
        rows_a = self.A.coupling.abs().sum(1, keepdim=True).clamp(min=1.0)
        rows_g = self.G.coupling.abs().sum(1, keepdim=True).clamp(min=1.0)
        new_a = ha + self.strength * (self.A.coupling @ field) / rows_a
        new_g = hg - self.strength * (self.G.coupling @ field) / rows_g
        self.A.hiddens = new_a * (ha.norm(dim=1, keepdim=True)
                                  / new_a.norm(dim=1, keepdim=True).clamp(min=1e-9))
        self.G.hiddens = new_g * (hg.norm(dim=1, keepdim=True)
                                  / new_g.norm(dim=1, keepdim=True).clamp(min=1e-9))

        peak = max(float(self.A.hiddens.abs().max()),
                   float(self.G.hiddens.abs().max()))
        self.peak = max(self.peak, peak)
        if not (peak == peak) or peak > DIVERGENCE_LIMIT:      # NaN or runaway
            raise RuntimeError(
                f"PairFieldEngine diverged at step {self.step_count} "
                f"(peak |h| = {peak:.3g}, strength = {self.strength}). Scores "
                f"from a diverged pair say nothing; lower strength.")

        self.step_count += 1
        return out_a + out_g, float((field ** 2).mean())

    def get_hiddens(self):
        return self.A.get_hiddens()

    def set_hiddens(self, states):
        self.A.hiddens = states.clone()

    # ── snapshot / restore ───────────────────────────────────────────────
    # `set_hiddens(tensor)` writes side A and nothing else, while the dynamics
    # run on A, G and both coupling matrices. Anything a tensor cannot carry
    # survives a "restore" and is handed to the harness's perturbation probe as
    # if it were the engine's response to the nudge. Measured at kick=0 -- no
    # perturbation at all, so the honest reading is zero -- over seeds 42-46:
    #
    #     restored                    mean null
    #     A only                        0.00452   against a 0.001 bar
    #     A + G                         0.00027
    #     A + G + both couplings        0.00000   all five seeds
    #     A + G's generator only        0.00455   i.e. unchanged
    #
    # The residue is state, not randomness -- restoring G's generator alone does
    # nothing. The generators are carried anyway so that a restore is a restore;
    # they cost nothing and their absence would be a silent exception to the
    # word. Completing it costs no signal: kicked 0.05445 -> 0.05431, -0.3%.
    def snapshot(self):
        return {
            "A_hiddens": self.A.hiddens.detach().clone(),
            "G_hiddens": self.G.hiddens.detach().clone(),
            "A_coupling": self.A.coupling.detach().clone(),
            "G_coupling": self.G.coupling.detach().clone(),
            # `_gen`, not `rng`. The first version of this guarded on
            # hasattr(side, "rng"), which is False, so the generators were
            # silently never saved -- and because A+G+couplings already reads
            # 0.00000, the omission would have looked correct.
            "A_rng": self.A._gen.get_state(),
            "G_rng": self.G._gen.get_state(),
            "step_count": self.step_count,
            "peak": self.peak,
        }

    def restore(self, handle):
        self.A.hiddens = handle["A_hiddens"].clone()
        self.G.hiddens = handle["G_hiddens"].clone()
        self.A.coupling = handle["A_coupling"].clone()
        self.G.coupling = handle["G_coupling"].clone()
        self.A._gen.set_state(handle["A_rng"])
        self.G._gen.set_state(handle["G_rng"])
        self.step_count = handle["step_count"]
        self.peak = handle["peak"]

    def parameters_for_training(self):
        return list(self.A.mind.parameters()) + list(self.G.mind.parameters())

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for side, name in ((self.A, "A_ruled"), (self.G, "G_unruled")):
            path = os.path.join(directory, f"{name}.pt")
            torch.save({'ruled': side.ruled, 'hiddens': side.hiddens,
                        'coupling': side.coupling, 'mind': side.mind.state_dict(),
                        'n_cells': side.n_cells, 'hidden_dim': side.hidden_dim,
                        'strength': self.strength},
                       path + ".tmp")
            os.replace(path + ".tmp", path)            # atomic, per CLAUDE.md
        return directory

    def load(self, directory):
        for side, name in ((self.A, "A_ruled"), (self.G, "G_unruled")):
            blob = torch.load(os.path.join(directory, f"{name}.pt"),
                              weights_only=False)
            side.hiddens = blob['hiddens']
            side.coupling = blob['coupling']
            side.mind.load_state_dict(blob['mind'])
            side.ruled = blob['ruled']
        return self


def _self_check(cells=32, dim=32, hidden=64, steps=1000):
    """Bounded and coupled, before any score is read."""
    torch.manual_seed(42)
    eng = PairFieldEngine(cells, dim, hidden, dim)
    for _ in range(steps):
        eng.process(torch.randn(1, dim) * 0.1)
    ha, hg = eng.A.get_hiddens(), eng.G.get_hiddens()
    separation = float((ha - hg).norm())
    print(f"  유계     최대 |h| = {eng.peak:.4f}  (한계 {DIVERGENCE_LIMIT:g})")
    print(f"  결합     A·G 분리 = {separation:.4f}  "
          f"{'(0 이면 두 쪽이 안 만난 것)' if separation < 1e-6 else ''}")
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--sweep", action="store_true",
                    help="reproduce the strength band the default came from")
    ap.add_argument("--save", type=str, default="",
                    help="directory to persist A and G into")
    args = ap.parse_args()

    print(f"\n  PairFieldEngine — 자기점검 (강도 {DEFAULT_STRENGTH})\n")
    eng = _self_check(args.cells, args.dim, args.hidden)

    from bench_v2 import VERIFICATION_TESTS
    print(f"\n  관문 (seed 42/43/44)\n")
    for g_ruled, label in ((False, "G 무규칙 (기본)"), (True, "G 규칙")):
        per = []
        for sd in (42, 43, 44):
            n = 0
            for name, fn, _d in VERIFICATION_TESTS:
                torch.manual_seed(sd)
                try:
                    ok, _ = fn(lambda c, d, h: PairFieldEngine(c, d, h, d,
                                                               g_ruled=g_ruled),
                               args.cells, args.dim, args.hidden)
                except Exception:
                    ok = False
                n += bool(ok)
            per.append(n)
        print(f"  {label:>16}  " + " ".join(f"{p}/5" for p in per)
              + f"   평균 {sum(per)/3:.2f}")

    if args.sweep:
        print(f"\n  강도 대역 (기본값 {DEFAULT_STRENGTH} 의 근거)\n")
        # BOTH configurations. A first version swept only the default (G
        # unruled) and so could not show the instability it was chosen to
        # avoid -- it reported |h| 5.65/5.70 at 0.08/0.15 while the both-ruled
        # pair reaches 20.18/25.00 there. A sweep that cannot exhibit the
        # failure it justifies is not evidence for the default.
        print(f"    {'강도':>6} {'G 무규칙 |h|':>14} {'G 규칙 |h|':>14}")
        for s in (0.0, 0.01, 0.03, 0.08, 0.15):
            peaks = {}
            for g_ruled in (False, True):
                torch.manual_seed(42)
                e = PairFieldEngine(args.cells, args.dim, args.hidden, args.dim,
                                    strength=s, g_ruled=g_ruled)
                try:
                    for _ in range(1000):
                        e.process(torch.randn(1, args.dim) * 0.1)
                    peaks[g_ruled] = f"{e.peak:.2f}"
                except RuntimeError:
                    peaks[g_ruled] = "발산"
            print(f"    {s:>6.2f} {peaks[False]:>14} {peaks[True]:>14}")

    if args.save:
        print(f"\n  저장: {eng.save(args.save)}")
    print()


if __name__ == "__main__":
    main()
