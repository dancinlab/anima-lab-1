#!/usr/bin/env python3
"""bench_exchangeability.py — is there anything for a split rule to detect?

The gate audit measured mean tension at frozen cell counts (gap_vs_cells.txt):

    n        2        4        8       16       32
    mean  0.00743  0.01080  0.01321  0.01442  0.01493

Fit those five to the sampling bias of comparing a sample to its OWN mean,
E[(x_i - xbar)^2] = sigma^2 (1 - 1/n), ONE free parameter: R^2 = 0.988.

If that holds on live runs, the population-size dependence of the split signal
is an estimator artefact, not a fact about the population -- and the four f(n)
laws in bench_tension_feedback.py were cancelling a curve whose true form is
(1 - 1/n), which is not among {1, n, 1/n, 1/sqrt(n)}.

The complementary quantity is what a DEPARTURE from that curve would mean:
between-cell structure. Under exchangeability (cells are i.i.d. noise about a
common mean) the centred output Gram matrix has a flat spectrum and
participation ratio ~ n-1. Clustered cells give PR << n-1.

Arms -- all three freeze n, so nothing here is a population-control proposal:

  SAME     every cell gets the same x_input.  This is what ships.
  MULTI    same x_input, but drawn from K=4 fixed modes.  Tests whether the
           architecture can see structure in the world when the world has some.
  COMPETE  cell i receives x scaled by its own responsibility for x,
           r_i ∝ exp(-||out_i - x||^2 / mean_j ||out_j - x||^2), renormalised
           so sum(r) = n.  Total drive is CONSERVED; only its allocation
           differs.  No constant is introduced -- the temperature is the
           population's own mean distance.  Tests whether differential
           exposure is what the split signal is missing.

    .venv/bin/python bench_exchangeability.py
"""
import argparse, sys
import numpy as np, torch
import consciousness_engine as CE
from consciousness_engine import ConsciousnessEngine, ConsciousnessCell

_orig_fwd = ConsciousnessCell.forward
def _gain_fwd(self, x, tension, h):
    return _orig_fwd(self, x * getattr(self, '_gain', 1.0), tension, h)
ConsciousnessCell.forward = _gain_fwd


def participation_ratio(M):
    """Effective number of directions among rows of centred M. n iid rows -> ~n-1."""
    C = M - M.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(C @ C.T), 0, None)
    s = ev.sum()
    return float(s * s / (ev @ ev)) if s > 1e-12 else 0.0


def run(n_cells, steps, seed, arm, K=4, warm=None, phi_every=10):
    warm = steps // 4 if warm is None else warm
    torch.manual_seed(seed); np.random.seed(seed)
    eng = ConsciousnessEngine(cell_dim=64, hidden_dim=128, initial_cells=n_cells,
                              max_cells=n_cells, split_threshold=1e9,
                              merge_threshold=-1.0)
    eng._calibrated = True          # freeze n: no calibration, no split, no merge
    modes = [torch.randn(64) for _ in range(K)]

    T, SURP, PR, PHI, COS = [], [], [], [], []
    prev_out = None
    for t in range(steps):
        x = torch.randn(64) if arm != "multi" else modes[t % K]

        if arm == "compete" and prev_out is not None:
            # hidden[:cell_dim] is the same truncation the engine's own coupling
            # term uses at consciousness_engine.py:399.
            d = ((prev_out[:, :64] - x.unsqueeze(0)) ** 2).mean(1)
            r = torch.softmax(-d / d.mean().clamp(min=1e-9), 0) * n_cells
            for m, g in zip(eng.cell_modules, r):
                m._gain = float(g)

        eng.step(x_input=x)
        out = torch.stack([s.hidden for s in eng.cell_states])
        if prev_out is not None and out.shape == prev_out.shape:
            SURP.append(((out - prev_out) ** 2).mean(1).numpy())
        prev_out = out.clone()

        if t >= warm:
            T.append([s.tension_history[-1] for s in eng.cell_states])
            PR.append(participation_ratio(out.numpy()))
            u = out / out.norm(dim=1, keepdim=True).clamp(min=1e-9)
            g = (u @ u.T).numpy()
            COS.append((g.sum() - np.trace(g)) / max(n_cells*(n_cells-1), 1))
            if t % phi_every == 0:
                PHI.append(eng._measure_phi_iit())

    T = np.array(T).ravel()
    S = np.array(SURP[warm:]).ravel()
    return dict(n=n_cells, mean=T.mean(), loo=T.mean() * n_cells / (n_cells - 1.0),
                q90=np.quantile(T, .90), mx=T.max(),
                s_ratio=float(S.max() / max(np.quantile(S, .90), 1e-12)),
                pr=float(np.mean(PR)), phi=float(np.mean(PHI)),
                cos=float(np.mean(COS)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--arms", nargs="+", default=["same", "multi", "compete"])
    a = ap.parse_args()

    for arm in a.arms:
        print(f"\n=== {arm.upper()}   {a.seeds} seeds x {a.steps} steps ===")
        print(f"{'n':>4} {'mean T':>10} {'LOO T':>10} {'max/q90':>8} {'surp':>7} "
              f"{'PR':>7} {'PR/(n-1)':>9} {'cos':>8} {'Phi':>8}")
        rows = []
        for n in a.cells:
            rs = [run(n, a.steps, s, arm) for s in range(42, 42 + a.seeds)]
            g = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            rows.append(g)
            print(f"{n:>4} {g['mean']:>10.6f} {g['loo']:>10.6f} "
                  f"{g['mx']/max(g['q90'],1e-12):>8.3f} {g['s_ratio']:>7.3f} "
                  f"{g['pr']:>7.2f} {g['pr']/max(n-1,1):>9.3f} "
                  f"{g['cos']:>8.4f} {g['phi']:>8.4f}")
            sys.stdout.flush()
        nn = np.array([r['n'] for r in rows], float)
        T = np.array([r['mean'] for r in rows]); x = 1 - 1/nn
        s2 = (x @ T) / (x @ x); pred = s2 * x
        r2 = 1 - ((T - pred) ** 2).sum() / max(((T - T.mean()) ** 2).sum(), 1e-18)
        L = np.array([r['loo'] for r in rows])
        print(f"  fit T = sigma^2(1-1/n):  sigma^2={s2:.6f}  R^2={r2:.5f}")
        print(f"  LOO across n: {L.min():.6f}..{L.max():.6f} "
              f"({(L.max()/max(L.min(),1e-12)-1)*100:+.1f}% spread)")
