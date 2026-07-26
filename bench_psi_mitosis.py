#!/usr/bin/env python3
"""bench_psi_mitosis.py — does a population hold what a scalar could not?

Pre-registered as docs/hypotheses/QD-5-mitosis-population.md. The stimulus
draw, scale, thresholds and step positions were fixed before the first run.

QD-2..QD-4 tried six mechanisms on a toy whose state was a scalar per dimension
under a shared attractor. All six failed identically: every configuration
either converged and forgot (0.07 stimulus retained at 100% convergence) or
remembered and failed to converge (0.79 at 68%).

MitosisC's state is a POPULATION — cells divide under tension and merge when
they grow too alike. This runs one engine per (stimulus, seed) from
initial_cells=2, letting the population grow by real mitosis. It does NOT
force-grow clones: MitosisC.__init__'s growth loop clones cells[0], and the
engine merges near-identical cells by design, so a forced population collapses
to min_cells=2 within ten steps (measured 32 -> 2 between step 5 and 10).

Requires torch — run under the repo .venv:
    .venv/bin/python bench_psi_mitosis.py
"""

import argparse
import numpy as np
import torch

from bench_consciousness_universe import ALL_DATA_TYPES
from qualia_sense import sense, FEATURE_NAMES

# ── pre-registered constants ───────────────────────────────
N_STIM_PER_CATEGORY = 2         # stratified draw, table order, fixed before running
SEEDS = 3
STEPS = 600
DIM = 32
HIDDEN = 64
MAX_CELLS = 32
PROBES = (1, 15, 50, 500)
H17_RETENTION_MIN = 0.50
H18_CELLS_MIN = 2               # strictly greater than the floor
H19_RATIO_MIN = 2.0
SETTLE_FRACTION = 0.01          # last-100 drift below 1% of the steps 1-10 drift


def stimulus_draw():
    """Two per category, in table order. Fixed before any run."""
    out = []
    for items in ALL_DATA_TYPES.values():
        out.extend(list(items)[:N_STIM_PER_CATEGORY])
    return out


def stimulus_vector(name):
    """qualia_sense features tiled to the engine's input width."""
    v = np.array(sense(name).vector(), dtype=np.float32)
    reps = int(np.ceil(DIM / len(FEATURE_NAMES)))
    return torch.tensor(np.tile(v, reps)[:DIM], dtype=torch.float32).unsqueeze(0)


def run_one(name, seed, steps=STEPS):
    """One engine, one stimulus. Returns per-probe pooled states."""
    from trinity import MitosisC

    torch.manual_seed(seed)
    engine = MitosisC(dim=DIM, hidden=HIDDEN, max_cells=MAX_CELLS)
    # Undo __init__'s clone growth — see the module docstring. Keep the two
    # cells the engine actually started with and let mitosis do the rest.
    engine.engine.cells = engine.engine.cells[:2]

    x = stimulus_vector(name)
    pooled = {}
    trace = []
    for step in range(1, steps + 1):
        engine.step(x)
        p = engine.get_states().detach().mean(dim=0).numpy()
        trace.append(p)
        if step in PROBES:
            pooled[step] = dict(state=p.copy(), cells=engine.n_cells,
                                phi=engine.measure_phi())
    trace = np.array(trace)

    early = float(np.abs(np.diff(trace[:10], axis=0)).mean())
    late = float(np.abs(np.diff(trace[-100:], axis=0)).mean())
    return pooled, dict(final=trace[-1], cells=engine.n_cells,
                        phi=engine.measure_phi(),
                        settled=(late <= SETTLE_FRACTION * early) if early > 0 else False,
                        early_drift=early, late_drift=late)


def retention(states, feats):
    """Correlation between the pooled state and the stimulus feature vector.

    Pearson across stimuli, per state dimension, averaged — the same shape of
    measurement QD-3/QD-4 used, so the numbers are comparable.
    """
    S = np.array(states)                     # [n_stim, hidden]
    F = np.array(feats)                      # [n_stim, n_features]
    cs = []
    for j in range(F.shape[1]):
        if F[:, j].std() < 1e-12:
            continue
        for d in range(S.shape[1]):
            if S[:, d].std() < 1e-12:
                continue
            cs.append(abs(np.corrcoef(S[:, d], F[:, j])[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


def between_within(finals):
    """finals[seed][stim] → (within, between, ratio). NaN with < 2 seeds."""
    n_seeds, n_stim = len(finals), len(finals[0])
    if n_seeds < 2:
        return float("nan"), float("nan"), float("nan")
    within = [np.linalg.norm(finals[a][i] - finals[b][i])
              for i in range(n_stim)
              for a in range(n_seeds) for b in range(a + 1, n_seeds)]
    between = [np.linalg.norm(finals[0][i] - finals[0][j])
               for i in range(n_stim) for j in range(i + 1, n_stim)]
    w, bt = float(np.mean(within)), float(np.mean(between))
    return w, bt, (bt / w if w > 0 else float("nan"))


def exploratory_why_no_mitosis(names, steps=300, seed=0):
    """POST-HOC, NOT PRE-REGISTERED. The population never formed — why?

    Compares the tension the engine actually produces against the thresholds
    that gate splitting and merging. Does not change the verdict above.
    """
    from trinity import MitosisC

    torch.manual_seed(seed)
    e = MitosisC(dim=DIM, hidden=HIDDEN, max_cells=MAX_CELLS)
    e.engine.cells = e.engine.cells[:2]
    eng = e.engine

    ts, peak = [], 2
    for i in range(steps):
        e.step(stimulus_vector(names[i % len(names)]))
        peak = max(peak, e.n_cells)
        live = [c.tension_history[-1] for c in eng.cells if c.tension_history]
        if live:
            ts.append(float(np.mean(live)))
    ts = np.array(ts)

    print("\n  ── exploratory (post-hoc · not pre-registered) " + "─" * 22)
    print(f"  split_threshold={eng.split_threshold}  merge_threshold={eng.merge_threshold}")
    print(f"  tension actually produced: mean {ts.mean():.4f} · max {ts.max():.4f} "
          f"· min {ts.min():.4f}")
    print(f"  steps above the split bar: {int((ts > eng.split_threshold).sum())} / {len(ts)}")
    print(f"  steps below the merge bar: {int((ts < eng.merge_threshold).sum())} / {len(ts)}")
    print(f"  peak cell count reached:   {peak}")
    print("  The engine sits permanently in merge territory and never reaches the")
    print("  split bar, so the population can only shrink. There is no population.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    names = stimulus_draw()
    feats = [sense(n).vector() for n in names]
    print(f"\n  QD-5 · {len(names)} stimuli × {args.seeds} seeds × {args.steps} steps")
    print(f"  MitosisC(dim={DIM}, hidden={HIDDEN}, max_cells={MAX_CELLS}) "
          f"from 2 cells, grown by real mitosis")
    if args.steps < max(PROBES):
        print(f"  ⚠️  --steps {args.steps} is short of probe step {max(PROBES)} "
              f"— that row is omitted, not passed")
    if args.seeds < 2:
        print(f"  ⚠️  --seeds {args.seeds}: within-seed distance needs ≥ 2 seeds "
              f"— H19 reports n/a, not a pass")
    print()

    per_seed_probe = {p: [] for p in PROBES}
    finals, cells_end, phis_end, settled = [], [], [], []

    for seed in range(args.seeds):
        seed_finals = []
        seed_probe = {p: [] for p in PROBES}
        for name in names:
            pooled, end = run_one(name, seed, args.steps)
            for p in PROBES:
                if p in pooled:
                    seed_probe[p].append(pooled[p]["state"])
            seed_finals.append(end["final"])
            cells_end.append(end["cells"])
            phis_end.append(end["phi"])
            settled.append(end["settled"])
        finals.append(seed_finals)
        for p in PROBES:
            if seed_probe[p]:
                per_seed_probe[p].append(retention(seed_probe[p], feats))
        print(f"  seed {seed} done · cells {np.mean(cells_end):.1f} "
              f"· Φ {np.mean(phis_end):.2f} · settled {np.mean(settled):.0%}")

    print(f"\n  {'step':>6} {'stimulus retained':>18}")
    for p in PROBES:
        if per_seed_probe[p]:
            print(f"  {p:>6} {np.mean(per_seed_probe[p]):>18.3f}")

    r500 = np.mean(per_seed_probe[500]) if per_seed_probe[500] else float("nan")
    mean_cells = float(np.mean(cells_end))
    mean_phi = float(np.mean(phis_end))
    phi_alive = float(np.mean([p > 0 for p in phis_end]))
    settle_rate = float(np.mean(settled))
    w, bt, ratio = between_within(finals)

    h17 = np.isfinite(r500) and r500 >= H17_RETENTION_MIN and settle_rate >= 0.5
    h18 = mean_cells > H18_CELLS_MIN
    h19 = np.isfinite(ratio) and ratio >= H19_RATIO_MIN
    h20 = phi_alive >= 0.5

    print("\n  ── pre-registered hypotheses " + "─" * 42)
    print(f"  H17  retained ≥ {H17_RETENTION_MIN} while settled   "
          f"r@500={r500:.3f} settled={settle_rate:.0%}      {'PASS' if h17 else 'FAIL'}")
    print(f"  H18  live cells > {H18_CELLS_MIN} at the end        "
          f"mean={mean_cells:.1f}{'':>17}{'PASS' if h18 else 'FAIL'}")
    rf = "n/a" if not np.isfinite(ratio) else f"{ratio:.2f}"
    wf = "n/a" if not np.isfinite(w) else f"{w:.4f}"
    print(f"  H19  between/within ≥ {H19_RATIO_MIN}             "
          f"within={wf} between={bt:.4f} ratio={rf}   {'PASS' if h19 else 'FAIL'}")
    print(f"  H20  Φ > 0 in ≥ half the runs          "
          f"{phi_alive:.0%} alive, mean Φ={mean_phi:.2f}{'':>6}{'PASS' if h20 else 'FAIL'}")

    ok = all([h17, h18, h19, h20])
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} ({sum([h17, h18, h19, h20])}/4)")

    if not h18:
        exploratory_why_no_mitosis(names)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
