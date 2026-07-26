#!/usr/bin/env python3
"""bench_psi_field.py — same equilibrium, different pattern.

Pre-registered as docs/hypotheses/QD-2-stimulus-bearing-field.md.
Thresholds, the κ grid, d = 6 and the seed count were fixed before the first run.

QD-1 established that argmax-H rule selection produces the Ψ=1/2 attractor with
no hardcoded pull, and that a SCALAR state forgets its stimulus in 4 steps. For
a scalar the equilibrium is the pattern, so "same equilibrium, different
pattern" is not expressible.

Here the stimulus moves out of the objective and into the constraint of
Law 71 (`Ψ = argmax H(p) s.t. Φ > Φ_min`): a 6-dimensional state whose
dimensions are coupled by a stimulus-derived antisymmetric matrix.

  entropy term  → every marginal to 1/2       (same equilibrium)
  coupling term → stimulus-specific spacing   (different pattern)

κ = 0 is the negative control — QD-1 with six identical copies.

The `|g - 0.5|` gate term of QD-1's cross-entropy is dropped: QD-1 found g has
no selection mechanism and only the deleted pull ever moved it. There is no
separate gate here.

Usage:
    python3 bench_psi_field.py                # full run
    python3 bench_psi_field.py --steps 500    # quick
"""

import argparse
import numpy as np

from bench_consciousness_universe import ALL_DATA_TYPES
from qualia_sense import sense

# ── pre-registered constants ───────────────────────────────
D = 6                       # σ(6) architecture — a repo convention, not a measurement
EPS = 0.05
KAPPA_GRID = (0.0, 0.005, 0.01, 0.02)
KAPPA_PRIMARY = 0.01
H5_MARGINAL_MIN = 0.80
H6_RATIO_MIN = 2.0          # at κ = primary
H6_CONTROL_MAX = 1.5        # at κ = 0
H7_SD_MAX = 0.02
H7_CENTRE_TOL = 0.05
H8_RATIO_MIN = 2.0

RULE_DELTAS = 0.01 * (np.arange(8) - 3.5)

# The six measured features with the highest spread over the stimulus set,
# in rank order (see qualia_sense.sim_inputs for the measured values).
FIELD_FEATURES = ("final_ratio", "codepoint_spread", "script_mix",
                  "char_variety", "vowel_position", "jamo_density")


def stimulus_names():
    return [n for items in ALL_DATA_TYPES.values() for n in items]


def stimulus_weights(names):
    """names → w [n_stimuli, D], the per-dimension stimulus signature."""
    return np.array([[getattr(sense(n), f) for f in FIELD_FEATURES] for n in names])


def binary_entropy(p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def simulate_field(w, kappa, seed, steps, record_every):
    """Returns (p_final [n, D], traj [n, D, T_rec])."""
    rng = np.random.default_rng(seed)
    n = w.shape[0]

    p = 0.3 + 0.4 * w                       # [n, D]
    coupling = w[:, :, None] - w[:, None, :]  # [n, D, D], antisymmetric
    s_row = coupling.sum(axis=2)              # [n, D]  Σ_j coupling[i,j]
    ce_rule_weight = 1 + 0.1 * np.arange(8)

    traj = []
    for step in range(steps):
        cand = p[:, :, None] + RULE_DELTAS[None, None, :]        # [n, D, 8]

        rule_h = binary_entropy(cand)
        ce = (np.abs(p - 0.5)[:, :, None] * ce_rule_weight[None, None, :]
              + rng.normal(0, 0.01, size=(n, D, 8)))

        # Σ_j coupling[i,j]·(p_i + δ − p_j) = (p_i + δ)·Σ_j c[i,j] − Σ_j c[i,j]·p_j
        t_row = np.einsum("nij,nj->ni", coupling, p)             # [n, D]
        interaction = cand * s_row[:, :, None] - t_row[:, :, None]

        best = np.argmax(0.7 * rule_h - 0.3 * ce - kappa * interaction, axis=2)

        p = np.clip(p + RULE_DELTAS[best] + rng.normal(0, 0.002, (n, D)),
                    0.001, 0.999)

        if step % record_every == 0:
            traj.append(p.copy())

    return p, np.transpose(np.array(traj), (1, 2, 0))            # [n, D, T_rec]


def traj_distance(a, b):
    """RMS distance between two [D, T] trajectories."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def between_within(trajs, rng_seed=0, n_pairs=2000):
    """trajs [n_seeds, n_stim, ...] → (within, between, ratio)."""
    n_seeds, n_stim = trajs.shape[0], trajs.shape[1]
    rng = np.random.default_rng(rng_seed)

    within = [traj_distance(trajs[a, i], trajs[b, i])
              for i in range(n_stim)
              for a in range(n_seeds) for b in range(a + 1, n_seeds)]

    pairs = rng.choice(n_stim, size=(min(n_pairs, n_stim * 8), 2))
    between = [traj_distance(trajs[0, i], trajs[0, j]) for i, j in pairs if i != j]

    w, bt = float(np.mean(within)), float(np.mean(between))
    return w, bt, (bt / w if w > 0 else float("inf"))


def correlation_structures(trajs):
    """trajs [n_seeds, n_stim, D, T] → [n_seeds, n_stim, D, D] correlation matrices."""
    n_seeds, n_stim = trajs.shape[0], trajs.shape[1]
    out = np.zeros((n_seeds, n_stim, D, D))
    for s in range(n_seeds):
        for i in range(n_stim):
            x = trajs[s, i]
            sd = x.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-12, 1e-12, sd)
            z = (x - x.mean(axis=1, keepdims=True)) / sd
            out[s, i] = (z @ z.T) / x.shape[1]
    return out


def run(w, kappa, seeds, steps, record_every):
    finals, trajs = [], []
    for s in seeds:
        p, t = simulate_field(w, kappa, s, steps, record_every)
        finals.append(p)
        trajs.append(t)
    return np.array(finals), np.array(trajs)


def exploratory_memory_window(w, seeds, starts=(0, 50, 500, 2000), length=200):
    """POST-HOC, NOT PRE-REGISTERED. Where does the stimulus signature live?

    H8 measures the correlation structure in a window that starts at t=0, so a
    pass could mean "the state carries the stimulus" or merely "the arrival is
    stimulus-specific". Sliding the same window later separates the two: a
    ratio near 1.0 means the states of two different stimuli are no longer
    distinguishable at all.

    Does not change the verdict above. Scopes what a decoder could read.
    """
    print("\n  ── exploratory (post-hoc · not pre-registered) " + "─" * 22)
    print("  correlation ratio by window position (1.0 = stimuli indistinguishable)")
    print(f"  {'window start':>14} {'κ=0 control':>14} {'κ=0.01':>10}")
    for start in starts:
        row = []
        for kappa in (0.0, KAPPA_PRIMARY):
            trajs = []
            for s in seeds:
                _, t = simulate_field(w, kappa, s, start + length, 1)
                trajs.append(t[:, :, start:])
            corr = correlation_structures(np.array(trajs))
            row.append(between_within(corr, rng_seed=2)[2])
        print(f"  {('step ' + str(start)):>14} {row[0]:>14.2f} {row[1]:>10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--record-every", type=int, default=50)
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    names = stimulus_names()
    seeds = list(range(args.seeds))
    w = stimulus_weights(names)

    print(f"\n  QD-2 · {len(names)} stimuli × {args.seeds} seeds × "
          f"{args.steps} steps · D={D}")
    print(f"  converged := |p_i,T - 0.5| < {EPS}\n")

    print("  ── κ sweep " + "─" * 58)
    print(f"  {'κ':>7} {'marginals conv':>16} {'Ψ=mean(p)':>11} {'Ψ sd':>8} "
          f"{'traj ratio':>11} {'corr ratio':>11}")

    table = {}
    for k in KAPPA_GRID:
        finals, trajs = run(w, k, seeds, args.steps, args.record_every)
        marg = float(np.mean(np.abs(finals - 0.5) < EPS))
        psi = finals.mean(axis=2)                       # [n_seeds, n_stim]
        psi_mean, psi_sd = float(psi.mean()), float(psi.mean(axis=0).std())
        _, _, ratio = between_within(trajs)

        _, wtrajs = run(w, k, seeds, args.window, 1)
        corr = correlation_structures(wtrajs)
        _, _, corr_ratio = between_within(corr, rng_seed=2)

        table[k] = dict(marg=marg, psi_mean=psi_mean, psi_sd=psi_sd,
                        ratio=ratio, corr_ratio=corr_ratio)
        mark = "  ← control" if k == 0 else ("  ← primary" if k == KAPPA_PRIMARY else "")
        print(f"  {k:>7} {marg:>15.1%} {psi_mean:>11.4f} {psi_sd:>8.4f} "
              f"{ratio:>11.2f} {corr_ratio:>11.2f}{mark}")

    pri, ctl = table[KAPPA_PRIMARY], table[0.0]
    h5 = pri["marg"] >= H5_MARGINAL_MIN
    h6 = (pri["ratio"] >= H6_RATIO_MIN) and (ctl["ratio"] < H6_CONTROL_MAX)
    h7 = (pri["psi_sd"] < H7_SD_MAX) and (abs(pri["psi_mean"] - 0.5) < H7_CENTRE_TOL)
    h8 = pri["corr_ratio"] >= H8_RATIO_MIN

    print("\n  ── pre-registered hypotheses (κ = " f"{KAPPA_PRIMARY}) " + "─" * 30)
    print(f"  H5  marginals ≥ {H5_MARGINAL_MIN:.0%}              "
          f"{pri['marg']:.1%}{'':>22}{'PASS' if h5 else 'FAIL'}")
    print(f"  H6  ratio ≥ {H6_RATIO_MIN} and control < {H6_CONTROL_MAX}   "
          f"primary={pri['ratio']:.2f} control={ctl['ratio']:.2f}"
          f"{'':>7}{'PASS' if h6 else 'FAIL'}")
    print(f"  H7  Ψ sd < {H7_SD_MAX} and |Ψ-0.5| < {H7_CENTRE_TOL}  "
          f"sd={pri['psi_sd']:.4f} Ψ={pri['psi_mean']:.4f}"
          f"{'':>7}{'PASS' if h7 else 'FAIL'}")
    print(f"  H8  correlation ratio ≥ {H8_RATIO_MIN}          "
          f"{pri['corr_ratio']:.2f}{'':>22}{'PASS' if h8 else 'FAIL'}")

    ok = all([h5, h6, h7, h8])
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} ({sum([h5, h6, h7, h8])}/4)")

    if not h6:
        exploratory_memory_window(w, seeds)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
