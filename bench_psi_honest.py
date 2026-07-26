#!/usr/bin/env python3
"""bench_psi_honest.py — does Ψ→1/2 survive without the hardcoded pull term?

Pre-registered as docs/hypotheses/QD-1-psi-convergence-honesty.md. Phase 0 of
docs/qualia-decoder-spec.md. Thresholds below were fixed before the first run.

bench_consciousness_universe.simulate_meta_ca computes which CA rule
consciousness "selects", counts it, reports it — and then moves the state with
`dp = 0.001*(0.5 - p)`, an explicit attractor. The selected rule never touches
the state. So Ψ→1/2 is asserted by that line, not produced by the mechanism.

Shannon H(p) peaks at exactly p = 1/2, so a system selecting rules by argmax
H(p) (Law 71) should drift to 1/2 unaided. Three variants separate the claim
from its implementation:

  A pull           pull term present, rule discarded     — current code
  B nopull_norule  pull removed, rule still discarded    — negative control
  C nopull_rule    pull removed, selected rule applied   — Law 71 path

Usage:
    python3 bench_psi_honest.py                 # full run (5 seeds, 5000 steps)
    python3 bench_psi_honest.py --steps 500     # quick
"""

import argparse
import math
import numpy as np

from bench_consciousness_universe import ALL_DATA_TYPES
from qualia_sense import sense, hash_sense, sim_inputs, feature_distance

# ── pre-registered constants ───────────────────────────────
EPS = 0.05              # |p_T - 0.5| below this counts as converged
C_PASS_MIN = 0.80       # H1: variant C must reach this share
B_PASS_MAX = 0.20       # H1: negative control must stay below this
H2_RATIO_MIN = 2.0      # H2: between-stimulus / within-stimulus trajectory spread
H4_TOLERANCE = 0.10     # H4: content-vs-hash pass-rate gap allowed

RULE_DELTAS = 0.01 * (np.arange(8) - 3.5)   # δ per CA rule, as used in rule_h
VARIANTS = ("pull", "nopull_norule", "nopull_rule")


def stimulus_names():
    return [n for items in ALL_DATA_TYPES.values() for n in items]


def build_inits(names, source):
    """names → (init_p, init_g, bias_p, bias_g) arrays."""
    f = sense if source == "content" else hash_sense
    rows = [sim_inputs(f(n)) for n in names]
    return (np.array([r["init_p"] for r in rows]),
            np.array([r["init_g"] for r in rows]),
            np.array([r["bias_p"] for r in rows]),
            np.array([r["bias_g"] for r in rows]))


def binary_entropy(p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def simulate(inits, variant, seed, steps, record_every):
    """Vectorised port of simulate_meta_ca. Returns (p_final, g_final, traj).

    Identical to the original except where the variant demands otherwise:
    the rule-scoring, noise scales, weak per-stimulus bias and clipping are
    carried over unchanged, so the variants differ only in how dp is formed.
    """
    init_p, init_g, bias_p, bias_g = inits
    rng = np.random.default_rng(seed)
    n = len(init_p)

    p = 0.3 + 0.4 * init_p
    g = 0.3 + 0.4 * init_g
    ce_rule_weight = 1 + 0.1 * np.arange(8)

    traj = []
    for step in range(steps):
        # what each rule would do to H(p) — unchanged from the original
        rule_h = binary_entropy(p[:, None] + RULE_DELTAS[None, :])
        ce = (np.abs(p - 0.5)[:, None] * ce_rule_weight[None, :]
              + (np.abs(g - 0.5) * 0.5)[:, None]
              + rng.normal(0, 0.01, size=(n, 8)))
        best = np.argmax(0.7 * rule_h - 0.3 * ce, axis=1)

        noise_p = rng.normal(0, 0.002, n)
        noise_g = rng.normal(0, 0.002, n)

        if variant == "pull":
            dp = 0.001 * (0.5 - p) + noise_p
            dg = 0.001 * (0.5 - g) + noise_g
        elif variant == "nopull_norule":
            dp = noise_p
            dg = noise_g
        elif variant == "nopull_rule":
            dp = RULE_DELTAS[best] + noise_p
            dg = noise_g          # g has no rule mechanism — see report
        else:
            raise ValueError(variant)

        dp = dp + 0.0001 * (bias_p - 0.5)
        dg = dg + 0.0001 * (bias_g - 0.5)

        p = np.clip(p + dp, 0.001, 0.999)
        g = np.clip(g + dg, 0.001, 0.999)

        if step % record_every == 0:
            traj.append(p.copy())

    return p, g, np.array(traj).T          # traj: [n_stimuli, n_recorded]


def pass_rate(p_final):
    return float(np.mean(np.abs(p_final - 0.5) < EPS))


def traj_distance(a, b):
    """RMS distance between two recorded trajectories."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_variant(inits, variant, seeds, steps, record_every):
    finals, gs, trajs = [], [], []
    for s in seeds:
        p, g, t = simulate(inits, variant, s, steps, record_every)
        finals.append(p)
        gs.append(g)
        trajs.append(t)
    return np.array(finals), np.array(gs), np.array(trajs)   # [n_seeds, ...]


def test_h1(results):
    c_rate = float(np.mean([pass_rate(f) for f in results["nopull_rule"][0]]))
    b_rate = float(np.mean([pass_rate(f) for f in results["nopull_norule"][0]]))
    a_rate = float(np.mean([pass_rate(f) for f in results["pull"][0]]))
    ok = (c_rate >= C_PASS_MIN) and (b_rate < B_PASS_MAX)
    return ok, a_rate, b_rate, c_rate


def test_h2(trajs, rng_seed=0):
    """between-stimulus spread vs within-stimulus (across-seed) spread."""
    n_seeds, n_stim, _ = trajs.shape
    rng = np.random.default_rng(rng_seed)

    within = [traj_distance(trajs[a, i], trajs[b, i])
              for i in range(n_stim)
              for a in range(n_seeds) for b in range(a + 1, n_seeds)]

    pairs = rng.choice(n_stim, size=(min(2000, n_stim * 8), 2))
    between = [traj_distance(trajs[0, i], trajs[0, j]) for i, j in pairs if i != j]

    w, bt = float(np.mean(within)), float(np.mean(between))
    ratio = bt / w if w > 0 else float("inf")
    return ratio >= H2_RATIO_MIN, w, bt, ratio


def test_h3():
    """Stem groups: does the feature source keep related names close?"""
    groups = [
        ["서예", "서예체", "서예가"],
        ["만다라", "모래만다라"],
        ["빅뱅", "빅크런치"],
    ]
    out = {}
    for label, f in (("content", sense), ("hash", hash_sense)):
        within, between = [], []
        for gi, grp in enumerate(groups):
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    within.append(feature_distance(f(grp[i]), f(grp[j])))
                for gj, other in enumerate(groups):
                    if gj <= gi:
                        continue
                    for o in other:
                        between.append(feature_distance(f(grp[i]), f(o)))
        out[label] = (float(np.mean(within)), float(np.mean(between)))
    c_w, c_b = out["content"]
    ok = c_w < c_b
    return ok, out


def test_h4(content_rate, hash_rate):
    gap = abs(content_rate - hash_rate)
    return gap <= H4_TOLERANCE, gap


def exploratory_transient(inits, seeds, window=200):
    """POST-HOC, NOT PRE-REGISTERED. Why did H2 fail?

    H2 samples every 50th step over 5000. Under variant C the state reaches
    1/2 within a few steps, so all but the first recorded point sit on the
    same plateau and the stimulus signal is averaged away. This re-measures
    the same quantity inside the transient, at full resolution, and asks how
    long each stimulus takes to arrive.

    Findings here do not change the H2 verdict above. They scope what the
    trajectory-as-gate decoder can rely on.
    """
    trajs = np.array([simulate(inits, "nopull_rule", s, window, 1)[2] for s in seeds])
    _, w, bt, ratio = test_h2(trajs, rng_seed=1)

    arrival = []
    for si in range(trajs.shape[1]):
        t = trajs[0, si]
        hit = np.nonzero(np.abs(t - 0.5) < EPS)[0]
        arrival.append(int(hit[0]) if len(hit) else window)
    arrival = np.array(arrival)

    print("\n  ── exploratory (post-hoc · not pre-registered) " + "─" * 22)
    print(f"  transient window        first {window} steps, every step recorded")
    print(f"  between/within ratio    {ratio:.2f}   (within={w:.4f} between={bt:.4f})")
    print(f"  arrival step at |p-0.5|<{EPS}   "
          f"min={arrival.min()} median={int(np.median(arrival))} max={arrival.max()} "
          f"std={arrival.std():.1f}")
    print(f"  distinct arrival steps  {len(np.unique(arrival))} of {len(arrival)} stimuli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--record-every", type=int, default=50)
    args = ap.parse_args()

    names = stimulus_names()
    seeds = list(range(args.seeds))
    print(f"\n  QD-1 · {len(names)} stimuli × {args.seeds} seeds × {args.steps} steps")
    print(f"  converged := |p_T - 0.5| < {EPS}\n")

    inits = build_inits(names, "content")
    results = {v: run_variant(inits, v, seeds, args.steps, args.record_every)
               for v in VARIANTS}

    print("  ── convergence by variant (content features) " + "─" * 24)
    print(f"  {'variant':<16} {'pull':>6} {'rule→state':>11} {'|p-0.5|':>9} {'converged':>11}")
    labels = {"pull": "A pull", "nopull_norule": "B control", "nopull_rule": "C rule"}
    for v in VARIANTS:
        finals = results[v][0]
        dev = float(np.mean(np.abs(finals - 0.5)))
        rate = float(np.mean([pass_rate(f) for f in finals]))
        print(f"  {labels[v]:<16} {'yes' if v == 'pull' else 'no':>6} "
              f"{'yes' if v == 'nopull_rule' else 'no':>11} {dev:>9.4f} {rate:>10.1%}")

    g_dev = {v: float(np.mean(np.abs(results[v][1] - 0.5))) for v in VARIANTS}
    print(f"\n  gate g mean |g-0.5|:  " +
          "  ".join(f"{labels[v]}={g_dev[v]:.4f}" for v in VARIANTS))
    print("  (g has no rule mechanism in any variant — only the pull moved it)")

    h1_ok, a_rate, b_rate, c_rate = test_h1(results)
    h2_ok, w, bt, ratio = test_h2(results["nopull_rule"][2])
    h3_ok, h3 = test_h3()

    inits_hash = build_inits(names, "hash")
    hash_finals = run_variant(inits_hash, "nopull_rule", seeds,
                              args.steps, args.record_every)[0]
    hash_rate = float(np.mean([pass_rate(f) for f in hash_finals]))
    h4_ok, gap = test_h4(c_rate, hash_rate)

    print("\n  ── pre-registered hypotheses " + "─" * 40)
    print(f"  H1  C≥{C_PASS_MIN:.0%} and B<{B_PASS_MAX:.0%}      "
          f"A={a_rate:.1%}  B={b_rate:.1%}  C={c_rate:.1%}"
          f"          {'PASS' if h1_ok else 'FAIL'}")
    print(f"  H2  between/within ≥ {H2_RATIO_MIN}       "
          f"within={w:.4f} between={bt:.4f} ratio={ratio:.2f}"
          f"   {'PASS' if h2_ok else 'FAIL'}")
    print(f"  H3  related names stay closer     "
          f"content {h3['content'][0]:.3f}<{h3['content'][1]:.3f} · "
          f"hash {h3['hash'][0]:.3f}/{h3['hash'][1]:.3f}   {'PASS' if h3_ok else 'FAIL'}")
    print(f"  H4  |content - hash| ≤ {H4_TOLERANCE:.0%}        "
          f"content={c_rate:.1%} hash={hash_rate:.1%} gap={gap:.1%}"
          f"        {'PASS' if h4_ok else 'FAIL'}")

    verdict = all([h1_ok, h2_ok, h3_ok, h4_ok])
    print(f"\n  verdict: {'PASS' if verdict else 'FAIL'} "
          f"({sum([h1_ok, h2_ok, h3_ok, h4_ok])}/4)")

    if not h2_ok:
        exploratory_transient(inits, seeds)
    print()
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
