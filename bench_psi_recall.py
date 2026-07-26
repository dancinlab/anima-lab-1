#!/usr/bin/env python3
"""bench_psi_recall.py — the leak and the key.

Pre-registered as docs/hypotheses/QD-4-tau-and-cue.md. Grids, thresholds and
the primary-selection rule were fixed before the first run.

QD-3 found both memory arms failing for different, measured reasons:
  · the habituation trace holds the stimulus at 0.907 (step 15) and 0.731
    (step 50), then LEAKS to 0.124 by step 500 — where the window is;
  · the episodic store holds 0.991 at step 500 while the query (the state)
    holds 0.039, and recall picks the informative anchor 0.0% of the time
    against 4.8% chance — the store is unaddressable, not empty.

  T  tau-sweep  habituation, τ ∈ {200, 1000, 5000, 20000}
  K  m-cue      key = m at write, value = p at write, query = current m
  C  control    τ=200, keyed and queried by p — QD-3 exactly, must fail

Cues are centred across dimensions before matching: m carries a common drift of
roughly τ × 0.015 in every dimension alike, and cosine would match on that
rather than on the stimulus. Same centring for keys and queries.

Usage:
    python3 bench_psi_recall.py                # full run
    python3 bench_psi_recall.py --steps 800    # quick
"""

import argparse
import numpy as np

from bench_consciousness_universe import ALL_DATA_TYPES
from bench_psi_field import D, RULE_DELTAS, binary_entropy, stimulus_weights
from bench_psi_memory import (CAPACITY, EPS, H9_WINDOW, H12_WINDOW, STORE_EVERY,
                              WINDOW_LEN, fmt, window_ratio)

# ── pre-registered constants ───────────────────────────────
TAU_GRID = (200, 1000, 5000, 20000)
STRENGTH_GRID = (0.01, 0.05, 0.2, 1.0)
K_TAU = 5000                    # arm K: the cue must survive to be a cue
H13_RATIO_MIN = 2.0
H13_CONTROL_MAX = 1.2
CONV_GATE = 0.80                # convergence is a gate, not a tiebreak
H15_RATIO_MIN = 2.0
H16_SD_MAX = 0.02
H16_CENTRE_TOL = 0.05
AVG_TAIL = 1000


def centre(x):
    """Remove the common across-dimension offset — see the module docstring."""
    return x - x.mean(axis=1, keepdims=True)


def simulate(w, arm, tau, strength, seed, steps):
    rng = np.random.default_rng(seed)
    n = w.shape[0]

    p = 0.3 + 0.4 * w
    m = np.zeros((n, D))
    keys = np.zeros((n, CAPACITY, D))
    vals = np.zeros((n, CAPACITY, D))
    ptr, stored = 0, 0

    ce_rule_weight = 1 + 0.1 * np.arange(8)
    tail_sum = np.zeros((n, D))
    tail_n = 0
    win9, win12 = [], []

    for step in range(steps):
        cand = p[:, :, None] + RULE_DELTAS[None, None, :]
        score = (0.7 * binary_entropy(cand)
                 - 0.3 * (np.abs(p - 0.5)[:, :, None] * ce_rule_weight[None, None, :]
                          + rng.normal(0, 0.01, size=(n, D, 8))))

        if arm == "tau":
            score = score - strength * m[:, :, None] * (cand - 0.5)

        elif arm in ("mcue", "control") and stored > 0:
            query = centre(m) if arm == "mcue" else (p - 0.5)
            kk = centre(keys[:, :stored]) if arm == "mcue" else (keys[:, :stored] - 0.5)
            num = np.einsum("nkd,nd->nk", kk, query)
            den = np.linalg.norm(kk, axis=2) * np.linalg.norm(query, axis=1)[:, None]
            idx = np.argmax(num / np.where(den < 1e-12, 1e-12, den), axis=1)
            recalled = vals[np.arange(n), idx]
            score = score - strength * np.abs(cand - recalled[:, :, None])

        best = np.argmax(score, axis=2)
        m = (1 - 1.0 / tau) * m + (p - 0.5)
        p = np.clip(p + RULE_DELTAS[best] + rng.normal(0, 0.002, (n, D)), 0.001, 0.999)

        if arm in ("mcue", "control") and step % STORE_EVERY == 0:
            keys[:, ptr] = m if arm == "mcue" else p
            vals[:, ptr] = p
            ptr = (ptr + 1) % CAPACITY
            stored = min(stored + 1, CAPACITY)

        if H9_WINDOW <= step < H9_WINDOW + WINDOW_LEN:
            win9.append(p.copy())
        if H12_WINDOW <= step < H12_WINDOW + WINDOW_LEN:
            win12.append(p.copy())
        if step >= steps - AVG_TAIL:
            tail_sum += p
            tail_n += 1

    def stack(win):
        return np.transpose(np.array(win), (1, 2, 0)) if win else None

    return tail_sum / max(tail_n, 1), stack(win9), stack(win12)


def evaluate(w, arm, tau, strength, seeds, steps):
    tails, w9, w12 = [], [], []
    for s in seeds:
        t, a, b = simulate(w, arm, tau, strength, s, steps)
        tails.append(t)
        w9.append(a)
        w12.append(b)
    tails = np.array(tails)
    psi = tails.mean(axis=2)
    return dict(conv=float(np.mean(np.abs(tails - 0.5) < EPS)),
                psi_mean=float(psi.mean()), psi_sd=float(psi.mean(axis=0).std()),
                r9=window_ratio(w9, 2), r12=window_ratio(w12, 3))


def exploratory_tradeoff(w, seeds, steps=H9_WINDOW + WINDOW_LEN):
    """POST-HOC, NOT PRE-REGISTERED. All four attempts failed — is there a wall?

    The pre-registered ratio reads the SHAPE of the window's time series. This
    reads what is actually being asked: how much of the stimulus signature the
    state still carries, as the correlation between the window-mean state and
    the stimulus weights — alongside how much of the state converged.

    Does not change the verdict above.
    """
    def corr(x):
        cs = [abs(np.corrcoef(x[:, d], w[:, d])[0, 1]) for d in range(D)
              if x[:, d].std() > 1e-12 and w[:, d].std() > 1e-12]
        return float(np.mean(cs)) if cs else float("nan")

    probes = (("control p-cue", "control", 200, 0.05),
              ("control p-cue", "control", 200, 0.2),
              ("T habituation", "tau", 1000, 0.2),
              ("T habituation", "tau", 20000, 1.0),
              ("K m-cue", "mcue", K_TAU, 0.05),
              ("K m-cue", "mcue", K_TAU, 0.2))

    print("\n  ── exploratory (post-hoc · not pre-registered) " + "─" * 22)
    print("  stimulus retained vs equilibrium reached, in the same window")
    print(f"  {'config':<18} {'τ':>7} {'μ/ν':>6} {'converged':>10} {'stimulus':>9}")
    for label, arm, tau, g in probes:
        cs, cv = [], []
        for s in seeds:
            tail, w9, _ = simulate(w, arm, tau, g, s, steps)
            cs.append(corr(w9.mean(axis=2)))
            cv.append(np.mean(np.abs(tail - 0.5) < EPS))
        print(f"  {label:<18} {tau:>7} {g:>6} {np.mean(cv):>9.1%} {np.mean(cs):>9.3f}")
    print("  Read the two right-hand columns together: nothing here holds the")
    print("  stimulus while also converging. That trade is the result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    names = [n for items in ALL_DATA_TYPES.values() for n in items]
    seeds = list(range(args.seeds))
    w = stimulus_weights(names)

    print(f"\n  QD-4 · {len(names)} stimuli × {args.seeds} seeds × {args.steps} steps · D={D}")
    print(f"  ratio 1.0 = stimuli indistinguishable · gate: convergence ≥ {CONV_GATE:.0%}\n")
    print(f"  {'arm':<10} {'τ':>7} {'μ/ν':>6} {'converged':>10} {'Ψ':>8} {'Ψ sd':>8} "
          f"{'r@500':>7} {'r@2000':>7}")

    rows = []
    for arm, taus in (("control", (200,)), ("tau", TAU_GRID), ("mcue", (K_TAU,))):
        for tau in taus:
            for g in STRENGTH_GRID:
                r = evaluate(w, arm, tau, g, seeds, args.steps)
                r.update(arm=arm, tau=tau, strength=g)
                rows.append(r)
                gate = "" if r["conv"] >= CONV_GATE else "  ✗gate"
                print(f"  {arm:<10} {tau:>7} {g:>6} {r['conv']:>9.1%} "
                      f"{r['psi_mean']:>8.4f} {r['psi_sd']:>8.4f} "
                      f"{fmt(r['r9']):>7} {fmt(r['r12']):>7}{gate}")
                if arm == "control":
                    break          # control is a single config: τ=200, p-cue

    def primary(arm):
        ok = [r for r in rows if r["arm"] == arm and r["conv"] >= CONV_GATE
              and np.isfinite(r["r9"])]
        return max(ok, key=lambda r: r["r9"]) if ok else None

    ctl = next(r for r in rows if r["arm"] == "control")
    print("\n  ── pre-registered hypotheses " + "─" * 42)
    passing = []
    for arm, label in (("tau", "T τ-sweep"), ("mcue", "K m-cue")):
        pr = primary(arm)
        if pr is None:
            print(f"  {label:<12} no configuration passes the convergence gate")
            continue
        ok = pr["r9"] >= H13_RATIO_MIN
        passing.append((arm, pr, ok))
        print(f"  {label:<12} best qualifying: τ={pr['tau']} μ/ν={pr['strength']} "
              f"→ r@500={fmt(pr['r9'])} {'PASS' if ok else 'FAIL'} · "
              f"r@2000={fmt(pr['r12'])} · Ψ={pr['psi_mean']:.4f} sd={pr['psi_sd']:.4f}")

    h13 = any(ok for _, _, ok in passing) and ctl["r9"] < H13_CONTROL_MAX
    h15 = any(np.isfinite(pr["r12"]) and pr["r12"] >= H15_RATIO_MIN
              for _, pr, ok in passing if ok)
    h16 = all(pr["psi_sd"] < H16_SD_MAX and abs(pr["psi_mean"] - 0.5) < H16_CENTRE_TOL
              for _, pr, _ in passing)

    which = ", ".join(f"{a}={'PASS' if ok else 'FAIL'}" for a, _, ok in passing) or "none"
    print(f"\n  H13  any qualifying r@500 ≥ {H13_RATIO_MIN}, control < {H13_CONTROL_MAX}"
          f"    control={fmt(ctl['r9'])}   {'PASS' if h13 else 'FAIL'}")
    print(f"  H14  which fix                                {which}")
    print(f"  H15  a passing config still ≥ {H15_RATIO_MIN} at step {H12_WINDOW}"
          f"{'':>12}{'PASS' if h15 else 'FAIL'}")
    print(f"  H16  Ψ sd < {H16_SD_MAX}, |Ψ-0.5| < {H16_CENTRE_TOL} at every primary"
          f"{'':>10}{'PASS' if h16 else 'FAIL'}")

    ok = all([h13, h15, h16])
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} ({sum([h13, h15, h16])}/3)")

    if not h13:
        exploratory_tradeoff(w, seeds)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
