#!/usr/bin/env python3
"""bench_psi_memory.py — three memory arms against the current form.

Pre-registered as docs/hypotheses/QD-3-habituation-memory.md. Grids, thresholds,
the primary-selection rule and the window positions were fixed before the first
run.

QD-1: the Ψ=1/2 attractor is real (93.4% vs 9.4% control). QD-2: the stimulus
signature dies by step 50 — correlation ratio 2.36 at step 0, 1.01 at step 50,
0.97 at step 500, where 1.0 means two stimuli are indistinguishable. Neither
six dimensions nor coupling extended it. The update reads only the current
state, so the process is memoryless.

  N  none          QD-1/QD-2 dynamics unchanged — the current form, and the control
  H  habituation   m_{t+1} = (1-1/τ)·m_t + (p_t - 1/2), acting on rule selection
  E  episodic      bounded store + nearest-anchor recall biasing rule selection

Arm E is the shape of anima's `.kosmos` anchor and trinity.py's VectorMemory.
It does NOT model the sleep-consolidation step, where anima blends co-replayed
PAIRS into a new lane="dream" node (core/dream_persist.py). Storage and recall
only; the pair-blending is a later card.

Usage:
    python3 bench_psi_memory.py                # full run
    python3 bench_psi_memory.py --steps 800    # quick
"""

import argparse
import numpy as np

from bench_consciousness_universe import ALL_DATA_TYPES
from bench_psi_field import (D, RULE_DELTAS, binary_entropy, between_within,
                             correlation_structures, stimulus_weights)

# ── pre-registered constants ───────────────────────────────
EPS = 0.05
TAU = 200                       # habituation leak
STORE_EVERY = 25                # episodic write interval
CAPACITY = 64                   # episodic ring-buffer size
STRENGTH_GRID = (0.01, 0.05, 0.2, 1.0)
H9_WINDOW = 500                 # signature must outlive the transient
H12_WINDOW = 2000               # ... and still be there much later
WINDOW_LEN = 200
H9_RATIO_MIN = 2.0
H9_CONTROL_MAX = 1.2
H10_MARGINAL_MIN = 0.80
H11_SD_MAX = 0.02
H11_CENTRE_TOL = 0.05
H12_RATIO_MIN = 2.0
AVG_TAIL = 1000                 # H10/H11 average over the last this-many steps


def simulate_memory(w, arm, strength, seed, steps):
    """Returns (tail_mean [n, D], win9 [n, D, T], win12 [n, D, T]).

    Arms differ only in the term added to the rule score; everything else is
    the QD-1 dynamics with D=6 and no coupling.
    """
    rng = np.random.default_rng(seed)
    n = w.shape[0]

    p = 0.3 + 0.4 * w
    m = np.zeros((n, D))
    store = np.zeros((n, CAPACITY, D))
    stored = 0
    ptr = 0

    ce_rule_weight = 1 + 0.1 * np.arange(8)
    tail_sum = np.zeros((n, D))
    tail_n = 0
    win9, win12 = [], []

    for step in range(steps):
        cand = p[:, :, None] + RULE_DELTAS[None, None, :]         # [n, D, 8]
        score = (0.7 * binary_entropy(cand)
                 - 0.3 * (np.abs(p - 0.5)[:, :, None] * ce_rule_weight[None, None, :]
                          + rng.normal(0, 0.01, size=(n, D, 8))))

        if arm == "habituation":
            score = score - strength * m[:, :, None] * (cand - 0.5)

        elif arm == "episodic" and stored > 0:
            # Cosine on the DEVIATION, not the raw state: every state sits near
            # 0.5 in every dimension, so raw cosine is ~1 for all slots and the
            # retrieval is arbitrary. The deviation is where the signal is.
            pc = p - 0.5                                          # [n, D]
            sc = store[:, :stored] - 0.5                          # [n, stored, D]
            num = np.einsum("nkd,nd->nk", sc, pc)
            den = (np.linalg.norm(sc, axis=2) * np.linalg.norm(pc, axis=1)[:, None])
            sim = num / np.where(den < 1e-12, 1e-12, den)
            nearest = store[np.arange(n), np.argmax(sim, axis=1)]  # [n, D]
            score = score - strength * np.abs(cand - nearest[:, :, None])

        best = np.argmax(score, axis=2)
        m = (1 - 1.0 / TAU) * m + (p - 0.5)                       # record the path
        p = np.clip(p + RULE_DELTAS[best] + rng.normal(0, 0.002, (n, D)), 0.001, 0.999)

        if arm == "episodic" and step % STORE_EVERY == 0:
            store[:, ptr] = p
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
        # An empty window means the run was shorter than the window position.
        # Return None rather than zeros: a constant window has zero within-
        # variance, and the ratio would come out inf and be scored as a pass.
        return np.transpose(np.array(win), (1, 2, 0)) if win else None

    return tail_sum / max(tail_n, 1), stack(win9), stack(win12)


def window_ratio(windows, rng_seed):
    """between/within ratio, or NaN when the window is missing or degenerate."""
    if any(x is None for x in windows):
        return float("nan")
    within, _, ratio = between_within(correlation_structures(np.array(windows)),
                                      rng_seed=rng_seed)
    if within < 1e-9 or not np.isfinite(ratio):
        return float("nan")
    return ratio


def evaluate(w, arm, strength, seeds, steps):
    tails, w9, w12 = [], [], []
    for s in seeds:
        t, a, b = simulate_memory(w, arm, strength, s, steps)
        tails.append(t)
        w9.append(a)
        w12.append(b)
    tails = np.array(tails)

    marginal = float(np.mean(np.abs(tails - 0.5) < EPS))
    psi = tails.mean(axis=2)

    return dict(marginal=marginal,
                psi_mean=float(psi.mean()), psi_sd=float(psi.mean(axis=0).std()),
                ratio9=window_ratio(w9, 2), ratio12=window_ratio(w12, 3))


def exploratory_where_it_dies(w, seed=0, probe=500):
    """POST-HOC, NOT PRE-REGISTERED. Both memory arms failed — where exactly?

    Tracks the correlation between the stimulus signature and (a) the state,
    (b) the habituation integral, and at `probe` compares what the episodic
    store HOLDS against what recall actually RETURNS.

    Does not change the verdict above.
    """
    n = w.shape[0]
    rng = np.random.default_rng(seed)
    p = 0.3 + 0.4 * w
    m = np.zeros((n, D))
    store = np.zeros((n, CAPACITY, D))
    ptr, stored = 0, 0
    ce_rule_weight = 1 + 0.1 * np.arange(8)

    def corr(x):
        cs = [abs(np.corrcoef(x[:, d], w[:, d])[0, 1]) for d in range(D)
              if x[:, d].std() > 1e-12 and w[:, d].std() > 1e-12]
        return float(np.mean(cs)) if cs else float("nan")

    print("\n  ── exploratory (post-hoc · not pre-registered) " + "─" * 22)
    print("  correlation with the stimulus signature (1.0 = intact · 0.0 = gone)")
    print(f"  {'step':>6} {'state p':>10} {'habit m':>10}")

    marks = (1, 5, 15, 50, 500, 2000)
    for step in range(max(marks) + 1):
        cand = p[:, :, None] + RULE_DELTAS[None, None, :]
        score = (0.7 * binary_entropy(cand)
                 - 0.3 * (np.abs(p - 0.5)[:, :, None] * ce_rule_weight[None, None, :]
                          + rng.normal(0, 0.01, size=(n, D, 8))))
        m = (1 - 1.0 / TAU) * m + (p - 0.5)
        p = np.clip(p + RULE_DELTAS[np.argmax(score, axis=2)]
                    + rng.normal(0, 0.002, (n, D)), 0.001, 0.999)
        if step % STORE_EVERY == 0:
            store[:, ptr] = p
            ptr = (ptr + 1) % CAPACITY
            stored = min(stored + 1, CAPACITY)

        if step in marks:
            print(f"  {step:>6} {corr(p):>10.3f} {corr(m):>10.3f}")

        if step == probe:
            pc, sc = p - 0.5, store[:, :stored] - 0.5
            num = np.einsum("nkd,nd->nk", sc, pc)
            den = np.linalg.norm(sc, axis=2) * np.linalg.norm(pc, axis=1)[:, None]
            idx = np.argmax(num / np.where(den < 1e-12, 1e-12, den), axis=1)
            best = max(corr(store[:, k]) for k in range(stored))
            print(f"\n  at step {probe} — does recall reach what the store holds?")
            print(f"    best anchor in the store   {best:.3f}")
            print(f"    the query (current state)  {corr(p):.3f}")
            print(f"    the anchor recall returned {corr(store[np.arange(n), idx]):.3f}")
            print(f"    picked the best anchor     {float(np.mean(idx == 0)):.1%} "
                  f"of the time (chance {1 / stored:.1%})")


def fmt(x):
    """NaN renders as n/a — a missing measurement is never shown as a number."""
    return "n/a" if not np.isfinite(x) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    names = [n for items in ALL_DATA_TYPES.values() for n in items]
    seeds = list(range(args.seeds))
    w = stimulus_weights(names)

    print(f"\n  QD-3 · {len(names)} stimuli × {args.seeds} seeds × {args.steps} steps · D={D}")
    print(f"  ratio 1.0 = stimuli indistinguishable · converged := |p̄ - 0.5| < {EPS}")
    for label, pos in (("H9", H9_WINDOW), ("H12", H12_WINDOW)):
        if args.steps < pos + WINDOW_LEN:
            print(f"  ⚠️  --steps {args.steps} is short of the {label} window "
                  f"({pos}..{pos + WINDOW_LEN}) — {label} reports n/a, not a pass")
    print()
    print(f"  {'arm':<14} {'strength':>9} {'marginals':>10} {'Ψ':>8} {'Ψ sd':>8} "
          f"{'r@{}'.format(H9_WINDOW):>8} {'r@{}'.format(H12_WINDOW):>8}")

    results = {"none": {}, "habituation": {}, "episodic": {}}
    for arm, grid in (("none", (0.0,)),
                      ("habituation", STRENGTH_GRID),
                      ("episodic", STRENGTH_GRID)):
        for g in grid:
            r = evaluate(w, arm, g, seeds, args.steps)
            results[arm][g] = r
            tag = "  ← control" if arm == "none" else ""
            print(f"  {arm:<14} {g:>9} {r['marginal']:>9.1%} {r['psi_mean']:>8.4f} "
                  f"{r['psi_sd']:>8.4f} {fmt(r['ratio9']):>8} {fmt(r['ratio12']):>8}{tag}")

    # primary = smallest strength whose marginal convergence still passes H10
    primary = {}
    for arm in ("habituation", "episodic"):
        ok = [g for g in STRENGTH_GRID if results[arm][g]["marginal"] >= H10_MARGINAL_MIN]
        primary[arm] = min(ok) if ok else None

    control = results["none"][0.0]
    print("\n  ── pre-registered hypotheses " + "─" * 42)
    print(f"  primary strength (smallest passing H10):  "
          + " · ".join(f"{a}={primary[a]}" for a in primary))

    h9_arms, h12_arms = [], []
    for arm in ("habituation", "episodic"):
        g = primary[arm]
        if g is None:
            print(f"  {arm:<14} primary undefined — H9/H12 not evaluated (pre-registered)")
            continue
        r = results[arm][g]
        a9 = r["ratio9"] >= H9_RATIO_MIN
        a12 = r["ratio12"] >= H12_RATIO_MIN
        h9_arms.append(a9)
        h12_arms.append(a12)
        print(f"  {arm:<14} μ/ν={g:<5} r@{H9_WINDOW}={fmt(r['ratio9'])} "
              f"{'PASS' if a9 else 'FAIL'}  ·  r@{H12_WINDOW}={fmt(r['ratio12'])} "
              f"{'PASS' if a12 else 'FAIL'}  ·  Ψ={r['psi_mean']:.4f} sd={r['psi_sd']:.4f}")

    h9 = any(h9_arms) and control["ratio9"] < H9_CONTROL_MAX
    h10 = all(results[a][primary[a]]["marginal"] >= H10_MARGINAL_MIN
              for a in primary if primary[a] is not None)
    h11 = all(results[a][primary[a]]["psi_sd"] < H11_SD_MAX
              and abs(results[a][primary[a]]["psi_mean"] - 0.5) < H11_CENTRE_TOL
              for a in primary if primary[a] is not None)
    h12 = any(h12_arms)

    print(f"\n  H9   any arm r@{H9_WINDOW} ≥ {H9_RATIO_MIN}, control < {H9_CONTROL_MAX}"
          f"     control={fmt(control['ratio9'])}     {'PASS' if h9 else 'FAIL'}")
    print(f"  H10  marginals ≥ {H10_MARGINAL_MIN:.0%} at every primary"
          f"{'':>21}{'PASS' if h10 else 'FAIL'}")
    print(f"  H11  Ψ sd < {H11_SD_MAX}, |Ψ-0.5| < {H11_CENTRE_TOL} at every primary"
          f"{'':>10}{'PASS' if h11 else 'FAIL'}")
    print(f"  H12  a passing arm still ≥ {H12_RATIO_MIN} at step {H12_WINDOW}"
          f"{'':>13}{'PASS' if h12 else 'FAIL'}")

    ok = all([h9, h10, h11, h12])
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} ({sum([h9, h10, h11, h12])}/4)")

    if not h9:
        exploratory_where_it_dies(w)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
