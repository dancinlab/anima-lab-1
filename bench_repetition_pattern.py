#!/usr/bin/env python3
"""bench_repetition_pattern.py — what does repeating a word do to the pattern?

Feeding `서예` and `서예서예서예` asks something the 170-name stimulus set
could not: is repetition a different experience, a stronger one, or the same
one? It also exercises the one measurement that was degenerate over that set —
`bigram_repeat` had spread 0.000 across all 170 names, because no single name
repeats a character pair.

Three readings, side by side:

  1. the sha256 path the bench shipped with — a hash, so `서예` and `서예서예`
     are as unrelated as `서예` and `빅뱅`;
  2. `qualia_sense` — measured from the string, so repetition should move the
     repetition-sensitive features and leave the phonological ones alone;
  3. the engine's own state after settling, which is what a decoder would read.

Reading 3 requires torch — run under the repo .venv:
    .venv/bin/python bench_repetition_pattern.py
Without torch the first two readings still run.
"""

import argparse
import math
import numpy as np

from qualia_sense import sense, hash_sense, feature_distance, FEATURE_NAMES

BLOCKS = "▁▂▃▄▅▆▇█"
STEMS = ("서예", "만다라", "검은사각형", "빅뱅")
MAX_REPEAT = 5
ENGINE_STEPS = 300


def bar(vec):
    """8 features → the ▁▂▃▄▅▆▇█ fingerprint the universe bench renders."""
    return "".join(BLOCKS[min(7, int(v * 8))] for v in vec)


def repeats(stem, n):
    return stem * n


def show_patterns(stem):
    print(f"\n  ── {stem} ── " + "─" * 52)
    print(f"  {'input':<20} {'content 무늬':<10}  {'hash 무늬':<10}  "
          f"{'content Δ':>10} {'hash Δ':>8}")
    base_c, base_h = sense(stem), hash_sense(stem)
    for n in range(1, MAX_REPEAT + 1):
        s = repeats(stem, n)
        c, h = sense(s), hash_sense(s)
        label = f"{stem}×{n}"
        print(f"  {label:<20} {bar(c.vector()):<10}  {bar(h.vector()):<10}  "
              f"{feature_distance(base_c, c):>10.3f} {feature_distance(base_h, h):>8.3f}")


def show_features(stem):
    print(f"\n  ── which measurements move · {stem} " + "─" * 34)
    head = "  ".join(f"{n[:8]:>8}" for n in FEATURE_NAMES)
    print(f"  {'input':<16} {head}")
    for n in range(1, MAX_REPEAT + 1):
        v = sense(repeats(stem, n)).vector()
        print(f"  {stem + '×' + str(n):<16} " + "  ".join(f"{x:>8.3f}" for x in v))


def monotonicity():
    """Does the distance from the single word grow with repetition?"""
    print("\n  ── is repetition read as 'more of the same' or as 'unrelated'? " + "─" * 6)
    print(f"  {'stem':<14} {'content: ×1→×5 distances':<34} {'monotone?':>10}")
    verdicts = []
    for stem in STEMS:
        base = sense(stem)
        ds = [feature_distance(base, sense(repeats(stem, n))) for n in range(2, MAX_REPEAT + 1)]
        mono = all(b >= a - 1e-9 for a, b in zip(ds, ds[1:]))
        verdicts.append(mono)
        print(f"  {stem:<14} {' → '.join(f'{d:.3f}' for d in ds):<34} "
              f"{'yes' if mono else 'no':>10}")

    print(f"\n  {'stem':<14} {'hash: ×1→×5 distances':<34} {'monotone?':>10}")
    for stem in STEMS:
        base = hash_sense(stem)
        ds = [feature_distance(base, hash_sense(repeats(stem, n))) for n in range(2, MAX_REPEAT + 1)]
        mono = all(b >= a - 1e-9 for a, b in zip(ds, ds[1:]))
        print(f"  {stem:<14} {' → '.join(f'{d:.3f}' for d in ds):<34} "
              f"{'yes' if mono else 'no':>10}")
    return verdicts


def cross_check():
    """Is ×5 of one stem still nearer its own stem than another stem is?"""
    print("\n  ── does repetition stay inside its own identity? " + "─" * 20)
    print(f"  {'':<16} {'dist to own ×1':>15} {'nearest other stem':>22}")
    ok = []
    for stem in STEMS:
        far = sense(repeats(stem, MAX_REPEAT))
        own = feature_distance(sense(stem), far)
        others = {o: feature_distance(sense(o), far) for o in STEMS if o != stem}
        best = min(others, key=others.get)
        held = own < others[best]
        ok.append(held)
        print(f"  {stem + '×5':<16} {own:>15.3f} {best + ' ' + f'{others[best]:.3f}':>22}"
              f"   {'holds' if held else 'LOST'}")
    return ok


def engine_reading():
    """Reading 3 — what the settled engine state does with repetition."""
    try:
        import torch
        from bench_mitosis_calibration import calibrate, make_engine
        from bench_psi_mitosis import stimulus_draw, DIM
    except ImportError as exc:
        print(f"\n  (engine reading skipped — {exc})")
        return

    derived, _ = calibrate(stimulus_draw())
    print(f"\n  ── engine state after {ENGINE_STEPS} steps "
          f"(derived split bar {derived:.4f}) " + "─" * 8)
    print(f"  {'input':<16} {'cells':>6} {'|state|':>9} "
          f"{'cos to ×1':>10}")

    def settle(text):
        v = np.array(sense(text).vector(), dtype=np.float32)
        reps = int(math.ceil(DIM / len(FEATURE_NAMES)))
        x = torch.tensor(np.tile(v, reps)[:DIM]).unsqueeze(0)
        eng = make_engine("absolute", derived, 0)
        for _ in range(ENGINE_STEPS):
            eng.process(x)
        st = torch.stack([c.hidden.squeeze(0) for c in eng.cells]).mean(dim=0)
        return st.detach().numpy(), len(eng.cells)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    for stem in STEMS[:2]:
        base_state = None
        for n in range(1, 4):
            state, n_cells = settle(repeats(stem, n))
            if base_state is None:
                base_state = state
            sim = cos(base_state, state)
            print(f"  {stem + '×' + str(n):<16} {n_cells:>6} "
                  f"{np.linalg.norm(state):>9.4f} {sim:>10.4f}")

    # Baseline: how far apart are DIFFERENT stems in the same state space?
    # Without this the repetition numbers have no scale.
    print("\n  baseline — different words, same measurement:")
    states = {st: settle(st)[0] for st in STEMS}
    for i, a in enumerate(STEMS):
        for b in STEMS[i + 1:]:
            print(f"    {a:>10} vs {b:<12} cos {cos(states[a], states[b]):.4f}")


DEEP_N = (1, 2, 3, 5, 8, 13, 20, 30, 50, 100, 200)
PAIR = ("서예", "만다라")


def saturation_probe():
    """Push repetition far. Does either path keep meaning anything?

    Two failure modes to tell apart:
      · the hash never converges — every N is fresh noise, so repetition can
        never be recognised as repetition;
      · qualia_sense may CONVERGE — `length` is min(n/16, 1) and saturates at
        16 characters, `bigram_repeat` asymptotes to 1, so past some N the
        pattern may freeze and stop distinguishing 20 repeats from 200.

    The null is the MEAN distance between two independent uniform [0,1]^8
    vectors, estimated by simulation. sqrt(8/6) = 1.155 is the root-MEAN-SQUARE
    distance and overstates the mean (Jensen), so using it would make ordinary
    hash noise look like commonality.
    """
    rng = np.random.default_rng(0)
    null = float(np.mean(np.linalg.norm(
        rng.random((200000, 8)) - rng.random((200000, 8)), axis=1)))
    a, b = PAIR
    print("\n  ── pushing repetition far · " + " vs ".join(PAIR) + " " + "─" * 22)
    print(f"  null = simulated mean distance between independent random "
          f"vectors: {null:.3f}\n")
    print(f"  {'N':>5} {'content 무늬 (서예×N)':<12} {'A↔B content':>12} {'A↔B hash':>10} "
          f"{'ΔN content':>11} {'ΔN hash':>9}")

    prev_c = prev_h = None
    rows = []
    for n in DEEP_N:
        ca, cb = sense(repeats(a, n)), sense(repeats(b, n))
        ha, hb = hash_sense(repeats(a, n)), hash_sense(repeats(b, n))
        d_c, d_h = feature_distance(ca, cb), feature_distance(ha, hb)
        step_c = feature_distance(prev_c, ca) if prev_c is not None else float("nan")
        step_h = feature_distance(prev_h, ha) if prev_h is not None else float("nan")
        prev_c, prev_h = ca, ha
        rows.append((n, d_c, d_h, step_c, step_h))
        sc = "  —  " if math.isnan(step_c) else f"{step_c:>11.4f}"
        sh = "  —  " if math.isnan(step_h) else f"{step_h:>9.4f}"
        print(f"  {n:>5} {bar(ca.vector()):<12} {d_c:>12.4f} {d_h:>10.4f} {sc} {sh}")

    tail = [r for r in rows if r[0] >= 20]
    c_steps = [r[3] for r in tail]
    h_steps = [r[4] for r in tail]
    h_dists = [r[2] for r in rows]
    print(f"\n  N ≥ 20 — content step size: {np.mean(c_steps):.5f}  "
          f"(frozen if ≈ 0)")
    print(f"  N ≥ 20 — hash step size:    {np.mean(h_steps):.5f}  "
          f"(never settles)")
    se = float(np.std(h_dists, ddof=1) / math.sqrt(len(h_dists)))
    z = (np.mean(h_dists) - null) / se if se > 0 else float("nan")
    print(f"  hash A↔B distance across all N: mean {np.mean(h_dists):.3f} "
          f"± {se:.3f} (se)  ·  null {null:.3f}  →  z = {z:+.2f}")
    print(f"  {'no commonality — indistinguishable from noise' if abs(z) < 2 else 'DIFFERS from the null — investigate'}")

    frozen_at = None
    for n, _, _, step_c, _ in rows:
        if not math.isnan(step_c) and step_c < 0.01 and frozen_at is None:
            frozen_at = n
    print(f"  content pattern first freezes (step < 0.01) at N = {frozen_at}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-engine", action="store_true")
    args = ap.parse_args()

    print("\n  반복 무늬 평가 — repetition pattern")
    print("  무늬 = the 8-feature fingerprint, ▁ lowest … █ highest")

    for stem in STEMS[:3]:
        show_patterns(stem)
    show_features("서예")
    mono = monotonicity()
    held = cross_check()

    saturation_probe()

    if not args.no_engine:
        engine_reading()

    print("\n  ── summary " + "─" * 58)
    print(f"  content features grow monotonically with repetition: "
          f"{sum(mono)}/{len(mono)} stems")
    print(f"  repetition stays nearer its own stem than any other: "
          f"{sum(held)}/{len(held)} stems")
    print()


if __name__ == "__main__":
    main()
