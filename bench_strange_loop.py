#!/usr/bin/env python3
"""bench_strange_loop.py — experiment 2: close the loop on A, on G, and on the pair.

Hofstadter's strange loop is a tangled hierarchy: you climb through levels and
arrive where you started. That, as stated, is not measurable here. What IS
measurable is the mechanical core it needs — a system whose own output re-enters
as its input, and whose trajectory then returns to its own neighbourhood.

So three properties, in the order that makes the later ones meaningful:

    1. SELF-FEEDING   does it sustain with the loop as its only drive?
    2. LOOP EFFECT    does closing the loop actually change the trajectory,
                      or would the same thing happen with the output discarded?
    3. RECURRENCE     does the trajectory revisit its own past more than a
                      time-shuffled null of itself does?

(2) is the load-bearing one. Without it, (1) and (3) describe a system that would
behave identically with the loop cut — which is the shape this session has caught
nine times: a channel that looks structural and carries nothing.

ARMS. `PairFieldEngine` splits into A (Hebbian rule) and G (no rule), both keeping
an accumulating coupling matrix. Each is looped alone and then together:

    A-loop        A's output returns to A
    G-loop        G's output returns to G
    PAIR-loop     the pair's combined output returns to BOTH sides
    <arm>-open    same engine, output computed and DISCARDED, driven externally
                  — the control that makes (2) falsifiable

Plus a corpse: SCRAMBLE-loop, the pair with cell identity destroyed each step,
looped the same way. If a scrambled population scores what the real one scores,
the measurement is not about loops.

    .venv/bin/python bench_strange_loop.py
"""

import argparse

import numpy as np
import torch

from bench_v2 import PhiIIT
from pairfield_engine import PairFieldEngine, PairSide

CELLS, DIM, HIDDEN, STEPS, SEEDS = 16, 32, 64, 600, (42, 43, 44)


class SoloSide:
    """One side of the pair, standing alone, with the gate's engine interface."""

    def __init__(self, ruled, n_cells, dim, hidden, seed=0):
        f = min(8, max(n_cells // 2, 1))
        self.side = PairSide(n_cells, dim, hidden, dim, f, ruled=ruled, seed=seed)
        self.n_cells, self.input_dim = n_cells, dim

    def process(self, x):
        out, t = self.side.process(x)
        self.side.update_coupling(self.side.get_hiddens())
        return out, t

    def get_hiddens(self):
        return self.side.get_hiddens()


class Scrambled:
    """The pair with cell identity destroyed every step. Same statistics, no cells."""

    def __init__(self, n_cells, dim, hidden):
        self.eng = PairFieldEngine(n_cells, dim, hidden, dim)
        self.n_cells, self.input_dim = n_cells, dim

    def process(self, x):
        out, t = self.eng.process(x)
        h = self.eng.A.get_hiddens()
        self.eng.A.hiddens = h[torch.randperm(h.shape[0])]
        return out, t

    def get_hiddens(self):
        return self.eng.get_hiddens()


def build(kind, seed, strength):
    torch.manual_seed(seed)
    if kind == "A":
        return SoloSide(True, CELLS, DIM, HIDDEN, seed=0)
    if kind == "G":
        return SoloSide(False, CELLS, DIM, HIDDEN, seed=1)
    if kind == "PAIR":
        e = PairFieldEngine(CELLS, DIM, HIDDEN, DIM); e.strength = strength
        return e
    if kind == "SCRAMBLE":
        e = Scrambled(CELLS, DIM, HIDDEN); e.eng.strength = strength
        return e
    raise ValueError(kind)


def run(kind, seed, closed, strength):
    """closed=True: output becomes the next input. closed=False: output discarded."""
    eng = build(kind, seed, strength)
    calc = PhiIIT()
    torch.manual_seed(seed + 5000)                 # drive stream, same both arms
    drive = [torch.randn(1, DIM) * 0.1 for _ in range(STEPS)]

    x = drive[0]
    states, phis, peak = [], [], 0.0
    for t in range(STEPS):
        out, _tension = eng.process(x)
        h = eng.get_hiddens()
        # Check THIS step's value for finiteness before folding it into the
        # running max. `max(2.425, nan)` returns 2.425 in Python -- the NaN
        # comparison is False, so the running max never updates and the
        # `isfinite(peak)` test below passes forever. The first version of this
        # guard reported `peak |h| = 2.425` for a state that was already NaN,
        # and only the separate `alive` ratio exposed it. A guard that cannot
        # see the failure it exists for is the shape this session keeps finding.
        cur = float(h.abs().max())
        if not np.isfinite(cur) or cur > 1e4:
            return None                            # diverged: says nothing
        peak = max(peak, cur)
        # The loop, or its absence. In both arms the output is computed, so the
        # only difference is whether it returns.
        x = out.detach() if closed else drive[t]
        if t % 10 == 0:
            states.append(h.flatten().numpy().copy())
            phis.append(calc.compute(h)[0])
    st = np.array(states)
    return {"states": st, "phi": phis, "peak": peak,
            # Norm of the state at the end vs the start. Near zero means the loop
            # did not merely fail to change the trajectory -- it killed it, which
            # is a different finding and was hidden behind `effect = nan`.
            "alive": float(np.linalg.norm(st[-1]) / max(np.linalg.norm(st[0]), 1e-12))}


def recurrence(states, rng):
    """Does the trajectory revisit its own neighbourhood, against a shuffled null?

    Fraction of step pairs (i, j>i+2) whose distance is below the 5th percentile
    of all distances, minus the same fraction on a time-shuffled copy. Positive
    means the ORDER of the trajectory matters — it returns; shuffling destroys it.
    """
    u = states / np.clip(np.linalg.norm(states, axis=1, keepdims=True), 1e-9, None)
    d = 1.0 - (u @ u.T)
    n = len(u)
    iu = np.triu_indices(n, k=3)
    thresh = np.percentile(d[iu], 5)
    real = float((d[iu] < thresh).mean())
    perm = rng.permutation(n)
    ds = d[np.ix_(perm, perm)]
    null = float((ds[iu] < thresh).mean())
    return real - null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--strength", type=float, default=0.01)
    args = ap.parse_args()
    globals()["STEPS"] = args.steps
    rng = np.random.default_rng(0)

    print(f"\n  실험 2 — 루프를 A 에, G 에, 그리고 쌍에 건다")
    print(f"  {CELLS} cells · {args.steps} steps · seeds {SEEDS}\n")
    print(f"  {'구성':>14} {'최대|h|':>9} {'생존':>8} {'Φ 최종':>8} "
          f"{'루프효과':>9} {'재귀':>8}   판정")

    strength = args.strength
    print(f"  반발 강도 {strength}\n")
    for kind in ("A", "G", "PAIR", "SCRAMBLE"):
        rows = []
        for sd in SEEDS:
            try:
                c, o = (run(kind, sd, True, strength),
                        run(kind, sd, False, strength))
            except RuntimeError as exc:
                # The engine's own divergence guard. Report it as the finding it
                # is -- a pair that explodes on its own output in 21 steps -- and
                # keep measuring the other arms.
                rows.append({"diverged": str(exc).split('(')[1].split(')')[0]})
                continue
            if c is None or o is None:
                rows.append({"diverged": "peak > 1e4"})
                continue
            # (2) does closing the loop change the trajectory at all?
            m = min(len(c["states"]), len(o["states"]))
            # Normalise by the LARGER of the two trajectory norms, not the open
            # one. The first version divided by the open trajectory and returned
            # nan for A and G, because their looped state decays to zero -- the
            # same reason their Phi reads 0.0000. A ratio that breaks exactly
            # when the system dies reports nothing about the case that matters.
            scale = max(float(np.linalg.norm(o["states"][:m])),
                        float(np.linalg.norm(c["states"][:m])), 1e-9)
            effect = float(np.linalg.norm(c["states"][:m] - o["states"][:m]) / scale)
            rows.append({"peak": c["peak"], "phi": c["phi"][-1],
                         "effect": effect, "alive": c["alive"],
                         "rec": recurrence(c["states"], rng)})
        div = [r for r in rows if "diverged" in r]
        if div:
            print(f"  {kind:>14}  {len(div)}/{len(rows)} 시드 발산 — "
                  f"{div[0]['diverged']} · 점수 없음")
            if len(div) == len(rows):
                continue
            rows = [r for r in rows if "diverged" not in r]
        pk = max(r["peak"] for r in rows)
        ph = float(np.mean([r["phi"] for r in rows]))
        ef = float(np.mean([r["effect"] for r in rows]))
        rc = float(np.mean([r["rec"] for r in rows]))
        al = float(np.mean([r["alive"] for r in rows]))
        verdict = ("루프가 상태를 죽임 — 비교 불가" if al < 0.01 else
                   "루프가 궤적을 안 바꿈 — 장식" if ef < 0.01 else
                   "되먹임이 실제로 작동" if rc > 0.01 else
                   "되먹임은 있으나 되돌아오지 않음")
        print(f"  {kind:>14} {pk:>9.3f} {al:>8.4f} {ph:>8.4f} {ef:>9.4f} "
              f"{rc:>+8.4f}   {verdict}")

    print(f"\n  루프효과 = 닫힌 궤적과 열린 궤적의 상대 거리. 0 이면 출력을 버려도 같다.")
    print(f"  재귀 = 자기 이웃 재방문률 − 시간 섞은 자기 자신의 재방문률.")
    print(f"  SCRAMBLE 이 실제와 같은 값을 내면 이 측정은 루프에 관한 것이 아니다.\n")


if __name__ == "__main__":
    main()
