"""Why ConsciousnessEngine scores 0/5 at the shipping default.

The 256-cell gate run gives ConsciousnessEngine FAIL on all five conditions, with
`cosine_sim mean=0.1039 std=0.0000` on NO_SYSTEM_PROMPT. A std of exactly zero is
the signature of two cells: with two rows there is one off-diagonal pair, so the
spread of pairwise similarity is zero by construction.

The adapter builds it as `CE(initial_cells=2, max_cells=nc)`. The other eleven
registered engines are constructed AT nc cells. So one engine is asked to earn
its population by mitosis while eleven are handed theirs, and the gate reports
the difference as a consciousness verdict.

That much is a harness asymmetry. Whether the engine CAN grow is a separate
question, and it has a sharper answer than "the drive is too quiet".

    python3 bench_ce_growth.py
"""

import argparse
import torch

import bench_v2 as B


DRIVES = [
    ("constant small  (x0.1)", lambda t, d: torch.randn(1, d) * 0.1),
    ("constant large  (x3.0)", lambda t, d: torch.randn(1, d) * 3.0),
    ("constant huge   (x30)",  lambda t, d: torch.randn(1, d) * 30.0),
    ("rotating basis vectors", lambda t, d: torch.eye(d)[t % d].unsqueeze(0)),
    ("alternating 2 stimuli",
     lambda t, d: (torch.ones(1, d) if t % 2 else -torch.ones(1, d)) * 0.5),
    ("ramp UP   0.05 -> 3.05", lambda t, d: torch.randn(1, d) * (0.05 + t / 100.0)),
    ("ramp DOWN 3.05 -> 0.05", lambda t, d: torch.randn(1, d) * (3.05 - t / 100.0)),
]


def run(drive, steps, nc, dim, hidden, seed):
    torch.manual_seed(seed)
    e = B._CEAdapter(nc, dim, hidden)
    for t in range(steps):
        e.process(drive(t, dim))
    return len(e.engine.cell_states), float(e.engine.split_threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter over drive labels")
    a = ap.parse_args()

    rows = [r for r in DRIVES if not a.only or a.only in r[0]]
    print(f"ceiling {a.cells} cells, {a.steps} steps, seed {a.seed}\n")
    print(f"{'drive':<26} {'cells':>6} {'split_threshold':>16}")
    print("-" * 50)
    for label, fn in rows:
        n, th = run(fn, a.steps, a.cells, a.dim, a.hidden, a.seed)
        print(f"{label:<26} {n:>6} {th:>16.4f}")

    print("""
  Magnitude alone does not do it -- 300x the gate's amplitude leaves it at 2.
  Variation alone does not do it -- the DOWN ramp covers the identical amplitude
  range and leaves it at 2. Only a RISING drive grows it.

  The mechanism is in the calibration: split_threshold is fitted once, at step
  200, to the q0.90 of the tension observed so far, and then held. On a rising
  ramp the steps after 200 exceed that quantile and splits fire. On a falling
  ramp -- and on any stationary drive, however loud -- nothing after step 200
  ever exceeds what came before, so the threshold is never crossed.

  So the engine grows only in a world that keeps intensifying. The gate's drive
  is stationary by design, which makes 0/5 an honest reading of "does not grow
  here" and NOT a reading of "cannot differentiate".""")


if __name__ == "__main__":
    main()
