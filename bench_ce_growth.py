"""Why ConsciousnessEngine scores 0/5 at the shipping default.

The 256-cell gate run gives ConsciousnessEngine FAIL on all five conditions, with
`cosine_sim mean=0.1039 std=0.0000` on NO_SYSTEM_PROMPT. A std of exactly zero is
the signature of two cells: with two rows there is one off-diagonal pair, so the
spread of pairwise similarity is zero by construction.

The adapter builds it as `CE(initial_cells=2, max_cells=nc)`. The other eleven
registered engines are constructed AT nc cells. So one engine is asked to earn
its population by mitosis while eleven are handed theirs, and the gate reports
the difference as a consciousness verdict.

WHAT DECIDES WHETHER IT GROWS

`split_threshold` is fitted ONCE, at step 200, to the q0.90 of the tension seen
so far, and then held for the rest of the run. If that window's tension sample is
degenerate -- `std/mean < 0.1` -- the calibrator refuses (correctly: any quantile
of a degenerate sample lands on the operating point itself) and the threshold
stays at its unreachable default of 0.3. Permanently. Nothing after step 200 can
undo it.

A sample goes degenerate at BOTH ends. Too quiet and every cell's tension is
equally near zero; too loud and every cell saturates equally. So growth happens
inside a band of drive amplitude and nowhere else, and the gate's drive
(`randn * 0.1`) sits below the band's lower edge.

    python3 bench_ce_growth.py --steps 1500

TWO RETRACTIONS, both from running this file:

  1. "Only a rising drive grows it." Measured at 300 steps, where constant
     drives showed 2 cells. They grow at 1500 -- constant x1.0 reaches 31 and
     x3.0 reaches 32. A 300-step window leaves 100 steps after a calibration
     that happens at 200, which is not enough for `split_patience` to fire. The
     window was barely longer than the thing it was measuring.

  2. "Rising versus falling is the variable." It is not. The DOWN ramp fails
     because its first 200 steps sit at amplitude 15-13, above the band -- not
     because it descends. Only the calibration window matters; the rest of the
     run has no say.
"""

import argparse
import torch

import bench_v2 as B


DRIVES = [
    ("constant x0.1  (the gate)", lambda t, d: torch.randn(1, d) * 0.1),
    ("constant x0.3",             lambda t, d: torch.randn(1, d) * 0.3),
    ("constant x1.0",             lambda t, d: torch.randn(1, d) * 1.0),
    ("constant x3.0",             lambda t, d: torch.randn(1, d) * 3.0),
    ("constant x10",              lambda t, d: torch.randn(1, d) * 10.0),
    ("constant x30",              lambda t, d: torch.randn(1, d) * 30.0),
    ("ramp UP   0.05 -> 15.0",
     lambda t, d: torch.randn(1, d) * (0.05 + t / 100.0)),
    ("ramp DOWN 15.0 -> 0.05",
     lambda t, d: torch.randn(1, d) * max(0.05, 15.0 - t / 100.0)),
]


def run(drive, steps, nc, dim, hidden, seed):
    torch.manual_seed(seed)
    e = B._CEAdapter(nc, dim, hidden)
    for t in range(steps):
        e.process(drive(t, dim))
    return len(e.engine.cell_states), float(e.engine.split_threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500,
                    help="must be well past the step-200 calibration; 300 is "
                         "short enough to report growth as absent (see the "
                         "retractions in this file's docstring)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args()

    if a.steps < 600:
        print(f"  ! {a.steps} steps leaves {a.steps - 200} after calibration; "
              f"growth can read as absent when it is only slow\n")

    rows = [r for r in DRIVES if not a.only or a.only in r[0]]
    print(f"ceiling {a.cells} cells, {a.steps} steps, seed {a.seed}\n")
    print(f"{'drive':<26} {'cells':>6} {'split_threshold':>16}")
    print("-" * 50)
    for label, fn in rows:
        n, th = run(fn, a.steps, a.cells, a.dim, a.hidden, a.seed)
        print(f"{label:<26} {n:>6} {th:>16.4f}", flush=True)

    print("""
  Growth lives inside a band of drive amplitude. Below it and above it the
  tension sample at step 200 is degenerate, the calibrator refuses, and
  split_threshold stays at an unreachable 0.3 for the rest of the run.

  The gate drives at x0.1, below the lower edge. That makes its 0/5 an honest
  reading of "does not grow here" and NOT a reading of "cannot differentiate" --
  and it makes the deployed runtime's cell count a question about what the
  world's amplitude is, not about the engine.""")


if __name__ == "__main__":
    main()
