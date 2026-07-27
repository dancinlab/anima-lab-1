#!/usr/bin/env python3
"""bench_mitosis_calibration.py — an absolute bar on a scale-free quantity.

Pre-registered as docs/hypotheses/QD-6-tension-calibration.md. Arms, the
threshold-derivation rule, k, and every bar were fixed before the first run.

mitosis.py:60 defines `tension = (output ** 2).mean()`, an absolute magnitude,
and gates division on an absolute `split_threshold`. Whether a cell ever
divides is therefore decided by how large the caller's vectors happen to be.
Measured: the engine's own randn default falls 3.6× short of the bar and
mitosis.demo()'s text_to_vector falls 150× short, while ~5× the default input
jumps straight to max_cells. Threshold-driven mitosis has never fired on any
path in this repo — the demo's one MITOSIS line is a forced split_cell() call.

  C  control    unchanged engine
  A  absolute   split_threshold re-derived from measured tension (median + 2·sd)
  R  relative   split when tension is k sd above THAT CELL's own running mean

Neither arm modifies mitosis.py. Only a winner earns an engine change.

Requires torch — run under the repo .venv:
    .venv/bin/python bench_mitosis_calibration.py
"""

import argparse
import numpy as np
import torch

from mitosis import MitosisEngine
from bench_psi_mitosis import (DIM, HIDDEN, MAX_CELLS, SETTLE_FRACTION,
                               stimulus_draw, stimulus_vector, retention)
from qualia_sense import sense

# ── pre-registered constants ───────────────────────────────
SEEDS = 3
STEPS = 600
K_SD = 2.0                      # arm R: "unusually high for this cell"
DERIVE_SD_MULT = 2.0            # arm A: threshold = median + 2·sd of measured tension
CALIB_STEPS = 200
SCALE_FACTOR = 10.0             # H23: multiply every input by this
H21_CELLS_MIN = 2               # strictly greater
H22_CELLS_MAX = MAX_CELLS       # strictly less
H23_DRIFT_MAX = 0.20            # arm R must move less than this under ×10
H24_RETENTION_MIN = 0.50
PROBE = 500


class RelativeMitosisEngine(MitosisEngine):
    """Arm R — split on a per-cell z-score instead of an absolute bar.

    Scale-invariant by construction: multiplying every input by any constant
    shifts a cell's mean and sd together and leaves the z-score unchanged.
    """

    def __init__(self, *a, k_sd=K_SD, **kw):
        super().__init__(*a, **kw)
        self.k_sd = k_sd

    def _check_splits(self):
        events = []
        if len(self.cells) >= self.max_cells:
            return events

        to_split = []
        for cell in self.cells:
            hist = cell.tension_history
            if len(hist) < max(self.split_patience, 20):
                continue
            base = np.array(hist[:-self.split_patience])
            sd = float(base.std())
            if sd < 1e-12:
                continue
            mean = float(base.mean())
            recent = hist[-self.split_patience:]
            if all((t - mean) / sd > self.k_sd for t in recent):
                to_split.append(cell)

        for cell in to_split:
            if len(self.cells) >= self.max_cells:
                break
            event = self.split_cell(cell)
            if event:
                events.append(event)
        return events


def make_engine(arm, threshold, seed):
    torch.manual_seed(seed)
    cls = RelativeMitosisEngine if arm == "relative" else MitosisEngine
    kw = dict(input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
              initial_cells=2, max_cells=MAX_CELLS)
    if arm != "relative":
        kw["split_threshold"] = threshold
    return cls(**kw)


def run_one(arm, threshold, name, seed, steps, scale=1.0):
    engine = make_engine(arm, threshold, seed)
    x = stimulus_vector(name) * scale
    trace, probe_state = [], None
    for step in range(1, steps + 1):
        engine.process(x)
        pooled = torch.stack([c.hidden.squeeze(0) for c in engine.cells]).mean(dim=0)
        trace.append(pooled.detach().numpy())
        if step == PROBE:
            probe_state = trace[-1].copy()
    trace = np.array(trace)
    early = float(np.abs(np.diff(trace[:10], axis=0)).mean())
    late = float(np.abs(np.diff(trace[-100:], axis=0)).mean())
    return dict(cells=len(engine.cells), probe=probe_state, final=trace[-1],
                settled=(late <= SETTLE_FRACTION * early) if early > 0 else False)


def calibrate(names, seed=0):
    """Arm A's threshold, derived from measurement — not chosen."""
    engine = make_engine("control", 0.3, seed)
    ts = []
    for i in range(CALIB_STEPS):
        engine.process(stimulus_vector(names[i % len(names)]))
        ts.extend(c.tension_history[-1] for c in engine.cells if c.tension_history)
    ts = np.array(ts)
    return float(np.median(ts) + DERIVE_SD_MULT * ts.std()), ts


def evaluate(arm, threshold, names, feats, seeds, steps, scale=1.0):
    cells, probes, finals, settled = [], [], [], []
    for seed in seeds:
        seed_probes = []
        for name in names:
            r = run_one(arm, threshold, name, seed, steps, scale)
            cells.append(r["cells"])
            settled.append(r["settled"])
            finals.append(r["final"])
            if r["probe"] is not None:
                seed_probes.append(r["probe"])
        if seed_probes:
            probes.append(retention(seed_probes, feats))
    return dict(cells=float(np.mean(cells)),
                retention=float(np.mean(probes)) if probes else float("nan"),
                settled=float(np.mean(settled)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    names = stimulus_draw()
    feats = [sense(n).vector() for n in names]
    seeds = list(range(args.seeds))

    print(f"\n  QD-6 · {len(names)} stimuli × {args.seeds} seeds × {args.steps} steps")
    if args.steps < PROBE:
        print(f"  ⚠️  --steps {args.steps} is short of probe step {PROBE} "
              f"— retention reports n/a, not a pass")

    derived, ts = calibrate(names)
    print(f"  arm A threshold derived from measurement: median {np.median(ts):.4f} "
          f"+ {DERIVE_SD_MULT}·sd {ts.std():.4f} = {derived:.4f}  (default 0.3)\n")

    arms = (("control", 0.3), ("absolute", derived), ("relative", None))
    print(f"  {'arm':<10} {'threshold':>10} {'cells':>7} {'settled':>9} {'retained':>9}")
    res = {}
    for arm, thr in arms:
        r = evaluate(arm, thr, names, feats, seeds, args.steps)
        res[arm] = r
        tf = "z>2.0sd" if thr is None else f"{thr:.4f}"
        rf = "n/a" if not np.isfinite(r["retention"]) else f"{r['retention']:.3f}"
        print(f"  {arm:<10} {tf:>10} {r['cells']:>7.1f} {r['settled']:>8.0%} {rf:>9}")

    print(f"\n  ── scale invariance (every input × {SCALE_FACTOR:g}) " + "─" * 24)
    scaled = {}
    for arm, thr in (("absolute", derived), ("relative", None)):
        r = evaluate(arm, thr, names, feats, seeds, args.steps, scale=SCALE_FACTOR)
        scaled[arm] = r
        base = res[arm]["cells"]
        drift = abs(r["cells"] - base) / base if base > 0 else float("inf")
        print(f"  {arm:<10} cells {base:>5.1f} → {r['cells']:>5.1f}   drift {drift:>6.1%}")

    def drift(arm):
        b = res[arm]["cells"]
        return abs(scaled[arm]["cells"] - b) / b if b > 0 else float("inf")

    formed = [a for a in ("absolute", "relative")
              if res[a]["cells"] > H21_CELLS_MIN]
    h21 = bool(formed) and res["control"]["cells"] == 2
    h22 = any(res[a]["cells"] < H22_CELLS_MAX for a in formed)
    h23 = (drift("relative") < H23_DRIFT_MAX) and (drift("absolute") >= H23_DRIFT_MAX)
    h24 = any(np.isfinite(res[a]["retention"])
              and res[a]["retention"] >= H24_RETENTION_MIN
              and res[a]["settled"] >= 0.5 for a in formed)

    print("\n  ── pre-registered hypotheses " + "─" * 42)
    print(f"  H21  a population forms (control stays 2)   "
          f"formed={formed or 'none'} control={res['control']['cells']:.1f}"
          f"{'':>3}{'PASS' if h21 else 'FAIL'}")
    cells_str = " · ".join(f"{a}={res[a]['cells']:.1f}" for a in formed) or "—"
    ret_str = " · ".join(f"{a}={res[a]['retention']:.3f}" for a in formed) or "—"
    print(f"  H22  and does not pin to the ceiling ({H22_CELLS_MAX})   "
          f"{cells_str}{'':>6}{'PASS' if h22 else 'FAIL'}")
    print(f"  H23  relative scale-invariant, absolute not  "
          f"R drift {drift('relative'):.1%} · A drift {drift('absolute'):.1%}"
          f"{'':>2}{'PASS' if h23 else 'FAIL'}")
    print(f"  H24  retention ≥ {H24_RETENTION_MIN} while settled          "
          f"{ret_str}{'':>4}{'PASS' if h24 else 'FAIL'}")

    ok = all([h21, h22, h23, h24])
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} ({sum([h21, h22, h23, h24])}/4)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
