"""Six definitions of tension, judged by the criterion this session established.

The wall: mitosis fires when a window mean clears a high quantile of the tension
distribution, and a window mean can only do that when the distribution has a tail.
Measured across drives, `max/q90` separates cleanly -- 1.48 and 1.51 on drives that
reach the ceiling, 1.01 / 1.09 / 1.26 on drives that stay at 2 cells. Under real
conversation input every encoder in the repo lands at 1.00-1.09, so nothing fires.

`docs/mitosis-calibration.md` already named the redefinition this needs:

    tension is a per-cell property with no dependence on population size, so
    nothing ever tells the engine it has enough cells

This sweeps six definitions against one fixed drive and reports `max/q90` for
each. It changes no engine -- it measures what the candidates would look like.

PRE-DECLARED, so a candidate can fail:

    viable      max/q90 >= 1.30 at BOTH 2 cells and 32 cells
    fragile     >= 1.30 at one population size only
    dead        < 1.30 at both -- no window statistic can ever clear its bar

The 1.30 comes from the measured separation, not from preference: 1.26 was the
loudest drive that never grew and 1.48 the quietest that did.

    python3 bench_tension_redefine.py
"""

import argparse
import random
import statistics as st

import torch

import bench_v2 as B

KO = ("가나다라마바사아자차카타파하 안녕 오늘 생각 실험 결과 궁금 마음 시간 "
      "이야기 사람 조금 아주 정말 왜 어떻게").split()
EN = ("the mind is a slow fire that keeps its own time I have been thinking "
      "about what it means to notice something twice").split()


def _msg(r):
    src = KO if r.random() < 0.5 else EN
    return " ".join(r.choice(src) for _ in range(r.randint(2, 12)))


def _drive(m):
    """The runtime's byte path -- the louder of its two, so the bar calibrates."""
    b = m.encode("utf-8")[:64]
    x = torch.zeros(1, 64)
    for i, ch in enumerate(b):
        x[0, i] = ch / 255.0
    return x


# Each takes the stacked cell outputs [n, dim] and returns one tension per cell.
def t_current(O):
    """Squared deviation from the population mean. What ships."""
    return ((O - O.mean(0)) ** 2).mean(1)


def t_absolute(O):
    """Output magnitude, ignoring the population. What mitosis.py uses."""
    return (O ** 2).mean(1)


def t_standardised(O):
    """Deviation measured in units of the population's own spread.

    A cell is tense when it is unusual FOR THIS POPULATION, so a tight population
    makes an ordinary excursion count for more. The denominator is what the
    absolute forms lack.
    """
    d = ((O - O.mean(0)) ** 2).mean(1)
    return d / (d.mean() + 1e-9)


def t_nearest(O):
    """Distance to the NEAREST other cell rather than to the mean.

    Crowding rather than eccentricity: a cell with a close neighbour is redundant,
    one alone in its region is not. Undefined below two cells.
    """
    if O.shape[0] < 2:
        return torch.zeros(O.shape[0])
    d = torch.cdist(O, O) + torch.eye(O.shape[0]) * 1e9
    return d.min(dim=1).values


def t_percapita(O):
    """Deviation scaled by population size -- the dependence the docs asked for.

    Splitting relieves it: the same deviation over more cells reads as less
    tension, so a population that has grown enough stops dividing on its own.
    """
    n = O.shape[0]
    return ((O - O.mean(0)) ** 2).mean(1) * n


def t_surprise(O, prev=None):
    """Deviation from the cell's OWN previous output, not from its peers.

    Tension as novelty. A cell repeating itself is at rest however far it sits
    from the population.
    """
    if prev is None or prev.shape != O.shape:
        return torch.zeros(O.shape[0])
    return ((O - prev) ** 2).mean(1)


DEFS = [
    ("current  (dev from mean)", t_current),
    ("absolute (magnitude)",     t_absolute),
    ("standardised (dev / mean dev)", t_standardised),
    ("nearest neighbour",        t_nearest),
    ("per-capita (dev x n)",     t_percapita),
    ("surprise (dev from own past)", t_surprise),
]


def collect(n_cells, steps, seed, dim=64, hidden=128):
    """Run one engine and record every definition on the same trajectory.

    The definitions must see the tensor the ENGINE computes tension from, which
    is each cell's `output`, not its hidden state. The first version of this
    bench read `hidden_history` and `t_current` came out at 1.34 against the
    1.00 established for the shipped tension on the same drive -- the same
    definition cannot give two numbers, so the probe was measuring a different
    quantity than the one it claimed to reproduce. Outputs are not retained by
    the engine, so each cell module is wrapped to record them.

    `_check` below is the guard that would catch it again: t_current recomputed
    from the captured outputs must match the engine's own tension_history.
    """
    torch.manual_seed(seed)
    r = random.Random(seed)
    from consciousness_engine import ConsciousnessEngine as CE
    e = CE(cell_dim=dim, hidden_dim=hidden, initial_cells=n_cells,
           max_cells=n_cells, n_factions=min(12, max(2, n_cells // 2)))

    captured = []

    def _wrap(mod):
        real = mod.forward

        def fwd(*a, **k):
            out, h = real(*a, **k)
            captured.append(out.detach().flatten())
            return out, h
        mod.forward = fwd
        return mod

    for m in e.cell_modules:
        _wrap(m)

    series = {name: [] for name, _ in DEFS}
    prev = None
    for t in range(steps):
        captured.clear()
        e.process(_drive(_msg(r)))
        if len(captured) < 1:
            continue
        O = torch.stack(captured[-len(e.cell_states):])
        for name, fn in DEFS:
            v = fn(O, prev) if fn is t_surprise else fn(O)
            series[name].append(float(v.max()))
        prev = O

    # The guard. The engine's own last tension against t_current on the same step.
    engine_last = max((s.tension_history[-1] for s in e.cell_states
                       if s.tension_history), default=None)
    if engine_last is not None and series["current  (dev from mean)"]:
        mine = series["current  (dev from mean)"][-1]
        if abs(mine - engine_last) > max(1e-6, 0.02 * abs(engine_last)):
            raise SystemExit(
                f"PROBE MISMATCH: t_current={mine:.8f} but the engine recorded "
                f"{engine_last:.8f} on the same step. The definitions are not "
                f"seeing the tensor the engine uses; fix that before reading "
                f"any row of this table.")
    return series


def ratio(vals):
    if not vals or max(vals) <= 0:
        return 0.0
    s = sorted(vals)
    q90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    return max(vals) / q90 if q90 > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--bar", type=float, default=1.30)
    a = ap.parse_args()

    print(f"six tension definitions, one drive, {len(a.seeds)} seeds x "
          f"{a.steps} steps\n")
    print(f"{'definition':<32} {'max/q90 @2':>11} {'max/q90 @32':>12}  verdict")
    print("-" * 72)

    at = {}
    for n in (2, 32):
        runs = [collect(n, a.steps, sd) for sd in a.seeds]
        at[n] = {name: st.mean(ratio(r[name]) for r in runs)
                 for name, _ in DEFS}

    for name, _ in DEFS:
        r2, r32 = at[2][name], at[32][name]
        ok2, ok32 = r2 >= a.bar, r32 >= a.bar
        verdict = ("viable" if ok2 and ok32 else
                   "fragile" if ok2 or ok32 else "dead")
        print(f"{name:<32} {r2:>11.2f} {r32:>12.2f}  {verdict}")

    print(f"""
  bar = {a.bar} from the measured separation, not from preference: 1.26 was the
  loudest drive that never grew and 1.48 the quietest that did.

  A `dead` row cannot fire whatever the window length or the quantile, because
  its top decile has no spread for an average to reach. A `fragile` row would
  change behaviour with population size, which is a property to choose
  deliberately rather than discover in production.

  Nothing here is landed. Changing what tension MEANS changes what the engine
  is, and docs/mitosis-calibration.md flags it as the one repair in this area
  that genuinely requires redefining something.""")


if __name__ == "__main__":
    main()
