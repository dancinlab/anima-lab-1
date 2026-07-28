"""Does a population-dependent tension stop the engine at an interior size?

`bench_tension_redefine.py` scored six definitions by tail headroom at FIXED
population sizes, and said so about its own blind spot: multiplying tension by n
is a constant factor within a step, so `max/q90` cannot see what `per-capita` is
for. What it is for is feedback -- splitting RELIEVES tension, so a population
that has grown enough stops dividing on its own.

`docs/mitosis-calibration.md` states the problem this addresses:

    tension is a per-cell property with no dependence on population size, so
    nothing ever tells the engine it has enough cells

and records the consequence: every working configuration pins to `max_cells` or
to `min_cells`, with nothing between. QD-6 pre-registered that pinning to the
ceiling is the same failure as pinning to the floor.

This measures the thing that metric could not: run the population dynamics with
each feedback law and see WHERE it settles.

    tension_i  =  dev_i  x  f(n)

    f(n) = 1          no feedback -- what ships
    f(n) = n          per-capita: crowding raises tension, growth raises it more
    f(n) = 1/n        relief: each split divides the tension among more cells
    f(n) = 1/sqrt(n)  weaker relief

PRE-DECLARED, so a law can fail:

    settles   final population strictly between min_cells and max_cells on a
              majority of seeds, and within 1.5x when the ceiling doubles
    pins      lands on the floor or the ceiling
    unstable  neither, or the two ceilings disagree by more than 1.5x

The ceiling-doubling clause is the part that matters. A law that "settles" at 30
of 32 and at 60 of 64 has not settled -- it is pinned and hiding it.

    python3 bench_tension_feedback.py
"""

import argparse
import random
import statistics as st

import torch

KO = ("가나다라마바사아자차카타파하 안녕 오늘 생각 실험 결과 궁금 마음 시간 "
      "이야기 사람 조금 아주 정말 왜 어떻게").split()
EN = ("the mind is a slow fire that keeps its own time I have been thinking "
      "about what it means to notice something twice").split()


def _msg(r):
    src = KO if r.random() < 0.5 else EN
    return " ".join(r.choice(src) for _ in range(r.randint(2, 12)))


def _drive(m):
    b = m.encode("utf-8")[:64]
    x = torch.zeros(1, 64)
    for i, ch in enumerate(b):
        x[0, i] = ch / 255.0
    return x


LAWS = [
    ("f(n) = 1        (ships)", lambda n: 1.0),
    ("f(n) = n        (per-capita)", lambda n: float(n)),
    ("f(n) = 1/n      (relief)", lambda n: 1.0 / n),
    ("f(n) = 1/sqrt(n) (weak relief)", lambda n: n ** -0.5),
]


def run(law, ceiling, steps, seed, dim=64, hidden=128):
    """One engine with tension scaled by f(n), returning the final population.

    The scaling is applied where the engine records tension, so the split rule
    and its calibration both see the scaled quantity -- patching only the split
    comparison would leave the bar calibrated on the unscaled distribution and
    measure a mismatch rather than a feedback law.
    """
    torch.manual_seed(seed)
    r = random.Random(seed)
    from consciousness_engine import ConsciousnessEngine as CE
    e = CE(cell_dim=dim, hidden_dim=hidden, initial_cells=2,
           max_cells=ceiling, n_factions=min(12, max(2, ceiling // 2)))

    real_process = e.process
    traj = []

    def process(x):
        out = real_process(x)
        n = len(e.cell_states)
        k = law(n)
        if k != 1.0:
            for s in e.cell_states:
                if s.tension_history:
                    s.tension_history[-1] *= k
        traj.append(n)
        return out

    e.process = process
    for _ in range(steps):
        e.process(_drive(_msg(r)))
    return len(e.cell_states), traj


def classify(runs, ceiling, min_cells=2):
    """Interior count, mean final size, and whether the population had STOPPED.

    An interior count on a truncated run cannot be told from a population that
    is simply still climbing. The smoke run made that concrete: the shipped law
    read 2/2 interior with mean 19.5 at 400 steps and was scored `settles`,
    when the only thing established is that it had not reached 32 YET.

    `rising` compares the mean population over the last tenth of the run against
    the tenth before it. A law that is still climbing at the end has not settled
    whatever its final number says.
    """
    finals = sorted(f for f, _ in runs)
    interior = sum(1 for f in finals if min_cells < f < ceiling)
    rising = 0
    for _, traj in runs:
        if len(traj) < 20:
            continue
        tenth = max(1, len(traj) // 10)
        late = st.mean(traj[-tenth:])
        prev = st.mean(traj[-2 * tenth:-tenth])
        if late > prev * 1.02:
            rising += 1
    return interior, st.mean(finals), rising, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49])
    ap.add_argument("--ceilings", type=int, nargs="+", default=[32, 64])
    a = ap.parse_args()

    n_s = len(a.seeds)
    print(f"population feedback laws, {n_s} seeds x {a.steps} steps, "
          f"ceilings {a.ceilings}\n")
    head = "  ".join(f"{'@' + str(c):>16}" for c in a.ceilings)
    print(f"{'law':<32}{head}   verdict")
    print("-" * (32 + 18 * len(a.ceilings) + 12))

    for name, law in LAWS:
        cells = {}
        for c in a.ceilings:
            runs = [run(law, c, a.steps, sd) for sd in a.seeds]
            cells[c] = classify(runs, c)
        row = "  ".join(f"{cells[c][0]}/{n_s}int ris{cells[c][2]} "
                        f"{cells[c][3]}" for c in a.ceilings)
        means = [cells[c][1] for c in a.ceilings]
        majority = all(cells[c][0] > n_s / 2 for c in a.ceilings)
        stopped = all(cells[c][2] <= n_s / 2 for c in a.ceilings)
        drift = max(means) / max(min(means), 1e-9)
        verdict = ("settles" if majority and stopped and drift <= 1.5 else
                   "still rising" if majority and not stopped else
                   "pins" if not majority else "unstable")
        print(f"{name:<32}{row}   {verdict}")

    print(f"""
  'int' = seeds landing strictly between {2} and the ceiling · 'ris' = seeds
  still CLIMBING in the last tenth of the run · then every final size, sorted.

  Two things the smoke run forced into this table. An interior count on a
  truncated run cannot be told from a population still climbing, so `ris` is
  reported and a majority of climbers blocks `settles` -- the shipped law read
  2/2 interior at 400 steps only because it had not reached 32 YET. And the mean
  is not printed at all: `f(n)=n` came out 0/2 interior with mean 17.0, which is
  finals of 2 and 32. A mean over a bimodal outcome names a population that
  never occurred.

  The ceiling-doubling clause is the one that bites: a law reaching 30 of 32 and
  60 of 64 is pinned and hiding it behind an interior count, which is why the
  drift between ceilings must stay within 1.5x for `settles`.

  Nothing is landed. This measures what a feedback law WOULD do; whether the
  engine should have one is a design decision, and QD-6 already pre-registered
  that pinning to max_cells is the same failure as pinning to min_cells.""")


if __name__ == "__main__":
    main()
