#!/usr/bin/env python3
"""bench_relative_population.py — both sides of the ledger relative, together.

42ad42f measured that the per-stimulus split design dies on real language, and
that the cause is on the other side: `merge_threshold = 0.05` is a raw magnitude
against a bare constant, and real corpus inter-cell tension sits below it 100.0%
of the time (median 0.0148, p90 0.0205) against 9.3% on synthetic vectors. Split
rates barely differed (7 vs 8); merges were 21 vs 6.

Fixing one half of the ledger is what produced that result, so this makes both
halves relative at once:

    split   a cell's rank among cells at this instant   > split_quantile
    merge   a pair's rank among pairs at this instant   < merge_quantile

## Pre-registered, before running

**P1 — the point.** Real corpus and synthetic converge. A rank discards both
scale and spread, so the 14× pairwise-similarity gap between corpus sentences
(mean cosine +0.191) and randn vectors (+0.014) should stop mattering. If the two
still diverge, relativising both sides did not address the cause.

**P2 — the risk, and it is the serious one.** A quantile always has a top and
always has a bottom. Both rules therefore fire forever regardless of the data,
and the settling point would be decided by `split_quantile`, `merge_quantile`,
`split_patience` and `merge_patience` — four constants replacing two. That is
not regulation, it is the same defect with a longer parameter list.

**P3 — the test that separates P1 from P2.** Feed two stimulus sets drawn from
the same corpus: NARROW (24 sentences chosen to be mutually similar) and BROAD
(24 chosen to be mutually dissimilar). A population that regulates against the
data must hold more cells on BROAD than on NARROW — more distinct things to
specialise on. If both land on the same number, the quantiles are dictating the
answer and P2 is confirmed. **This is the pass condition; the convergence in P1
is necessary but not sufficient, and on its own it would look like success.**

Recorded here before the first run so the outcome cannot be read backwards.

    .venv/bin/python bench_relative_population.py
"""

import argparse
import itertools

import numpy as np
import torch

from bench_specialization_split import SpecializationEngine
from bench_specialization_corpus import load_stimuli, encode, CORPUS
from bench_rank_tension import DIM, HIDDEN
from qualia_sense import text_vector

STEPS, SEEDS, N_STIMULI = 400, 3, 24
SPLIT_Q, MERGE_Q = 0.9, 0.15


class RelativePopulationEngine(SpecializationEngine):
    """Split and merge both decided by rank, never by a raw magnitude.

    Inherits the per-stimulus split axis. Adds the mirror on the merge side:
    each pair's recorded inter-cell tension becomes its rank among the pairs
    ACTUALLY UPDATED this step. That qualifier matters — above a few cells
    `process` samples `min(4, n-1)` partners per cell rather than all pairs, so
    ranking every stored key would mix this step's values with stale ones.
    Lengths are captured before and after to find the touched keys exactly.
    """

    def __init__(self, *a, merge_quantile=MERGE_Q, **kw):
        super().__init__(*a, **kw)
        self.merge_threshold = merge_quantile      # now a quantile, not a magnitude

    def process(self, text_vec, label=""):
        before = {k: len(v) for k, v in self._inter_tension_history.items()}
        result = super().process(text_vec, label)

        touched = [k for k, v in self._inter_tension_history.items()
                   if len(v) > before.get(k, 0)]
        if len(touched) < 2:
            return result
        raw = np.array([self._inter_tension_history[k][-1] for k in touched])
        order = np.argsort(np.argsort(raw))            # 0 .. n-1
        denom = len(touched) - 1
        for key, rank in zip(touched, order):
            self._inter_tension_history[key][-1] = float(rank) / denom
        return result


def build_sets(lines, k=N_STIMULI):
    """NARROW = mutually similar, BROAD = mutually dissimilar, same corpus.

    Greedy on the cosine matrix: NARROW grows by always adding the sentence with
    the highest mean similarity to what is already chosen, BROAD by the lowest.
    Both draw from one pool so the only difference is internal spread.
    """
    mat = np.array([text_vector(s, DIM) for s in lines], dtype=np.float64)
    norm = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    sims = norm @ norm.T
    np.fill_diagonal(sims, 0.0)

    def grow(sign):
        chosen = [int(np.argmax(sign * sims.sum(axis=1)))]
        while len(chosen) < k:
            score = sign * sims[:, chosen].mean(axis=1)
            score[chosen] = -np.inf
            chosen.append(int(np.argmax(score)))
        return chosen

    def spread(idx):
        return float(np.mean([sims[a][b] for a, b in itertools.combinations(idx, 2)]))

    narrow, broad = grow(+1), grow(-1)
    return ([lines[i] for i in narrow], spread(narrow),
            [lines[i] for i in broad], spread(broad))


def run(vectors, seed, steps, split_q=SPLIT_Q, merge_q=MERGE_Q,
        max_cells=32, initial_cells=8):
    torch.manual_seed(seed)
    eng = RelativePopulationEngine(
        input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
        initial_cells=initial_cells, max_cells=max_cells, noise_scale=0.05,
        split_threshold=split_q, merge_quantile=merge_q)
    k = len(vectors)
    for i in range(steps):
        eng.stimulus = i % k
        eng.process(vectors[i % k])
    counts = {"split": 0, "merge": 0}
    for ev in eng.event_log:
        if ev.get("type") in counts:
            counts[ev["type"]] += 1
    return len(eng.cells), counts


def avg(vectors, seeds, **kw):
    out = [run(vectors, s, **kw) for s in range(seeds)]
    return (float(np.mean([o[0] for o in out])),
            float(np.mean([o[1]["split"] for o in out])),
            float(np.mean([o[1]["merge"] for o in out])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    kw = dict(steps=args.steps)

    pool = load_stimuli(CORPUS, N_STIMULI * 5)
    narrow, s_narrow, broad, s_broad = build_sets(pool)
    v_narrow, v_broad = encode(narrow), encode(broad)
    torch.manual_seed(99)
    v_synth = [torch.randn(1, DIM) for _ in range(N_STIMULI)]

    print(f"\n  양쪽 상대량 — 분열 = 상위 {SPLIT_Q:.0%} · 병합 = 하위 {MERGE_Q:.0%}")
    print(f"  {args.seeds} seeds × {args.steps} steps · 출발 8 · 바닥 2 · 천장 32\n")

    print("  P1 — 실제 언어와 합성이 수렴하는가 (맨 상수일 때는 2.0 vs 18.0 이었다)")
    print(f"  {'자극':>12} {'세포':>7} {'분열':>7} {'병합':>7}")
    rows = {}
    for name, vecs in (("실제(넓음)", v_broad), ("실제(좁음)", v_narrow),
                       ("합성 randn", v_synth)):
        rows[name] = avg(vecs, args.seeds, **kw)
        c, sp, mg = rows[name]
        print(f"  {name:>12} {c:>7.1f} {sp:>7.1f} {mg:>7.1f}")
    gap = abs(rows["실제(넓음)"][0] - rows["합성 randn"][0])
    print(f"  실제(넓음) ↔ 합성 차이 {gap:.1f}개"
          f"  ({'수렴 — P1 통과' if gap <= 4.0 else '여전히 갈림 — P1 실패'})")

    print(f"\n  P3 — 데이터 구조에 반응하는가 (이게 통과 조건)")
    print(f"  좁은 묶음 내부 유사도 {s_narrow:+.3f} · 넓은 묶음 {s_broad:+.3f}")
    n, b = rows["실제(좁음)"][0], rows["실제(넓음)"][0]
    print(f"  좁음 {n:.1f}개  vs  넓음 {b:.1f}개   차이 {b - n:+.1f}")
    print("  " + ("넓은 쪽이 더 많다 — 집단이 데이터에 반응한다"
                  if b > n + 1.0 else
                  "구별하지 못한다 — 분위수가 답을 정하고 있다 (P2 확정)"))

    print(f"\n  P2 — 상수가 답을 정하는가: 분위수를 흔들면")
    print(f"  {'분열/병합':>12} {'좁음':>7} {'넓음':>7} {'합성':>7}")
    for sq, mq in ((0.8, 0.10), (0.9, 0.15), (0.95, 0.25)):
        r = [avg(v, args.seeds, split_q=sq, merge_q=mq, **kw)[0]
             for v in (v_narrow, v_broad, v_synth)]
        print(f"  {sq:.2f}/{mq:.2f}".rjust(14) +
              "".join(f"{x:>7.1f}" for x in r))

    print(f"\n  천장 16/32/64 (실제 넓음)")
    ceil = [avg(v_broad, args.seeds, max_cells=c, **kw)[0] for c in (16, 32, 64)]
    print("  " + " / ".join(f"{v:.1f}" for v in ceil) +
          f"   ({max(ceil)/max(min(ceil),1e-9):.2f}배)\n")


if __name__ == "__main__":
    main()
