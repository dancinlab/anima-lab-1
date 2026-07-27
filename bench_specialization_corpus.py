#!/usr/bin/env python3
"""bench_specialization_corpus.py — the same split axis, on real language.

`bench_specialization_split.py` measured per-stimulus aggregation on
`torch.randn` vectors: it settles between the floor and the ceiling (9–14 cells
from a 2-cell start), holds within 1.24× against a 4× ceiling, and cuts scale
drift from 967% to 79%. Three seeds on synthetic vectors is enough to refute a
design and not enough to land one, which is what the ING board records.

This is the gate that record names. Two questions, and the second is the one
that matters:

    1. do the three properties survive real corpus stimuli?
    2. is the specialisation MEANINGFUL, or just a partition?

A cell that consistently leads on some stimulus has divided the work only if the
stimuli it owns belong together. If ownership is arbitrary, "specialisation" is
a name for an arbitrary cut and the design earns nothing.

So: after the run, each stimulus is assigned the cell holding the highest mean
rank for it, and we ask whether same-owner stimulus pairs are more similar to
each other than different-owner pairs. Similarity is cosine on the same
`text_vector` encoding the engine was fed, so the test uses no information the
engine did not have.

The effect size is meaningless on its own — any partition of 24 items into a few
groups produces some within/between gap by chance, and with few cells the gap
can be large. The control is a shuffle of the ownership labels across the same
group sizes, 400 times, which holds the partition structure fixed and destroys
only the correspondence to the engine. That is the null this has to beat.

    .venv/bin/python bench_specialization_corpus.py
"""

import argparse
import random

import numpy as np
import torch

from bench_specialization_split import SpecializationEngine, STIMULI, PATIENCE
from bench_rank_tension import DIM, HIDDEN
from qualia_sense import text_vector

CORPUS = "data/corpus.txt"
STEPS, SEEDS, SHUFFLES = 400, 3, 400
N_STIMULI = 24


def load_stimuli(path, n, seed=0):
    """Distinct sentences sampled from across the file, not from its head.

    The corpus sampler in QD-11 read only the opening bytes and called a clean
    corpus degenerate on the strength of it. Sampling from several offsets is
    the fix that found; keeping it here costs nothing.
    """
    rng = random.Random(seed)
    # Binary, because a byte offset lands mid-character in UTF-8 and text mode
    # raises on the partial one. The discarded first readline handles the same
    # problem at the line level; decoding leniently handles it at the byte level.
    with open(path, "rb") as fh:
        size = fh.seek(0, 2)
        lines, seen = [], set()
        for _ in range(n * 40):
            fh.seek(rng.randrange(0, max(size - 4096, 1)))
            fh.readline()                       # partial line, discard
            line = fh.readline().decode("utf-8", "ignore").strip()
            if line.startswith(("A:", "B:")):
                line = line[2:].strip()
            if 40 <= len(line) <= 200 and line not in seen:
                seen.add(line)
                lines.append(line)
            if len(lines) >= n:
                break
    return lines


def encode(lines):
    return [torch.tensor([text_vector(s, DIM)], dtype=torch.float32) for s in lines]


def run(vectors, bar, seed, steps, max_cells=32, initial_cells=2, scale=1.0):
    torch.manual_seed(seed)
    eng = SpecializationEngine(
        input_dim=DIM, hidden_dim=HIDDEN, output_dim=DIM,
        initial_cells=initial_cells, max_cells=max_cells,
        noise_scale=0.05, split_threshold=bar)
    k = len(vectors)
    for i in range(steps):
        eng.stimulus = i % k
        eng.process(vectors[i % k] * scale)
    return eng


def ownership(eng):
    """stimulus index → id of the cell holding its highest mean rank."""
    owners = {}
    for stim, bucket in eng._by_stimulus.items():
        best, who = -1.0, None
        for cell_id, hist in bucket.items():
            if len(hist) >= PATIENCE:
                score = float(np.mean(hist[-PATIENCE:]))
                if score > best:
                    best, who = score, cell_id
        if who is not None:
            owners[stim] = who
    return owners


def coherence(owners, sims):
    """mean same-owner similarity − mean different-owner similarity."""
    within, between = [], []
    keys = sorted(owners)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (within if owners[a] == owners[b] else between).append(sims[a][b])
    if not within or not between:
        return None
    return float(np.mean(within) - np.mean(between))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    seeds = range(args.seeds)

    lines = load_stimuli(CORPUS, N_STIMULI)
    vectors = encode(lines)
    print(f"\n  실제 코퍼스 자극 — {CORPUS} 에서 문장 {len(lines)}개 "
          f"({args.seeds} seeds × {args.steps} steps · 바닥 2 · 천장 32)")
    print(f"  예: {lines[0][:64]}…\n")

    print("  ① 세 지표가 실제 언어에서도 살아있는가")
    print(f"  {'bar':>6} {'세포':>7}   (합성벡터: 9.0 / 9.3 / 14.0 / 13.0)")
    settle = []
    for bar in (0.7, 0.8, 0.9, 0.95):
        v = float(np.mean([len(run(vectors, bar, s, args.steps).cells) for s in seeds]))
        settle.append(v)
        print(f"  {bar:>6} {v:>7.1f}" + ("  ← 사이" if 2.5 < v < 31.5 else "  ← 바닥/천장"))
    print(f"  기준선 0.7→0.95 정착점 {max(settle)/max(min(settle),1e-9):.2f}배 이동"
          f"  (합성 1.56배)")

    ceil = [float(np.mean([len(run(vectors, 0.9, s, args.steps, max_cells=c).cells)
                           for s in seeds])) for c in (16, 32, 64)]
    print(f"\n  천장 16/32/64 → {ceil[0]:.1f} / {ceil[1]:.1f} / {ceil[2]:.1f}"
          f"   ({max(ceil)/max(min(ceil),1e-9):.2f}배, 합성 1.24배)")

    a = float(np.mean([len(run(vectors, 0.9, s, args.steps).cells) for s in seeds]))
    b = float(np.mean([len(run(vectors, 0.9, s, args.steps, scale=10.0).cells)
                       for s in seeds]))
    print(f"  입력 ×10 → {a:.1f} → {b:.1f}   이동 "
          f"{abs(b-a)/max(a,1e-9)*100:.0f}%   (합성 79% · raw 967%)")

    print("\n  ② 전문화가 의미를 갖는가 — 같은 세포가 맡은 문장끼리 더 비슷한가")
    mat = np.array([text_vector(s, DIM) for s in lines], dtype=np.float64)
    norm = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    sims = norm @ norm.T

    print(f"  {'seed':>5} {'세포':>5} {'묶음':>5} {'효과':>9} {'귀무평균':>9} {'z':>7}")
    zs = []
    rng = random.Random(0)
    for s in seeds:
        eng = run(vectors, 0.9, s, args.steps)
        owners = ownership(eng)
        eff = coherence(owners, sims)
        if eff is None:
            print(f"  {s:>5} {len(eng.cells):>5}     —  묶음 1개 — 비교 불가")
            continue
        labels = list(owners.values())
        null = []
        for _ in range(SHUFFLES):
            rng.shuffle(labels)
            v = coherence(dict(zip(owners, labels)), sims)
            if v is not None:
                null.append(v)
        mu, sd = float(np.mean(null)), float(np.std(null))
        z = (eff - mu) / max(sd, 1e-12)
        zs.append(z)
        print(f"  {s:>5} {len(eng.cells):>5} {len(set(owners.values())):>5} "
              f"{eff:>+9.4f} {mu:>+9.4f} {z:>+7.2f}")

    if zs:
        m = float(np.mean(zs))
        print(f"\n  평균 z = {m:+.2f}  "
              f"({'귀무를 못 넘음 — 임의 분할과 구별되지 않는다' if m < 2.0 else '귀무 대비 유의 — 분업이 의미를 갖는다'})")
        print("  귀무 = 같은 묶음 크기로 소유 라벨만 섞기 ×"
              f"{SHUFFLES}, 분할 구조는 고정하고 엔진과의 대응만 파괴\n")


if __name__ == "__main__":
    main()
