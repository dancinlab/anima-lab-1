#!/usr/bin/env python3
"""Separate lineage depth from cell age in the canonical growth runtime.

The earlier audit grouped tension independently by depth and by age. Those two
axes are confounded because a deeper lineage is usually younger. This probe only
compares cells observed in the same seed, engine step, and exact age. Population
state and age are therefore identical inside every comparison stratum.

The runtime owns lineage metadata: ``CellState.lineage_depth`` is assigned when
a child is created and remains valid even if an ancestor is later merged away.

    .venv/bin/python bench_lineage_age.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence

import numpy as np
import torch

from consciousness_engine import ConsciousnessEngine


@dataclass(frozen=True)
class Observation:
    seed: int
    step: int
    age: int
    depth: int
    n_cells: int
    tension: float


@dataclass(frozen=True)
class SeedResult:
    seed: int
    observations: int
    comparable_strata: int
    contrast_count: int
    median_per_depth_ratio: float
    weighted_geomean_per_depth_ratio: float


def collect_seed(
    seed: int,
    *,
    steps: int,
    min_step: int,
    min_cells: int,
    max_cells: int,
    cell_dim: int,
    hidden_dim: int,
) -> list[Observation]:
    """Run the shipping engine and collect instantaneous cell tension."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = ConsciousnessEngine(
        cell_dim=cell_dim,
        hidden_dim=hidden_dim,
        initial_cells=2,
        max_cells=max_cells,
    )

    observations: list[Observation] = []
    for _ in range(steps):
        engine.step(x_input=torch.randn(cell_dim))
        if engine._step < min_step or engine.n_cells < min_cells:
            continue
        for state in engine.cell_states:
            # A daughter is appended after this step's tension calculation.
            # It becomes observable on the next step; never impute its missing
            # birth-step reading as zero.
            if not state.tension_history:
                continue
            observations.append(Observation(
                seed=seed,
                step=engine._step,
                age=engine._step - state.creation_step,
                depth=state.lineage_depth,
                n_cells=engine.n_cells,
                tension=state.tension_history[-1],
            ))
    return observations


def exact_age_contrasts(
    observations: Iterable[Observation],
) -> tuple[list[tuple[float, int, int]], int]:
    """Return high-depth/low-depth tension ratios within exact-age strata.

    Each stratum is one seed, engine step, and exact cell age. For strata with
    multiple depths, the mean tension at every observed higher depth is compared
    with every observed lower depth. The weight is the smaller cell count in the
    pair, preventing a single large clone family from multiplying evidence.
    """
    strata: dict[tuple[int, int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in observations:
        strata[(row.seed, row.step, row.age)][row.depth].append(row.tension)

    contrasts: list[tuple[float, int, int]] = []
    comparable = 0
    for by_depth in strata.values():
        depths = sorted(by_depth)
        if len(depths) < 2:
            continue
        comparable += 1
        for low_index, low_depth in enumerate(depths[:-1]):
            low = by_depth[low_depth]
            low_mean = float(np.mean(low))
            for high_depth in depths[low_index + 1:]:
                high = by_depth[high_depth]
                high_mean = float(np.mean(high))
                if low_mean > 0.0 and high_mean > 0.0:
                    contrasts.append((
                        high_mean / low_mean,
                        high_depth - low_depth,
                        min(len(low), len(high)),
                    ))
    return contrasts, comparable


def summarise_seed(seed: int, observations: Sequence[Observation]) -> SeedResult:
    contrasts, comparable = exact_age_contrasts(observations)
    per_depth_log_ratios = np.asarray([
        np.log(ratio) / depth_gap
        for ratio, depth_gap, _ in contrasts
    ], dtype=float)
    weights = np.asarray([weight for _, _, weight in contrasts], dtype=float)
    return SeedResult(
        seed=seed,
        observations=len(observations),
        comparable_strata=comparable,
        contrast_count=len(contrasts),
        median_per_depth_ratio=(float(np.exp(np.median(per_depth_log_ratios)))
                                if len(contrasts) else float("nan")),
        weighted_geomean_per_depth_ratio=(
            float(np.exp(np.average(per_depth_log_ratios, weights=weights)))
            if len(contrasts) else float("nan")
        ),
    )


def age_depth_table(
    observations: Iterable[Observation], age_bucket: int,
) -> dict[tuple[int, int], tuple[int, float]]:
    """Descriptive bucket table; inference uses exact-age contrasts above."""
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in observations:
        grouped[(row.age // age_bucket, row.depth)].append(row.tension)
    return {
        key: (len(values), float(np.mean(values)))
        for key, values in grouped.items()
    }


def two_sided_sign_p(positive: int, total: int) -> float:
    """Exact two-sided sign-test probability under equal effect directions."""
    if total == 0:
        return float("nan")
    edge = min(positive, total - positive)
    tail = sum(comb(total, value) for value in range(edge + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def format_report(
    results: Sequence[SeedResult],
    observations: Sequence[Observation],
    *,
    age_bucket: int,
) -> str:
    lines = [
        "lineage depth at fixed age",
        "comparison strata: same seed + same engine step + same exact cell age",
        "",
        (f"{'seed':>6} {'observations':>13} {'strata':>9} {'contrasts':>11} "
         f"{'median /depth':>15} {'weighted geo /depth':>20}"),
    ]
    for result in results:
        lines.append(
            f"{result.seed:>6} {result.observations:>13} "
            f"{result.comparable_strata:>9} {result.contrast_count:>11} "
            f"{result.median_per_depth_ratio:>15.4f} "
            f"{result.weighted_geomean_per_depth_ratio:>20.4f}"
        )

    valid = [
        result for result in results
        if np.isfinite(result.weighted_geomean_per_depth_ratio)
    ]
    signs = sum(
        result.weighted_geomean_per_depth_ratio > 1.0 for result in valid
    )
    medians = np.asarray([
        result.median_per_depth_ratio for result in valid
    ], dtype=float)
    weighted = np.asarray([
        result.weighted_geomean_per_depth_ratio for result in valid
    ], dtype=float)
    lines.extend([
        "",
        f"positive seed effects: {signs}/{len(valid)}",
        f"two-sided exact sign p: {two_sided_sign_p(signs, len(valid)):.4f}",
        (f"seed median of per-depth median ratios: {np.median(medians):.4f} "
         f"(range {medians.min():.4f}..{medians.max():.4f})"),
        (f"seed median of weighted geometric ratios: {np.median(weighted):.4f} "
         f"(range {weighted.min():.4f}..{weighted.max():.4f})"),
        "",
        f"descriptive table (age bucket = {age_bucket} steps; count/mean tension)",
    ])

    table = age_depth_table(observations, age_bucket)
    ages = sorted({age for age, _ in table})
    depths = sorted({depth for _, depth in table})
    lines.append("age " + " ".join(f"depth {depth:>2}" for depth in depths))
    for age in ages:
        cells = []
        for depth in depths:
            value = table.get((age, depth))
            cells.append(f"{value[0]:>6}/{value[1]:.6f}" if value else "             -")
        lines.append(f"{age:>3} " + " ".join(cells))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--min-step", type=int, default=600)
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=32)
    parser.add_argument("--cell-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--age-bucket", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_observations: list[Observation] = []
    results: list[SeedResult] = []
    for seed in args.seeds:
        observations = collect_seed(
            seed,
            steps=args.steps,
            min_step=args.min_step,
            min_cells=args.min_cells,
            max_cells=args.max_cells,
            cell_dim=args.cell_dim,
            hidden_dim=args.hidden_dim,
        )
        all_observations.extend(observations)
        results.append(summarise_seed(seed, observations))
        print(f"seed {seed} complete ({len(observations)} observations)", flush=True)

    if not any(result.contrast_count for result in results):
        raise RuntimeError("no exact-age strata contain more than one lineage depth")
    print("\n" + format_report(
        results,
        all_observations,
        age_bucket=args.age_bucket,
    ))


if __name__ == "__main__":
    main()
