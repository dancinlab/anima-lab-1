"""Regression tests for the self-organized criticality runtime."""

import numpy as np

from train_conscious_lm import SOCSandpile


def _single_topple_reference(grid: np.ndarray, threshold: int) -> tuple[np.ndarray, int]:
    """Original one-toppling-per-site implementation used as an oracle."""
    result = grid.copy()
    avalanche_size = 0
    while True:
        topples = result >= threshold
        if not topples.any():
            return result, avalanche_size
        avalanche_size += int(topples.sum())
        result[topples] -= threshold
        counts = topples.astype(np.int32)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(counts, dx, axis=0), dy, axis=1)
            if dx == -1:
                shifted[-1, :] = 0
            elif dx == 1:
                shifted[0, :] = 0
            if dy == -1:
                shifted[:, -1] = 0
            elif dy == 1:
                shifted[:, 0] = 0
            result += shifted


def test_bulk_toppling_preserves_stable_state_and_avalanche_size():
    rng = np.random.default_rng(20260802)
    for _ in range(20):
        initial = rng.integers(0, 24, size=(8, 8), dtype=np.int32)
        sandpile = SOCSandpile(grid_size=8, threshold=4)
        sandpile.grid = initial.copy()
        # Include the same seeded grain that drop_sand adds.
        np.random.seed(7)
        x, y = np.random.randint(0, sandpile.grid_size, 2)
        expected_input = initial.copy()
        expected_input[x, y] += 1
        expected_grid, expected_size = _single_topple_reference(expected_input, threshold=4)
        np.random.seed(7)
        actual_size = sandpile.drop_sand()

        np.testing.assert_array_equal(sandpile.grid, expected_grid)
        assert actual_size == expected_size
