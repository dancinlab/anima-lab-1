import torch

from bench_lineage_age import (
    Observation,
    exact_age_contrasts,
    summarise_seed,
    two_sided_sign_p,
)
from consciousness_engine import ConsciousnessEngine


def test_lineage_depth_is_native_and_survives_removed_ancestor():
    engine = ConsciousnessEngine(
        cell_dim=8,
        hidden_dim=16,
        initial_cells=2,
        max_cells=5,
        phi_ratchet=False,
    )
    founder = engine.cells[0]
    first = engine.split_cell(founder)
    child = next(cell for cell in engine.cells if cell.cell_id == first["child_id"])
    second = engine.split_cell(child)
    grandchild_id = second["child_id"]

    child_idx = engine._find_idx(child.cell_id)
    engine._remove_cell(child_idx)
    grandchild = next(
        state for state in engine.cell_states if state.cell_id == grandchild_id
    )

    assert grandchild.lineage_depth == 2
    assert engine.status()["cells"][-1]["lineage_depth"] == 2


def test_process_exposes_canonical_lineage_metadata():
    engine = ConsciousnessEngine(
        cell_dim=8,
        hidden_dim=16,
        initial_cells=2,
        max_cells=3,
        phi_ratchet=False,
    )
    engine.split_cell(engine.cells[0])

    result = engine.process(torch.zeros(8))

    assert [cell["lineage_depth"] for cell in result["per_cell"]] == [0, 0, 1]
    assert all(cell["age"] >= 0 for cell in result["per_cell"])


def test_exact_age_contrasts_reject_age_and_step_confounding():
    observations = [
        Observation(42, 700, 10, 0, 16, 1.0),
        Observation(42, 700, 10, 1, 16, 2.0),
        Observation(42, 700, 20, 2, 16, 100.0),
        Observation(42, 701, 10, 3, 16, 100.0),
    ]

    contrasts, comparable = exact_age_contrasts(observations)

    assert comparable == 1
    assert contrasts == [(2.0, 1, 1)]
    result = summarise_seed(42, observations)
    assert result.median_per_depth_ratio == 2.0
    assert result.weighted_geomean_per_depth_ratio == 2.0


def test_sign_test_uses_seed_as_the_unit_of_replication():
    assert two_sided_sign_p(5, 8) == 0.7265625
