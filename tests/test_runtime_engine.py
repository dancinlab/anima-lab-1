import torch

from anima_unified import AnimaUnified
from bench_v2 import ENGINE_REGISTRY
from mitosis import MitosisEngine
from pairfield_engine import PairFieldEngine


def test_gate_registry_uses_real_mitosis_engine():
    adapter = ENGINE_REGISTRY["MitosisEngine"](4, 8, 16)

    assert isinstance(adapter.engine, MitosisEngine)


def test_runtime_factory_uses_gate_qualified_pairfield():
    engine = AnimaUnified._create_runtime_engine(4, 8, 16)

    assert isinstance(engine, PairFieldEngine)
    assert engine.max_cells == 4
    result = engine.process_runtime(torch.randn(1, 8))
    assert result["n_cells"] == 4
    assert len(result["per_cell"]) == 4


def test_pairfield_cell_view_updates_live_state():
    engine = PairFieldEngine(4, 8, 16, 8)
    replacement = torch.ones(1, 16)
    field_before = engine.A.hiddens[2] - engine.G.hiddens[2]

    engine.cells[2].hidden = replacement

    assert torch.equal(engine.A.hiddens[2], replacement.squeeze(0))
    assert torch.allclose(engine.A.hiddens[2] - engine.G.hiddens[2], field_before)


def test_pairfield_declares_ownership_of_runtime_dynamics():
    engine = PairFieldEngine(4, 8, 16, 8)

    assert engine.manages_cell_diversity
    assert engine.manages_cell_dynamics
    assert not engine.supports_population_growth


def test_mitosis_snapshot_restores_complete_runtime_state():
    engine = MitosisEngine(8, 16, 8, initial_cells=4, max_cells=4)
    handle = engine.snapshot()
    engine.process(torch.randn(1, 8))

    engine.restore(handle)

    assert engine.step == 0
    assert all(not cell.tension_history for cell in engine.cells)
