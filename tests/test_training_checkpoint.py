"""Regression tests for trajectory-preserving ConsciousLM checkpoints."""

import random

import numpy as np
import pytest
import torch

from mitosis import MitosisEngine
from train_conscious_lm import (
    capture_rng_state,
    restore_mitosis_state,
    restore_rng_state,
    restore_scheduler_progress,
    save_checkpoint,
)


class _Phi:
    phi_history = [0.25, 0.5]


def _optimizer_and_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.1, 1.0 - step / 20)
    )
    return parameter, optimizer, scheduler


def test_rng_state_round_trip():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1).item())

    random.random()
    np.random.random()
    torch.rand(1)
    assert restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1).item())

    assert actual == pytest.approx(expected)


def test_scheduler_state_continues_at_next_step():
    _, optimizer, scheduler = _optimizer_and_scheduler()
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    checkpoint = {
        "step": 6,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }

    optimizer.step()
    scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]

    _, resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler()
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_scheduler_progress(resumed_scheduler, checkpoint)
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_legacy_scheduler_keeps_checkpoint_lr_and_progress():
    _, optimizer, scheduler = _optimizer_and_scheduler()
    optimizer.param_groups[0]["lr"] = 0.037
    restore_scheduler_progress(scheduler, {"step": 12})

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.037)
    assert scheduler.last_epoch == 12
    scheduler.step()
    assert scheduler.last_epoch == 13


def test_checkpoint_is_atomic_and_restores_engine_snapshot(tmp_path):
    model = torch.nn.Linear(2, 2)
    loss_ensemble = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_ensemble.parameters()), lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    mitosis = MitosisEngine(input_dim=2, hidden_dim=4, output_dim=2, max_cells=3)
    mitosis.step = 9
    path = tmp_path / "step_9.pt"

    save_checkpoint(
        str(path), 9, model, optimizer, loss_ensemble, mitosis, _Phi(),
        "language", {"steps": 10}, scheduler=scheduler,
        best_val_loss=0.4, skip_count=2,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["next_step"] == 10
    assert checkpoint["best_val_loss"] == pytest.approx(0.4)
    assert checkpoint["skip_count"] == 2
    assert not path.with_suffix(".pt.tmp").exists()

    restored = MitosisEngine(input_dim=2, hidden_dim=4, output_dim=2, max_cells=3)
    assert restore_mitosis_state(restored, checkpoint, torch.device("cpu")) == 2
    assert restored.step == 9
    assert [cell.cell_id for cell in restored.cells] == [0, 1]
