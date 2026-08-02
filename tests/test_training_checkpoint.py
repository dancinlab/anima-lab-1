"""Regression tests for trajectory-preserving ConsciousLM checkpoints."""

import argparse
import random

import numpy as np
import pytest
import torch

from mitosis import MitosisEngine
from conscious_lm import ConsciousLM
from train_conscious_lm import (
    capture_rng_state,
    evaluate_language_model,
    prepare_corpus_data,
    restore_mitosis_state,
    restore_rng_state,
    restore_scheduler_progress,
    save_checkpoint,
    validate_resume_corpus,
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


def test_separate_validation_corpus_has_content_identity(tmp_path):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_bytes(b"training corpus bytes " * 64)
    validation_path.write_bytes(b"held out validation material " * 64)

    train_data, validation_data, identity = prepare_corpus_data(argparse.Namespace(
        data=str(train_path),
        val_data=str(validation_path),
    ))

    assert bytes(train_data.tolist()) == train_path.read_bytes()
    assert bytes(validation_data.tolist()) == validation_path.read_bytes()
    assert identity["mode"] == "separate"
    assert identity["train_sha256"] != identity["validation_sha256"]


def test_identical_train_and_validation_are_rejected(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"same corpus content " * 64)

    with pytest.raises(RuntimeError, match="identical content"):
        prepare_corpus_data(argparse.Namespace(
            data=str(corpus),
            val_data=str(corpus),
        ))


def test_resume_rejects_silent_corpus_change():
    checkpoint = {"config": {"data_identity": {"source_sha256": "old"}}}
    current = {"source_sha256": "new"}

    with pytest.raises(RuntimeError, match="corpus identity differs"):
        validate_resume_corpus(checkpoint, current)
    assert validate_resume_corpus(checkpoint, current, allow_change=True)


def test_legacy_resume_records_identity_without_claiming_a_change():
    assert not validate_resume_corpus({"config": {}}, {"source_sha256": "new"})
    assert validate_resume_corpus(
        {"config": {}}, {"source_sha256": "new"}, allow_change=True
    )


def test_validation_is_deterministic_and_does_not_consume_rng():
    model = ConsciousLM(
        vocab_size=256,
        d_model=16,
        n_head=4,
        n_layer=1,
        block_size=8,
        dropout=0.37,
    )
    data = torch.arange(128, dtype=torch.long) % 256
    rng = torch.get_rng_state()
    expected_next = torch.rand(1)
    torch.set_rng_state(rng)

    first = evaluate_language_model(
        model, data, batch_size=2, block_size=8,
        device=torch.device("cpu"), evaluation_bytes=40,
    )
    second = evaluate_language_model(
        model, data, batch_size=3, block_size=8,
        device=torch.device("cpu"), evaluation_bytes=40,
    )

    assert first == pytest.approx(second)
    assert torch.rand(1) == pytest.approx(expected_next)
