from dataclasses import replace
from pathlib import Path

import pytest

from training_runtime import (
    ThroughputGovernor,
    load_cell_growth_policy,
    load_training_run,
    supervise_training,
)


def test_policy_loads_from_ssot():
    policy = load_cell_growth_policy()
    assert policy.window_steps == 20
    assert policy.divisions_per_step == 2


def test_p90_rejects_costly_steps_hidden_by_a_fast_mean():
    policy = load_cell_growth_policy()
    governor = ThroughputGovernor(policy)
    for duration in [0.1] * 17 + [3.0] * 3:
        governor.observe(duration)
    assert not governor.growth_allowed()
    assert governor.quantile_seconds == pytest.approx(3.0)


def test_hysteresis_requires_sustained_recovery():
    policy = replace(load_cell_growth_policy(), window_steps=4)
    governor = ThroughputGovernor(policy)
    for duration in [3.0] * 4:
        governor.observe(duration)
    assert not governor.growth_allowed()

    for duration in [1.7] * 4:
        governor.observe(duration)
    assert not governor.growth_allowed()
    for duration in [1.5] * 4:
        governor.observe(duration)
    assert governor.growth_allowed()


def test_checkpoint_policy_mismatch_is_rejected():
    policy = load_cell_growth_policy()
    governor = ThroughputGovernor(policy)
    state = governor.state_dict()
    state["policy"]["step_budget_seconds"] = 99.0
    with pytest.raises(RuntimeError, match="policy differs"):
        ThroughputGovernor(policy, state)


def _write_supervisor_config(path: Path, *, entrypoint: str = "worker.py") -> None:
    path.write_text(
        f"""
[cell_growth]
unlimited_horizon=2
step_budget_seconds=1.0
window_steps=2
quantile=0.9
recovery_ratio=0.8
divisions_per_step=1
gpu_reserve_mib=0
[supervisor]
stall_seconds=2
poll_seconds=1
terminate_grace_seconds=1
restart_delay_seconds=1
[runs.smoke]
target="gpu"
service="anima-training-smoke"
entrypoint="{entrypoint}"
checkpoint="checkpoints/best.pt"
log="logs/train.log"
args=["--flag"]
environment=["RUN_MODE=test"]
""",
        encoding="utf-8",
    )


def test_training_run_resolves_paths_from_explicit_research_root(tmp_path):
    config = tmp_path / "training.toml"
    root = tmp_path / "remote-root"
    _write_supervisor_config(config)

    run = load_training_run("smoke", config, root=root)

    assert run.root == root.resolve()
    assert run.entrypoint == root.resolve() / "worker.py"
    assert run.checkpoint == root.resolve() / "checkpoints/best.pt"
    assert run.command[-3:] == ("--flag", "--resume", str(run.checkpoint))


def test_supervisor_terminates_a_live_process_without_step_progress(tmp_path):
    config = tmp_path / "training.toml"
    _write_supervisor_config(config)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/best.pt").write_bytes(b"checkpoint")
    (tmp_path / "worker.py").write_text(
        "import time\nprint('loaded', flush=True)\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    run = load_training_run("smoke", config)

    assert supervise_training(run) == 75
    assert (tmp_path / "logs/train.log").read_text(encoding="utf-8") == "loaded\n"


def test_supervisor_accepts_step_progress_and_clean_completion(tmp_path):
    config = tmp_path / "training.toml"
    _write_supervisor_config(config)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/best.pt").write_bytes(b"checkpoint")
    (tmp_path / "worker.py").write_text(
        "print('    48000 | language | 1.0', flush=True)\n",
        encoding="utf-8",
    )
    run = load_training_run("smoke", config)

    assert supervise_training(run) == 0
