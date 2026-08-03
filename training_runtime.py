"""Canonical runtime policies shared by ConsciousLM training variants.

The module also supervises long GPU runs.  A run is considered healthy only
when it emits a new training step within the configured deadline; this catches
CUDA driver stalls that leave the Python process alive indefinitely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import argparse
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
import tomllib
from typing import Iterable


@dataclass(frozen=True)
class CellGrowthPolicy:
    unlimited_horizon: int
    step_budget_seconds: float
    window_steps: int
    quantile: float
    recovery_ratio: float
    divisions_per_step: int
    gpu_reserve_mib: int


@dataclass(frozen=True)
class SupervisorPolicy:
    stall_seconds: int
    poll_seconds: int
    terminate_grace_seconds: int
    restart_delay_seconds: int


@dataclass(frozen=True)
class TrainingRun:
    name: str
    target: str
    service: str
    root: Path
    entrypoint: Path
    checkpoint: Path
    log: Path
    environment: tuple[str, ...]
    args: tuple[str, ...]
    policy: SupervisorPolicy

    @property
    def command(self) -> tuple[str, ...]:
        return (
            sys.executable,
            "-u",
            str(self.entrypoint),
            *self.args,
            "--resume",
            str(self.checkpoint),
        )


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.@-]+$")
STEP_LINE = re.compile(rb"^\s*(\d+)\s+\|", re.MULTILINE)


def _inside(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay inside the configured research root")
    return root / candidate


def load_training_run(
    name: str,
    path: Path | None = None,
    *,
    root: Path | None = None,
) -> TrainingRun:
    """Load and validate one supervised run from the training SSOT."""
    config_path = path or Path(__file__).with_name("training.toml")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    try:
        values = config["runs"][name]
        policy = SupervisorPolicy(**config["supervisor"])
    except KeyError as error:
        raise ValueError(f"unknown training run {name!r}") from error
    if not SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError("invalid training run name")
    for label in ("target", "service"):
        if not SAFE_IDENTIFIER.fullmatch(values[label]):
            raise ValueError(f"invalid training {label}")
    if min(
        policy.stall_seconds,
        policy.poll_seconds,
        policy.terminate_grace_seconds,
        policy.restart_delay_seconds,
    ) <= 0:
        raise ValueError("supervisor durations must be positive")
    if policy.stall_seconds <= policy.poll_seconds:
        raise ValueError("stall deadline must exceed the polling interval")
    run_root = (root or config_path.resolve().parent).resolve()
    environment = tuple(values.get("environment", ()))
    for assignment in environment:
        key, separator, _value = assignment.partition("=")
        if separator != "=" or not key.isidentifier():
            raise ValueError(f"invalid environment assignment {assignment!r}")
    return TrainingRun(
        name=name,
        target=values["target"],
        service=values["service"],
        root=run_root,
        entrypoint=_inside(run_root, values["entrypoint"], "entrypoint"),
        checkpoint=_inside(run_root, values["checkpoint"], "checkpoint"),
        log=_inside(run_root, values["log"], "log"),
        environment=environment,
        args=tuple(values.get("args", ())),
        policy=policy,
    )


def load_cell_growth_policy(path: Path | None = None) -> CellGrowthPolicy:
    config_path = path or Path(__file__).with_name("training.toml")
    with config_path.open("rb") as handle:
        section = tomllib.load(handle)["cell_growth"]
    policy = CellGrowthPolicy(**section)
    if policy.unlimited_horizon < 2 or policy.divisions_per_step < 1:
        raise ValueError("cell growth counts must be positive")
    if policy.step_budget_seconds <= 0 or policy.window_steps < 2:
        raise ValueError("throughput window and budget must be positive")
    if not 0 < policy.quantile <= 1 or not 0 < policy.recovery_ratio < 1:
        raise ValueError("throughput quantile and recovery ratio are out of range")
    if policy.gpu_reserve_mib < 0:
        raise ValueError("GPU reserve cannot be negative")
    return policy


class ThroughputGovernor:
    """Hysteretic cell-growth gate based on costly-step latency."""

    def __init__(self, policy: CellGrowthPolicy, state: dict | None = None):
        self.policy = policy
        self.durations: deque[float] = deque(maxlen=policy.window_steps)
        self.halted = False
        if state:
            self.load_state_dict(state)

    def observe(self, duration_seconds: float) -> None:
        if math.isfinite(duration_seconds) and duration_seconds >= 0:
            self.durations.append(float(duration_seconds))

    @property
    def quantile_seconds(self) -> float | None:
        if len(self.durations) < self.policy.window_steps:
            return None
        ordered = sorted(self.durations)
        rank = max(0, math.ceil(len(ordered) * self.policy.quantile) - 1)
        return ordered[rank]

    def growth_allowed(self) -> bool:
        measured = self.quantile_seconds
        if measured is None:
            return not self.halted
        threshold = self.policy.step_budget_seconds
        if self.halted:
            threshold *= self.policy.recovery_ratio
        self.halted = measured >= threshold
        return not self.halted

    def state_dict(self) -> dict:
        return {
            "policy": asdict(self.policy),
            "durations": list(self.durations),
            "halted": self.halted,
        }

    def load_state_dict(self, state: dict) -> None:
        saved_policy = state.get("policy")
        if saved_policy and saved_policy != asdict(self.policy):
            raise RuntimeError("checkpoint cell-growth policy differs from training.toml")
        self.durations.clear()
        for duration in state.get("durations", []):
            self.observe(float(duration))
        self.halted = bool(state.get("halted", False))


def gpu_has_headroom(device: object, policy: CellGrowthPolicy) -> bool:
    if getattr(device, "type", str(device)) != "cuda":
        return True
    try:
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes > policy.gpu_reserve_mib * 1024 * 1024
    except Exception:
        return True


def seed_halted_governor(
    policy: CellGrowthPolicy,
    durations: Iterable[float] = (),
) -> ThroughputGovernor:
    """Build a halted governor for explicit recovery of a legacy checkpoint."""
    governor = ThroughputGovernor(policy)
    for duration in durations:
        governor.observe(duration)
    governor.halted = True
    return governor


def _terminate_process_group(process: subprocess.Popen, grace_seconds: int) -> None:
    """Terminate the complete training pipeline and reap it when possible."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            # A kernel-blocked CUDA task cannot be reaped until the driver
            # returns. systemd still owns the cgroup and will finish cleanup.
            pass


def supervise_training(run: TrainingRun) -> int:
    """Run one checkpointed job and fail fast when step progress stalls."""
    if not run.entrypoint.is_file():
        raise FileNotFoundError(f"training entrypoint not found: {run.entrypoint}")
    if not run.checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {run.checkpoint}")
    run.log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for assignment in run.environment:
        key, _separator, value = assignment.partition("=")
        environment[key] = value

    process = subprocess.Popen(
        run.command,
        cwd=run.root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=0,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_progress = time.monotonic()
    latest_step: int | None = None
    pending = b""
    print(f"[supervisor] started {run.name} pid={process.pid}", flush=True)
    with run.log.open("ab", buffering=0) as log:
        while True:
            events = selector.select(timeout=run.policy.poll_seconds)
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                log.write(chunk)
                buffer = pending + chunk
                final_newline = buffer.rfind(b"\n")
                if final_newline < 0:
                    pending = buffer
                    continue
                complete = buffer[:final_newline + 1]
                pending = buffer[final_newline + 1:]
                matches = STEP_LINE.findall(complete)
                if matches:
                    latest_step = int(matches[-1])
                    last_progress = time.monotonic()

            code = process.poll()
            if code is not None:
                if pending:
                    log.write(pending)
                print(
                    f"[supervisor] {run.name} exited code={code} step={latest_step}",
                    flush=True,
                )
                return code
            stalled_for = time.monotonic() - last_progress
            if stalled_for >= run.policy.stall_seconds:
                print(
                    f"[supervisor] {run.name} stalled for {stalled_for:.0f}s "
                    f"after step={latest_step}; terminating for checkpoint resume",
                    flush=True,
                )
                _terminate_process_group(process, run.policy.terminate_grace_seconds)
                return 75


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", metavar="RUN")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if not args.supervise:
        parser.error("--supervise is required")
    run = load_training_run(args.supervise, args.config, root=args.root)
    return supervise_training(run)


if __name__ == "__main__":
    raise SystemExit(main())
