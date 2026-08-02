"""Canonical runtime policies shared by ConsciousLM training variants."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
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
