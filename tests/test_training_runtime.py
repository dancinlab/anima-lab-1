from dataclasses import replace

import pytest

from training_runtime import ThroughputGovernor, load_cell_growth_policy


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
