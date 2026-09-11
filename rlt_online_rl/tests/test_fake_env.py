from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlt_online_rl.fake_env import DeterministicChunkEnv


def _plan(*, chunk_len: int = 10, action_dim: int = 7):
    return SimpleNamespace(
        action_chunk=np.full((chunk_len, action_dim), 0.05, dtype=np.float32),
        ref_chunk=np.full((chunk_len, action_dim), 0.04, dtype=np.float32),
        source=2,
        actor_param_version=3,
        start_features=SimpleNamespace(
            z_rl=np.zeros((2048,), dtype=np.float32),
            proprio=np.zeros((7,), dtype=np.float32),
            ref_chunk=np.zeros((chunk_len, action_dim), dtype=np.float32),
        ),
    )


def test_fake_env_is_deterministic_and_emits_env_driver_trace() -> None:
    env_a = DeterministicChunkEnv(max_env_steps=12)
    env_b = DeterministicChunkEnv(max_env_steps=12)
    plan = _plan()

    for env in (env_a, env_b):
        np.testing.assert_array_equal(env.reset()["state"], np.zeros((7,), dtype=np.float32))

    result_a = env_a.execute_chunk(control_hz=50.0, policy_planner=lambda _obs, _step: plan)
    result_b = env_b.execute_chunk(control_hz=50.0, policy_planner=lambda _obs, _step: plan)

    next_a, rewards_a, done_a, info_a = result_a
    next_b, rewards_b, done_b, info_b = result_b
    np.testing.assert_array_equal(next_a["state"], next_b["state"])
    assert rewards_a == rewards_b
    assert done_a is False
    assert done_b is False
    assert len(info_a["step_trace"]) == 10
    assert info_a["step_trace"][0]["source"] == 2
    assert info_a["step_trace"][0]["actor_param_version"] == 3
    np.testing.assert_array_equal(info_a["chunk_start_features"].z_rl, np.zeros((2048,), dtype=np.float32))
    assert info_a["policy_anchor_features"] == [plan.start_features]
    assert info_a["policy_anchor_offsets"] == [0]
    assert info_b["success"] == 0


def test_fake_env_horizon_terminates_inside_chunk() -> None:
    env = DeterministicChunkEnv(max_env_steps=3)
    env.reset()
    _, rewards, done, info = env.execute_chunk(control_hz=0.0, policy_planner=lambda _obs, _step: _plan())
    assert done is True
    assert len(rewards) == 3
    assert info["step_trace"][-1]["done"] is True


def test_fake_env_rejects_wrong_action_shape() -> None:
    env = DeterministicChunkEnv()
    env.reset()
    with pytest.raises(ValueError, match="Expected action shape"):
        env.step(np.zeros((6,), dtype=np.float32))


def test_aloha_full_reference_preserves_passive_arm_and_joint_ranges() -> None:
    from rlt_online_rl.aloha_sim_env import AlohaSingleArmChunkEnv

    env = AlohaSingleArmChunkEnv(seed=0, max_env_steps=3)
    try:
        observation = env.reset()
        full_reference = np.asarray(observation["state"], dtype=np.float32)
        full_reference[2] = 1.16
        active = np.array([0.1, -0.8, 1.2, 0.2, -0.4, 0.1, 0.7], dtype=np.float32)

        merged_reference = env.merge_active_arm_with_full_reference(active, full_reference)
        assert np.allclose(merged_reference[:7], active)
        assert np.allclose(merged_reference[7:], full_reference[7:])
        assert np.isclose(merged_reference[9], 1.16)

        _, _, _, _, info = env.step_full(full_reference)
        assert np.isclose(info["full_action"][2], 1.16)
    finally:
        env.close()
