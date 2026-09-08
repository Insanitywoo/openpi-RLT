"""Deterministic, hardware-free environment for Online RL integration tests.

This module intentionally has no ROS, simulator, or robot dependencies.  It
implements the chunk-execution protocol consumed by :class:`EnvDriver` so the
fake Machine A -> Actor -> Replay -> Learner path can be exercised unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rlt_online_rl.inference import PolicyPlan


class DeterministicChunkEnv:
    """Small absolute-action environment with deterministic finite episodes.

    The state is seven-dimensional by default, matching the initial Online RL
    contract. Each action is interpreted as an absolute target state, clipped
    to a fixed safety range. Episodes end after ``max_env_steps`` execution
    steps; success is reported only when the terminal state reaches the fixed
    goal tolerance. The fixed horizon guarantees replay transitions even when
    an untrained Actor does not reach the goal.
    """

    def __init__(
        self,
        *,
        state_dim: int = 7,
        action_dim: int = 7,
        max_env_steps: int = 20,
        action_limit: float = 0.25,
        success_tolerance: float = 0.03,
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive.")
        if action_dim > state_dim:
            raise ValueError("action_dim must not exceed state_dim.")
        if max_env_steps <= 0:
            raise ValueError("max_env_steps must be positive.")
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._max_env_steps = int(max_env_steps)
        self._action_limit = float(action_limit)
        self._success_tolerance = float(success_tolerance)
        self._goal = np.linspace(0.03, 0.09, self._action_dim, dtype=np.float32)
        self._state = np.zeros((self._state_dim,), dtype=np.float32)
        self._env_step = 0

    def current_phase_name(self) -> str:
        """Expose Online RL collection phase without a human/ROS controller."""
        return "online:deterministic_fake"

    def reset(self) -> dict[str, np.ndarray]:
        self._state.fill(0.0)
        self._env_step = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (self._action_dim,):
            raise ValueError(f"Expected action shape {(self._action_dim,)}, got {action_array.shape}.")
        self._state[: self._action_dim] = np.clip(action_array, -self._action_limit, self._action_limit)
        self._env_step += 1
        distance = float(np.linalg.norm(self._state[: self._action_dim] - self._goal))
        terminated = self._env_step >= self._max_env_steps
        success = int(terminated and distance <= self._success_tolerance)
        reward = float(-distance)
        return self._observation(), reward, terminated, False, {"success": success, "distance": distance}

    def execute_chunk(
        self,
        *,
        control_hz: float,
        policy_planner: Callable[[dict[str, Any], int], PolicyPlan],
    ) -> tuple[dict[str, np.ndarray], list[float], bool, dict[str, Any]]:
        """Execute one planned chunk and emit an EnvDriver-compatible step trace."""
        del control_hz  # The fake environment never sleeps; runs are deterministic and fast.
        start_observation = self._observation()
        plan = policy_planner(start_observation, 0)
        action_chunk = np.asarray(plan.action_chunk, dtype=np.float32)
        ref_chunk = np.asarray(plan.ref_chunk, dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[1] < self._action_dim:
            raise ValueError(f"action_chunk must have shape [T, >= {self._action_dim}], got {action_chunk.shape}.")
        if ref_chunk.ndim != 2 or ref_chunk.shape[1] < self._action_dim:
            raise ValueError(f"ref_chunk must have shape [T, >= {self._action_dim}], got {ref_chunk.shape}.")

        rewards: list[float] = []
        step_trace: list[dict[str, Any]] = []
        next_observation = start_observation
        done = False
        success = 0
        horizon = min(action_chunk.shape[0], ref_chunk.shape[0])
        for local_step in range(horizon):
            observation = next_observation
            action = action_chunk[local_step, : self._action_dim]
            ref_action = ref_chunk[local_step, : self._action_dim]
            next_observation, reward, terminated, truncated, env_info = self.step(action)
            done = bool(terminated or truncated)
            success = int(env_info.get("success", 0))
            rewards.append(float(reward))
            step_trace.append(
                {
                    "observation": observation,
                    "action": action.copy(),
                    "ref_action": ref_action.copy(),
                    "reward": float(reward),
                    "next_observation": next_observation,
                    "source": int(plan.source),
                    "actor_param_version": int(plan.actor_param_version),
                    "done": done,
                    "human_controlled": False,
                }
            )
            if done:
                break

        return next_observation, rewards, done, {
            "success": success,
            "source": int(plan.source),
            "step_trace": step_trace,
            "chunk_start_features": {
                "z_rl": np.asarray(plan.start_features.z_rl, dtype=np.float32),
                "proprio": np.asarray(plan.start_features.proprio, dtype=np.float32),
                "ref_chunk": np.asarray(plan.start_features.ref_chunk, dtype=np.float32),
            },
            "policy_anchor_offsets": [0],
            "policy_anchor_features": [
                {
                    "z_rl": np.asarray(plan.start_features.z_rl, dtype=np.float32),
                    "proprio": np.asarray(plan.start_features.proprio, dtype=np.float32),
                    "ref_chunk": np.asarray(plan.start_features.ref_chunk, dtype=np.float32),
                }
            ],
        }

    def _observation(self) -> dict[str, np.ndarray]:
        return {"state": self._state.copy()}
