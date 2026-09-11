"""Gym-ALOHA rollout adapter with an explicit 7-D single-arm contract.

The public ALOHA model/dataset is bi-manual (14 dimensions).  Online RL for
this project remains a 7-D action editor: this adapter selects one named arm,
passes only that arm to Actor/Replay, and holds the other simulated arm at its
latest joint position.  It never silently truncates a 14-D action.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from rlt_online_rl.inference import PolicyPlan

ArmName = Literal["left", "right"]


class AlohaSingleArmChunkEnv:
    """Adapt the 14-D Gym-ALOHA task to EnvDriver's 7-D chunk protocol."""

    def __init__(
        self,
        *,
        task: str = "gym_aloha/AlohaTransferCube-v0",
        arm: ArmName = "left",
        seed: int = 0,
        max_env_steps: int = 400,
        prompt: str = "Transfer cube",
    ) -> None:
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}.")
        if max_env_steps <= 0:
            raise ValueError("max_env_steps must be positive.")
        try:
            import gym_aloha  # noqa: F401
            import gymnasium
        except ImportError as exc:  # pragma: no cover - guarded by project dependency.
            raise RuntimeError(
                "Gym-ALOHA dependencies are missing; sync rlt_online_rl before using this adapter."
            ) from exc

        self._gym = gymnasium.make(task, obs_type="pixels_agent_pos")
        self._arm = arm
        self._arm_slice = slice(0, 7) if arm == "left" else slice(7, 14)
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._max_env_steps = int(max_env_steps)
        self._prompt = str(prompt)
        self._last_raw_obs: dict[str, Any] | None = None
        self._passive_arm_action = np.zeros((7,), dtype=np.float32)
        self._env_steps = 0

    def close(self) -> None:
        self._gym.close()

    def current_phase_name(self) -> str:
        return "online:aloha_single_arm_sim"

    def reset(self) -> dict[str, Any]:
        raw_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_raw_obs = raw_obs
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        self._passive_arm_action = state[7:14].copy() if self._arm == "left" else state[0:7].copy()
        self._env_steps = 0
        return self._convert_observation(raw_obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._last_raw_obs is None:
            raise RuntimeError("Call reset() before step().")
        single_arm = np.asarray(action, dtype=np.float32).reshape(-1)
        if single_arm.shape != (7,):
            raise ValueError(f"Expected a 7-D {self._arm}-arm action, got {single_arm.shape}.")
        return self.step_full(self._merge_single_arm_action(single_arm))

    def step_full(self, full_action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step with an explicit standard-ALOHA 14-D absolute joint action.

        ``gym_aloha`` advertises generic ``[-1, 1]`` Box bounds even though its
        joint-position task contract legitimately contains values such as
        elbow=1.16. Clipping to that Box silently corrupts replayed demos and
        VLA reference actions, so this adapter validates shape/finite values
        but deliberately preserves the original absolute controls.
        """
        if self._last_raw_obs is None:
            raise RuntimeError("Call reset() before step_full().")
        full_action = np.asarray(full_action, dtype=np.float32).reshape(-1)
        if full_action.shape != (14,):
            raise ValueError(f"Expected a full 14-D ALOHA action, got {full_action.shape}.")
        if not np.all(np.isfinite(full_action)):
            raise ValueError("Full ALOHA action must contain only finite values.")
        raw_obs, reward, terminated, truncated, info = self._gym.step(full_action)
        self._last_raw_obs = raw_obs
        self._env_steps += 1
        truncated = bool(truncated or self._env_steps >= self._max_env_steps)
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        self._passive_arm_action = state[7:14].copy() if self._arm == "left" else state[0:7].copy()
        info = dict(info)
        info["success"] = int(bool(info.get("is_success", False)))
        info["active_arm"] = self._arm
        info["full_action"] = full_action.copy()
        return self._convert_observation(raw_obs), float(reward), bool(terminated), truncated, info

    def execute_chunk(
        self,
        *,
        control_hz: float,
        policy_planner: Callable[[dict[str, Any], int], PolicyPlan],
    ) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        del control_hz  # Gym simulation advances deterministically; no wall-clock sleeps.
        if self._last_raw_obs is None:
            raise RuntimeError("Call reset() before execute_chunk().")
        start_observation = self._convert_observation(self._last_raw_obs)
        plan = policy_planner(start_observation, 0)
        action_chunk = np.asarray(plan.action_chunk, dtype=np.float32)
        ref_chunk = np.asarray(plan.ref_chunk, dtype=np.float32)
        full_ref_chunk = None if plan.full_ref_chunk is None else np.asarray(plan.full_ref_chunk, dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
            raise ValueError(f"action_chunk must be [T, >=7], got {action_chunk.shape}.")
        if ref_chunk.ndim != 2 or ref_chunk.shape[1] < 7:
            raise ValueError(f"ref_chunk must be [T, >=7], got {ref_chunk.shape}.")
        if full_ref_chunk is not None and (full_ref_chunk.ndim != 2 or full_ref_chunk.shape[1] < 14):
            raise ValueError(f"full_ref_chunk must be [T, >=14], got {full_ref_chunk.shape}.")

        rewards: list[float] = []
        trace: list[dict[str, Any]] = []
        done = False
        success = 0
        next_observation = start_observation
        for local_step in range(min(len(action_chunk), len(ref_chunk))):
            observation = next_observation
            action = action_chunk[local_step, :7]
            ref_action = ref_chunk[local_step, :7]
            if full_ref_chunk is None:
                # Backward-compatible fallback for environments that intentionally
                # hold the passive arm. Transfer-cube evaluation always supplies
                # a full VLA chunk.
                full_action = self._merge_single_arm_action(action)
            else:
                full_action = self.merge_active_arm_with_full_reference(action, full_ref_chunk[local_step])
            next_observation, reward, terminated, truncated, info = self.step_full(full_action)
            done = bool(terminated or truncated)
            success = int(info.get("success", 0))
            rewards.append(float(reward))
            trace.append(
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

        return (
            next_observation,
            rewards,
            done,
            {
                "success": success,
                "source": int(plan.source),
                "step_trace": trace,
                "chunk_start_features": plan.start_features,
                "policy_anchor_offsets": [0],
                "policy_anchor_features": [plan.start_features],
                "active_arm": self._arm,
            },
        )

    def merge_active_arm_with_full_reference(self, action: np.ndarray, full_reference: np.ndarray) -> np.ndarray:
        """Replace only the active arm in a standard 14-D VLA reference."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        full_reference = np.asarray(full_reference, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"Expected a 7-D {self._arm}-arm action, got {action.shape}.")
        if full_reference.shape != (14,):
            raise ValueError(f"Expected a 14-D VLA reference action, got {full_reference.shape}.")
        full_action = full_reference.copy()
        full_action[self._arm_slice] = action
        return full_action

    def _merge_single_arm_action(self, action: np.ndarray) -> np.ndarray:
        full = np.empty((14,), dtype=np.float32)
        if self._arm == "left":
            full[:7] = action
            full[7:14] = self._passive_arm_action
        else:
            full[:7] = self._passive_arm_action
            full[7:14] = action
        return full

    def _convert_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(raw_obs["pixels"]["top"], dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected top image [H, W, 3], got {image.shape}.")
        state14 = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        if state14.shape != (14,):
            raise ValueError(f"Expected Gym-ALOHA 14-D state, got {state14.shape}.")
        return {
            "images": {"cam_high": np.ascontiguousarray(image.transpose(2, 0, 1))},
            # Machine A's Pi0/ALOHA preprocessing requires the complete
            # 14-D bimanual proprio vector. EnvDriver deliberately consumes
            # only the first 7 dimensions as the online single-arm contract;
            # the active-arm action mapping remains explicit in _merge...().
            "state": state14.copy(),
            "prompt": self._prompt,
        }
