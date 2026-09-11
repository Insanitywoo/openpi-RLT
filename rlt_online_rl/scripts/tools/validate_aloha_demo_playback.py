#!/usr/bin/env python3
"""Validate the Gym-ALOHA 14-D action contract with a saved LeRobot demo.

The public Gym-ALOHA action-space metadata uses generic [-1, 1] bounds even
though its simulated joint-position controller expects values outside that
range. This tool replays a stored standard-ALOHA demo through
``AlohaSingleArmChunkEnv.step_full`` and records the native reward/success.

Run it with the root Python 3.11 environment, which contains pyarrow:
    python rlt_online_rl/scripts/tools/validate_aloha_demo_playback.py \
      --dataset-root .../aloha_sim_transfer_cube_human --episode-id 0
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.aloha_sim_env import AlohaSingleArmChunkEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_actions(dataset_root: Path, episode_id: int) -> list[np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific guard.
        raise RuntimeError("pyarrow is required; run this tool in the root OpenPI Python environment.") from exc
    actions: list[tuple[int, np.ndarray]] = []
    for path in sorted((dataset_root / "data" / "chunk-000").glob("*.parquet")):
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        for action, index, frame in zip(
            table["action"].to_pylist(),
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            if int(index) == episode_id:
                actions.append((int(frame), np.asarray(action, dtype=np.float32)))
    if not actions:
        raise ValueError(f"No actions found for episode_id={episode_id} under {dataset_root}")
    actions.sort(key=lambda item: item[0])
    return [action for _, action in actions]


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    actions = _load_actions(dataset_root, args.episode_id)
    seed = args.episode_id if args.seed is None else args.seed
    env = AlohaSingleArmChunkEnv(seed=seed, max_env_steps=len(actions))
    rewards: list[float] = []
    success = 0
    try:
        env.reset()
        for action in actions:
            _, reward, terminated, truncated, info = env.step_full(action)
            rewards.append(float(reward))
            success = max(success, int(info["success"]))
            if terminated or truncated:
                break
    finally:
        env.close()
    result = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "episode_id": args.episode_id,
        "seed": seed,
        "actions_available": len(actions),
        "actions_executed": len(rewards),
        "max_reward": max(rewards, default=0.0),
        "final_reward": rewards[-1] if rewards else 0.0,
        "success": success,
        "max_joint_abs": float(np.max(np.abs(np.stack(actions, axis=0)[:, :6]))),
    }
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["success"] != 1:
        raise RuntimeError("Demo playback did not reach success; action/reset contract is not validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
