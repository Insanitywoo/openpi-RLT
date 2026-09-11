#!/usr/bin/env python3
"""Evaluate a standard 14-D ALOHA VLA policy on fixed Gym-ALOHA seeds.

Unlike the Online-RL evaluator, this is a pure VLA reference baseline. It
executes the complete bimanual [T, 14] action chunk returned by
``scripts/serve_policy.py`` and persists episodes in the common report format.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from rlt_online_rl.aloha_sim_env import AlohaSingleArmChunkEnv
from rlt_online_rl.replay import RawEpisodeChunk
from rlt_online_rl.replay import RawEpisodeStep
from rlt_online_rl.replay import RawEpisodeTrace
from rlt_online_rl.runtime_logging import append_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-env-steps", type=int, default=300)
    parser.add_argument("--chunk-horizon", type=int, default=10)
    parser.add_argument("--prompt", default="Transfer cube")
    return parser.parse_args()


def _frame(observation: dict[str, Any]) -> np.ndarray:
    image = np.asarray(observation["images"]["cam_high"], dtype=np.uint8)
    return np.ascontiguousarray(np.transpose(image[:3], (1, 2, 0)))


def _run_episode(
    policy: WebsocketClientPolicy,
    *,
    episode_id: int,
    seed: int,
    args: argparse.Namespace,
    episodes_dir: Path,
) -> dict[str, Any]:
    env = AlohaSingleArmChunkEnv(seed=seed, max_env_steps=args.max_env_steps, prompt=args.prompt)
    observation = env.reset()
    observations = [observation]
    frames = [_frame(observation)]
    steps: list[RawEpisodeStep] = []
    chunks: list[RawEpisodeChunk] = []
    total_reward = 0.0
    done = False
    success = 0
    env_step = 0
    chunk_id = 0
    started = time.time()
    try:
        while not done and env_step < args.max_env_steps:
            result = policy.infer(observation)
            full_chunk = np.asarray(result["actions"], dtype=np.float32)
            if full_chunk.ndim != 2 or full_chunk.shape[1] < 14:
                raise ValueError(f"VLA server returned invalid action chunk {full_chunk.shape}")
            stop = min(len(full_chunk), args.chunk_horizon)
            chunk_start = env_step
            for action in full_chunk[:stop, :14]:
                next_observation, reward, terminated, truncated, info = env.step_full(action)
                done = bool(terminated or truncated)
                observations.append(next_observation)
                frames.append(_frame(next_observation))
                steps.append(
                    RawEpisodeStep(
                        observation_idx=len(observations) - 2,
                        next_observation_idx=len(observations) - 1,
                        action=action.copy(),
                        ref_action=action.copy(),
                        reward=float(reward),
                        done=done,
                        source=0,
                        collection_phase="evaluation",
                        success=int(info["success"]),
                        episode_id=episode_id,
                        step_id=env_step,
                    )
                )
                observation = next_observation
                total_reward += float(reward)
                success = max(success, int(info["success"]))
                env_step += 1
                if done:
                    break
            chunks.append(
                RawEpisodeChunk(
                    episode_id=episode_id,
                    chunk_step_id=chunk_id,
                    observation_idx=chunk_start,
                    step_start=chunk_start,
                    step_stop=env_step,
                    source=0,
                    collection_phase="evaluation",
                    done=done,
                    success=success,
                )
            )
            chunk_id += 1
    finally:
        env.close()
    trace = RawEpisodeTrace(
        episode_id=episode_id,
        chunk_len=args.chunk_horizon,
        observations=observations,
        steps=steps,
        chunks=chunks,
        summary={"condition": "vla_reference", "seed": seed, "duration_sec": time.time() - started},
    )
    path = episodes_dir / f"episode_{episode_id:06d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "episode_id": episode_id,
        "seed": seed,
        "condition": "vla_reference",
        "success": success,
        "reward_sum": total_reward,
        "raw_step_count": len(steps),
        "raw_chunk_count": len(chunks),
        "duration_sec": time.time() - started,
        "raw_episode_path": str(path),
        "video_frame_count": len(frames),
    }


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0 or len(args.seeds) < args.episodes:
        raise ValueError("episodes must be positive and seeds must cover every episode")
    output_dir = args.output_dir.expanduser().resolve()
    episodes_dir = output_dir / "replay" / "episodes"
    metrics_dir = output_dir / "metrics"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    for episode_id, seed in enumerate(args.seeds[: args.episodes]):
        record = _run_episode(policy, episode_id=episode_id, seed=int(seed), args=args, episodes_dir=episodes_dir)
        append_jsonl(metrics_dir / "rollout_metrics.jsonl", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
