#!/usr/bin/env python3
"""Fixed-seed reference-only vs Actor-refined Gym-ALOHA evaluation.

This evaluator intentionally bypasses Learner/Replay. It evaluates two frozen
execution policies against the same seed list and records the raw RGB frames,
rewards, actions and VLA reference actions. The output directory is compatible
with ``tools/build_experiment_report.py``.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.aloha_sim_env import AlohaSingleArmChunkEnv
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.inference import ActorClient
from rlt_online_rl.inference import MachineAFeatureClient
from rlt_online_rl.inference import maybe_refine_chunk
from rlt_online_rl.inference import normalize_feature_payload
from rlt_online_rl.replay import RawEpisodeChunk
from rlt_online_rl.replay import RawEpisodeStep
from rlt_online_rl.replay import RawEpisodeTrace
from rlt_online_rl.runtime_logging import append_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("reference_only", "actor_refined"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine-a-ws-url", default="ws://127.0.0.1:18000")
    parser.add_argument("--actor-service-url", default="http://127.0.0.1:9201")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/tasks/aloha_sim/online_rl_single_arm.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-env-steps", type=int, default=300)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--prompt", default="Transfer cube")
    parser.add_argument("--chunk-horizon", type=int, default=10)
    parser.add_argument("--actor-timeout-sec", type=float, default=5.0)
    return parser.parse_args()


def _frame_from_observation(observation: dict[str, Any]) -> np.ndarray:
    image = np.asarray(observation["images"]["cam_high"])
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = np.transpose(image[:3], (1, 2, 0))
    return np.ascontiguousarray(image, dtype=np.uint8)


def _write_config_snapshot(output_dir: Path, args: argparse.Namespace, system: Any) -> None:
    payload = {
        "schema_version": 1,
        "condition": args.condition,
        "machine_a_ws_url": args.machine_a_ws_url,
        "actor_service_url": args.actor_service_url,
        "config": str(args.config.resolve()),
        "episodes": args.episodes,
        "seeds": args.seeds,
        "max_env_steps": args.max_env_steps,
        "arm": args.arm,
        "prompt": args.prompt,
        "chunk_horizon": args.chunk_horizon,
        "started_at_unix": time.time(),
        "rl_config": vars(system.rl),
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _run_episode(
    *,
    condition: str,
    episode_id: int,
    seed: int,
    args: argparse.Namespace,
    system: Any,
    feature_client: MachineAFeatureClient,
    actor_client: ActorClient | None,
    episodes_dir: Path,
) -> dict[str, Any]:
    env = AlohaSingleArmChunkEnv(
        arm=args.arm,
        seed=seed,
        max_env_steps=args.max_env_steps,
        prompt=args.prompt,
    )
    observation = env.reset()
    observations = [observation]
    steps: list[RawEpisodeStep] = []
    chunks: list[RawEpisodeChunk] = []
    frames: list[np.ndarray] = [_frame_from_observation(observation)]
    total_reward = 0.0
    success = 0
    done = False
    env_step = 0
    chunk_id = 0
    actor_version = -1
    actor_versions: set[int] = set()
    action_deviations: list[float] = []
    start = time.time()
    try:
        while not done and env_step < args.max_env_steps:
            payload = normalize_feature_payload(
                feature_client.get_features(observation), system.rl, observation=observation
            )
            ref_chunk = np.asarray(payload["ref_chunk"], dtype=np.float32)
            action_chunk = ref_chunk.copy()
            source = 0
            if condition == "actor_refined":
                if actor_client is None:
                    raise RuntimeError("actor_refined requires actor service")
                refined = maybe_refine_chunk(
                    actor_client,
                    z_rl=np.asarray(payload["z_rl"], dtype=np.float32),
                    proprio=np.asarray(payload["proprio"], dtype=np.float32),
                    ref_chunk=ref_chunk,
                    request_id=f"eval-{condition}-{episode_id}-{chunk_id}",
                    episode_id=episode_id,
                    step_id=env_step,
                    deterministic=True,
                    on_error_fallback=False,
                )
                action_chunk = np.asarray(refined.refined_chunk, dtype=np.float32)
                source = int(refined.source)
                actor_version = int(refined.actor_param_version)
                actor_versions.add(actor_version)
            action_chunk = action_chunk[: args.chunk_horizon, : system.rl.action_dim]
            ref_chunk = ref_chunk[: len(action_chunk), : system.rl.action_dim]
            chunk_start = env_step
            for _local_step, (raw_action, raw_ref_action) in enumerate(zip(action_chunk, ref_chunk, strict=True)):
                action = np.asarray(raw_action, dtype=np.float32)
                ref_action = np.asarray(raw_ref_action, dtype=np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                observation = next_observation
                observations.append(observation)
                frames.append(_frame_from_observation(observation))
                action_deviations.append(float(np.linalg.norm(action - ref_action)))
                steps.append(
                    RawEpisodeStep(
                        observation_idx=len(observations) - 2,
                        next_observation_idx=len(observations) - 1,
                        action=action.copy(),
                        ref_action=ref_action.copy(),
                        reward=float(reward),
                        done=done,
                        source=source,
                        collection_phase="evaluation",
                        success=int(info.get("success", 0)),
                        episode_id=episode_id,
                        step_id=env_step,
                        actor_param_version=actor_version,
                    )
                )
                total_reward += float(reward)
                success = max(success, int(info.get("success", 0)))
                env_step += 1
                if done or env_step >= args.max_env_steps:
                    break
            chunks.append(
                RawEpisodeChunk(
                    episode_id=episode_id,
                    chunk_step_id=chunk_id,
                    observation_idx=chunk_start,
                    step_start=chunk_start,
                    step_stop=env_step,
                    source=source,
                    collection_phase="evaluation",
                    done=done,
                    success=success,
                    start_z_rl=np.asarray(payload["z_rl"], dtype=np.float32),
                    start_proprio=np.asarray(payload["proprio"], dtype=np.float32),
                    start_ref_chunk=np.asarray(ref_chunk, dtype=np.float32),
                )
            )
            chunk_id += 1
    finally:
        env.close()

    trace = RawEpisodeTrace(
        episode_id=episode_id,
        chunk_len=system.rl.chunk_len,
        observations=observations,
        steps=steps,
        chunks=chunks,
        summary={
            "condition": condition,
            "seed": seed,
            "duration_sec": time.time() - start,
            "success": success,
            "reward_sum": total_reward,
            "actor_versions": sorted(actor_versions),
        },
    )
    episode_path = episodes_dir / f"episode_{episode_id:06d}.pkl"
    episode_path.write_bytes(__import__("pickle").dumps(trace, protocol=__import__("pickle").HIGHEST_PROTOCOL))
    return {
        "episode_id": episode_id,
        "seed": seed,
        "condition": condition,
        "success": success,
        "reward_sum": total_reward,
        "raw_step_count": len(steps),
        "raw_chunk_count": len(chunks),
        "duration_sec": time.time() - start,
        "actor_version_min": min(actor_versions) if actor_versions else -1,
        "actor_version_max": max(actor_versions) if actor_versions else -1,
        "actor_version_unique_count": len(actor_versions),
        "action_deviation_mean": float(np.mean(action_deviations)) if action_deviations else 0.0,
        "action_deviation_max": max(action_deviations, default=0.0),
        "raw_episode_path": str(episode_path),
        "video_frame_count": len(frames),
    }


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0 or args.max_env_steps <= 0 or args.chunk_horizon <= 0:
        raise ValueError("episodes, max-env-steps and chunk-horizon must be positive")
    if len(args.seeds) < args.episodes:
        raise ValueError("provide at least as many --seeds as --episodes")
    output_dir = args.output_dir.expanduser().resolve()
    episodes_dir = output_dir / "replay" / "episodes"
    metrics_dir = output_dir / "metrics"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    system = load_system_config_yaml(str(args.config.expanduser().resolve()))
    _write_config_snapshot(output_dir, args, system)
    feature_client = MachineAFeatureClient(args.machine_a_ws_url, recv_timeout_sec=30.0)
    actor_client = (
        ActorClient(args.actor_service_url, timeout_sec=args.actor_timeout_sec, max_retries=0)
        if args.condition == "actor_refined"
        else None
    )
    try:
        for episode_id, seed in enumerate(args.seeds[: args.episodes]):
            record = _run_episode(
                condition=args.condition,
                episode_id=episode_id,
                seed=int(seed),
                args=args,
                system=system,
                feature_client=feature_client,
                actor_client=actor_client,
                episodes_dir=episodes_dir,
            )
            append_jsonl(metrics_dir / "rollout_metrics.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        feature_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
