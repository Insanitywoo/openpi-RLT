#!/usr/bin/env python3
"""Build durable CSV/JSON/Markdown/PNG/MP4 artifacts from an experiment run.

The tool intentionally uses only the standard library plus NumPy, OpenCV and
(optional) Matplotlib. It supports both:

* Online RL run directories containing ``metrics/*.jsonl`` and raw episodes.
* RLT Stage 1 checkpoint directories containing ``metrics/rlt_metrics.jsonl``.

Examples:
    python scripts/tools/build_experiment_report.py runs/online_rl/foo
    python scripts/tools/build_experiment_report.py checkpoints/rlt_pi0_aloha/foo
"""

from __future__ import annotations

# ruff: noqa: RUF001, UP017
import argparse
from collections.abc import Iterable
import csv
import dataclasses
from datetime import datetime
from datetime import timezone
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import numpy as np

ONLINE_RL_SRC = Path(__file__).resolve().parents[2] / "src"
if str(ONLINE_RL_SRC) not in sys.path:
    sys.path.insert(0, str(ONLINE_RL_SRC))

SOURCE_NAMES = {0: "BASE", 1: "RL", 2: "HUMAN", 3: "MIXED"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Online RL run dir or RLT checkpoint experiment dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="Report output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG generation")
    parser.add_argument("--skip-videos", action="store_true", help="Skip MP4 generation")
    parser.add_argument("--video-subsample", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--max-videos", type=int, default=0, help="Maximum videos to export; 0 means all")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                value = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _stringify_cell(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for raw_key in row:
            key = str(raw_key)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        if keys:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _stringify_cell(row.get(key)) for key in keys})
    return len(rows)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_raw_episode(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _camera_frame(observation: Any) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    images = observation.get("images", observation.get("image", {}))
    if not isinstance(images, dict) or not images:
        return None
    image = images.get("cam_high")
    if image is None:
        image = next(iter(images.values()))
    array = np.asarray(image)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array, initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _episode_summary(path: Path, trace: Any) -> dict[str, Any]:
    steps = list(getattr(trace, "steps", []) or [])
    chunks = list(getattr(trace, "chunks", []) or [])
    rewards = [_safe_float(getattr(step, "reward", 0.0)) for step in steps]
    deviations: list[float] = []
    source_counts = dict.fromkeys(SOURCE_NAMES.values(), 0)
    for step in steps:
        source_counts[SOURCE_NAMES.get(int(getattr(step, "source", 1)), "UNKNOWN")] = (
            source_counts.get(SOURCE_NAMES.get(int(getattr(step, "source", 1)), "UNKNOWN"), 0) + 1
        )
        action = np.asarray(getattr(step, "action", []), dtype=np.float32)
        reference = np.asarray(getattr(step, "ref_action", []), dtype=np.float32)
        if action.shape == reference.shape and action.size:
            deviations.append(float(np.linalg.norm(action - reference)))
    summary: dict[str, Any] = {
        "episode_id": int(getattr(trace, "episode_id", -1)),
        "chunk_len": int(getattr(trace, "chunk_len", 0)),
        "raw_step_count": len(steps),
        "raw_chunk_count": len(chunks),
        "frame_count": len(list(getattr(trace, "observations", []) or [])),
        "reward_sum": float(sum(rewards)),
        "reward_mean": _mean(rewards),
        "reward_last": rewards[-1] if rewards else 0.0,
        "success": int(any(int(getattr(step, "success", 0)) for step in steps)),
        "done": int(any(bool(getattr(step, "done", False)) for step in steps)),
        "intervention_count": int(sum(bool(getattr(step, "intervention_flag", False)) for step in steps)),
        "action_deviation_mean": _mean(deviations),
        "action_deviation_p95": _quantile(deviations, 0.95),
        "action_deviation_max": max(deviations, default=0.0),
    }
    summary.update({f"source_{key.lower()}_steps": value for key, value in source_counts.items()})
    raw_summary = getattr(trace, "summary", {})
    if isinstance(raw_summary, dict):
        for key in ("duration_sec", "actor_version_start", "actor_version_end", "collection_phase"):
            if key in raw_summary and key not in summary:
                summary[key] = raw_summary[key]
    return summary


def _make_episode_videos(run_dir: Path, output_dir: Path, *, subsample: int, max_videos: int) -> list[str]:
    try:
        import cv2
    except ImportError as exc:
        print(f"视频导出跳过: 缺少 OpenCV ({exc})", file=sys.stderr)
        return []
    episode_paths = sorted((run_dir / "replay" / "episodes").glob("episode_*.pkl"))
    if max_videos > 0:
        episode_paths = episode_paths[:max_videos]
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for episode_path in episode_paths:
        trace = _load_raw_episode(episode_path)
        observations = list(getattr(trace, "observations", []) or [])
        frames = [
            frame for frame in (_camera_frame(obs) for obs in observations[:: max(1, subsample)]) if frame is not None
        ]
        if not frames:
            continue
        height, width = frames[0].shape[:2]
        output_path = video_dir / f"episode_{int(getattr(trace, 'episode_id', 0)):06d}.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0 / max(1, subsample), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
        try:
            for frame in frames:
                output_frame = frame
                if output_frame.shape[:2] != (height, width):
                    output_frame = cv2.resize(output_frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        exported.append(str(output_path.relative_to(output_dir)))
    return exported


def _import_matplotlib():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_series(
    ax: Any, records: list[dict[str, Any]], key: str, *, label: str | None = None, color: str | None = None
) -> None:
    xs = [_safe_float(row.get("global_step", row.get("step", idx))) for idx, row in enumerate(records)]
    ys = [_safe_float(row.get(key), default=float("nan")) for row in records]
    valid = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(y)]
    if valid:
        ax.plot([x for x, _ in valid], [y for _, y in valid], label=label or key, color=color, linewidth=1.5)


def _save_plots(
    output_dir: Path,
    rlt_metrics: list[dict[str, Any]],
    supervised_metrics: list[dict[str, Any]],
    online_learner: list[dict[str, Any]],
    rollout: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> list[str]:
    try:
        plt = _import_matplotlib()
    except ImportError as exc:
        print(f"PNG 绘图跳过: 缺少 Matplotlib ({exc})", file=sys.stderr)
        return []
    plots: list[str] = []
    if rlt_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for key, label, color in (
            ("loss", "total loss", "tab:blue"),
            ("rlt_loss", "RLT loss", "tab:orange"),
            ("mse", "prefix MSE", "tab:green"),
        ):
            _plot_series(
                axes[0, 0 if key == "loss" else 1 if key == "rlt_loss" else 2],
                rlt_metrics,
                key,
                label=label,
                color=color,
            )
        axes[0, 0].set_title("RLT total loss")
        axes[0, 1].set_title("RLT reconstruction loss")
        axes[0, 2].set_title("Prefix MSE")
        for ax in axes[0]:
            ax.grid(alpha=0.3)
            ax.legend()
        for key, title, color in (
            ("grad_norm", "Gradient norm", "tab:red"),
            ("param_norm", "Parameter norm", "tab:purple"),
            ("vla_loss", "VLA loss", "tab:brown"),
        ):
            ax = axes[1, (0 if key == "grad_norm" else 1 if key == "param_norm" else 2)]
            _plot_series(ax, rlt_metrics, key, label=key, color=color)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
        for ax in axes.flat:
            ax.set_xlabel("global_step")
        path = output_dir / "rlt_training_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plots.append(path.name)
    if supervised_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        for ax, key, title, color in (
            (axes[0], "loss", "Supervised loss", "tab:blue"),
            (axes[1], "grad_norm", "Gradient norm", "tab:red"),
            (axes[2], "param_norm", "Parameter norm", "tab:purple"),
        ):
            _plot_series(ax, supervised_metrics, key, label=key, color=color)
            ax.set_title(title)
            ax.set_xlabel("global_step")
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
        path = output_dir / "supervised_training_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plots.append(path.name)
    if online_learner:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for ax, keys, title in (
            (axes[0, 0], ("critic_loss",), "Critic loss"),
            (axes[0, 1], ("actor_loss",), "Actor loss"),
            (axes[0, 2], ("actor_q", "target_q_mean"), "Actor / target Q"),
            (axes[1, 0], ("bc_penalty", "delta_penalty"), "BC / delta penalty"),
            (axes[1, 1], ("replay_size",), "Replay size"),
            (axes[1, 2], ("actor_version",), "Actor version"),
        ):
            for key in keys:
                _plot_series(ax, online_learner, key, label=key)
            ax.set_title(title)
            ax.set_xlabel("global_step")
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
        path = output_dir / "online_learner_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plots.append(path.name)
    if rollout:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        xs = list(range(len(rollout)))
        for ax, key, title in (
            (axes[0, 0], "success", "Episode success"),
            (axes[0, 1], "duration_sec", "Duration (sec)"),
            (axes[0, 2], "transitions_written", "Transitions written"),
            (axes[1, 0], "fallback_count", "Fallback count"),
            (axes[1, 1], "intervention_count", "Intervention count"),
            (axes[1, 2], "actor_version_end", "Actor version"),
        ):
            ys = [_safe_float(row.get(key)) for row in rollout]
            ax.plot(xs, ys, marker=".", linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("episode index")
            ax.grid(alpha=0.3)
        path = output_dir / "rollout_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plots.append(path.name)
    if episodes and any(row.get("action_deviation_mean", 0.0) for row in episodes):
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        xs = [int(row.get("episode_id", idx)) for idx, row in enumerate(episodes)]
        ax.plot(xs, [row.get("action_deviation_mean", 0.0) for row in episodes], marker="o", label="mean")
        ax.plot(xs, [row.get("action_deviation_p95", 0.0) for row in episodes], marker=".", label="p95")
        ax.set_title("Action deviation from VLA reference")
        ax.set_xlabel("episode_id")
        ax.set_ylabel("L2 norm")
        ax.grid(alpha=0.3)
        ax.legend()
        path = output_dir / "action_deviation.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plots.append(path.name)
    return plots


def _build_markdown(run_dir: Path, output_dir: Path, metadata: dict[str, Any], files: dict[str, Any]) -> None:
    lines = [
        f"# Experiment report: `{run_dir.name}`",
        "",
        f"- Generated at (UTC): `{metadata['generated_at_utc']}`",
        f"- Source directory: `{run_dir}`",
        f"- Git revision: `{metadata.get('git_revision') or 'unknown'}`",
        f"- Report directory: `{output_dir}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in metadata.get("summary", {}).items():
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Artifacts", "", "| Artifact | Description |", "|---|---|"])
    for name, description in files.items():
        if isinstance(description, list):
            lines.extend(f"| `{item}` | {name} |" for item in description)
        else:
            lines.append(f"| `{name}` | {description} |")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- 训练 loss、Q 值和 success 需要结合足够长的曲线与固定种子评估；短 smoke 不能作为收敛结论。",
            "- `action_deviation_*` 衡量 Actor 执行动作相对 VLA `ref_action` 的偏离，不等同于任务成功率。",
            "- MP4 来自 raw episode 中保存的 `cam_high` 图像帧；若没有图像帧，报告会保留表格但跳过视频。",
            "",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {run_dir}")
    if args.video_subsample < 1:
        raise ValueError("--video-subsample must be >= 1")
    output_dir = (
        (args.output_dir or (run_dir / "reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    online_learner = _read_jsonl(run_dir / "metrics" / "learner_metrics.jsonl")
    online_rollout = _read_jsonl(run_dir / "metrics" / "rollout_metrics.jsonl")
    replay_stats = _read_jsonl(run_dir / "metrics" / "replay_stats.jsonl")
    rlt_metrics = _read_jsonl(run_dir / "metrics" / "rlt_metrics.jsonl")
    if not rlt_metrics:
        rlt_metrics = _read_jsonl(run_dir / "rlt_metrics.jsonl")
    supervised_metrics = _read_jsonl(run_dir / "metrics" / "training_metrics.jsonl")
    if not supervised_metrics:
        supervised_metrics = _read_jsonl(run_dir / "training_metrics.jsonl")

    episode_summaries: list[dict[str, Any]] = []
    episode_paths = sorted((run_dir / "replay" / "episodes").glob("episode_*.pkl"))
    for path in episode_paths:
        trace = _load_raw_episode(path)
        episode_summaries.append(_episode_summary(path, trace))
    if online_rollout:
        by_id = {int(row.get("episode_id", -1)): row for row in online_rollout}
        for row in episode_summaries:
            rollout_row = by_id.get(int(row["episode_id"]))
            if rollout_row:
                for key in (
                    "duration_sec",
                    "actor_version_start",
                    "actor_version_end",
                    "fallback_count",
                    "transitions_written",
                    "dropped_transitions",
                    "collection_phase",
                ):
                    if key in rollout_row:
                        row[key] = rollout_row[key]

    counts = {
        "learner_records": len(online_learner),
        "rollout_records": len(online_rollout),
        "replay_stats_records": len(replay_stats),
        "rlt_metric_records": len(rlt_metrics),
        "supervised_metric_records": len(supervised_metrics),
        "raw_episodes": len(episode_summaries),
    }
    summary: dict[str, Any] = dict(counts)
    if online_learner:
        summary.update(
            {
                "latest_global_step": online_learner[-1].get("global_step"),
                "latest_replay_size": online_learner[-1].get("replay_size"),
                "latest_actor_version": online_learner[-1].get("actor_version"),
            }
        )
    if online_rollout:
        summary.update(
            {
                "rollout_success_rate": _mean([_safe_float(row.get("success")) for row in online_rollout]),
                "rollout_fallback_total": sum(_safe_float(row.get("fallback_count")) for row in online_rollout),
                "rollout_dropped_transition_total": sum(
                    _safe_float(row.get("dropped_transitions")) for row in online_rollout
                ),
            }
        )
    if episode_summaries:
        summary.update(
            {
                "raw_success_rate": _mean([_safe_float(row.get("success")) for row in episode_summaries]),
                "raw_reward_mean": _mean([_safe_float(row.get("reward_sum")) for row in episode_summaries]),
                "action_deviation_mean": _mean(
                    [_safe_float(row.get("action_deviation_mean")) for row in episode_summaries]
                ),
            }
        )
    if supervised_metrics:
        latest = supervised_metrics[-1]
        supervised_losses = [_safe_float(row.get("loss"), default=float("nan")) for row in supervised_metrics]
        supervised_losses = [value for value in supervised_losses if math.isfinite(value)]
        summary.update(
            {
                "latest_supervised_step": latest.get("global_step", latest.get("step")),
                "latest_supervised_loss": latest.get("loss"),
                "best_supervised_loss": min(supervised_losses) if supervised_losses else None,
            }
        )
    if rlt_metrics:
        latest = rlt_metrics[-1]
        losses = [_safe_float(row.get("rlt_loss", row.get("loss")), default=float("nan")) for row in rlt_metrics]
        losses = [value for value in losses if math.isfinite(value)]
        window = min(50, max(1, len(losses) // 2))
        recent = losses[-window:]
        previous = losses[-2 * window : -window]
        recent_mean = _mean(recent)
        previous_mean = _mean(previous) if previous else 0.0
        relative_improvement = (previous_mean - recent_mean) / abs(previous_mean) if previous_mean else 0.0
        summary.update(
            {
                "latest_rlt_step": latest.get("global_step", latest.get("step")),
                "latest_rlt_loss": latest.get("rlt_loss", latest.get("loss")),
                "best_rlt_loss": min(losses) if losses else None,
                "rlt_convergence_window_records": window,
                "rlt_recent_loss_mean": recent_mean,
                "rlt_recent_loss_std": float(np.std(recent)) if recent else 0.0,
                "rlt_previous_loss_mean": previous_mean,
                "rlt_relative_improvement": relative_improvement,
            }
        )

    row_counts = {
        "learner_metrics.csv": _write_csv(output_dir / "learner_metrics.csv", online_learner),
        "rollout_metrics.csv": _write_csv(output_dir / "rollout_metrics.csv", online_rollout),
        "replay_stats.csv": _write_csv(output_dir / "replay_stats.csv", replay_stats),
        "rlt_metrics.csv": _write_csv(output_dir / "rlt_metrics.csv", rlt_metrics),
        "training_metrics.csv": _write_csv(output_dir / "training_metrics.csv", supervised_metrics),
        "episode_summary.csv": _write_csv(output_dir / "episode_summary.csv", episode_summaries),
    }
    metadata = {
        "report_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(run_dir),
        "git_revision": _git_revision(),
        "summary": summary,
        "row_counts": row_counts,
        "video_subsample": args.video_subsample,
    }
    _write_json(output_dir / "summary.json", metadata)

    plots = (
        []
        if args.skip_plots
        else _save_plots(output_dir, rlt_metrics, supervised_metrics, online_learner, online_rollout, episode_summaries)
    )
    videos = (
        []
        if args.skip_videos
        else _make_episode_videos(run_dir, output_dir, subsample=args.video_subsample, max_videos=args.max_videos)
    )
    files: dict[str, Any] = {
        "CSV tables": list(row_counts),
        "summary.json": "machine-readable report summary",
        "experiment_report.md": "human-readable report",
        "PNG plots": plots or "not generated",
        "MP4 videos": videos or "not generated",
    }
    _build_markdown(run_dir, output_dir, metadata, files)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "summary": summary, "plots": plots, "videos": videos},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
