from __future__ import annotations

import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from rlt_online_rl.replay import RawEpisodeChunk
from rlt_online_rl.replay import RawEpisodeStep
from rlt_online_rl.replay import RawEpisodeTrace


def test_build_experiment_report_exports_durable_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    episode_dir = run_dir / "replay" / "episodes"
    metrics_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    (metrics_dir / "learner_metrics.jsonl").write_text(
        json.dumps({"global_step": 1, "critic_loss": 2.0, "replay_size": 3}) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "rollout_metrics.jsonl").write_text(
        json.dumps({"episode_id": 7, "success": 1, "fallback_count": 0, "dropped_transitions": 0}) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "training_metrics.jsonl").write_text(
        json.dumps({"global_step": 1, "loss": 0.5, "grad_norm": 0.2, "param_norm": 1.0}) + "\n",
        encoding="utf-8",
    )
    observations = [
        {"images": {"cam_high": np.zeros((3, 8, 10), dtype=np.uint8)}},
        {"images": {"cam_high": np.ones((3, 8, 10), dtype=np.uint8)}},
    ]
    steps = [
        RawEpisodeStep(
            observation_idx=0,
            next_observation_idx=1,
            action=np.ones((7,), dtype=np.float32),
            ref_action=np.zeros((7,), dtype=np.float32),
            reward=1.0,
            done=True,
            success=1,
            episode_id=7,
            step_id=0,
        )
    ]
    trace = RawEpisodeTrace(
        episode_id=7,
        chunk_len=1,
        observations=observations,
        steps=steps,
        chunks=[RawEpisodeChunk(7, 0, 0, 0, 1, 1)],
    )
    with (episode_dir / "episode_000007.pkl").open("wb") as handle:
        pickle.dump(trace, handle)

    report_dir = tmp_path / "report"
    script = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "build_experiment_report.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(run_dir),
            "--output-dir",
            str(report_dir),
            "--skip-plots",
            "--skip-videos",
        ],
        check=True,
    )

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["report_schema_version"] == 1
    assert summary["summary"]["latest_global_step"] == 1
    assert summary["summary"]["raw_episodes"] == 1
    assert summary["summary"]["raw_success_rate"] == 1.0
    assert summary["summary"]["latest_supervised_loss"] == 0.5
    assert (report_dir / "learner_metrics.csv").read_text(encoding="utf-8").startswith("global_step")
    assert (report_dir / "training_metrics.csv").read_text(encoding="utf-8").startswith("global_step")
    episode_csv = (report_dir / "episode_summary.csv").read_text(encoding="utf-8")
    assert "action_deviation_mean" in episode_csv
    assert "2.645" in episode_csv
    assert (report_dir / "experiment_report.md").is_file()
