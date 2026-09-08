"""Offline reader for the consolidated LeRobot v3 ALOHA simulation dataset.

The pinned LeRobot client in this repository expects one parquet/video file per
episode.  The public ``lerobot/aloha_sim_transfer_cube_human`` dataset is now
published in the newer consolidated-v3 layout instead: three parquet shards
hold all 50 episodes and one video contains all frames.  Mixing the two layouts
causes the upstream client to download a second, per-episode copy into the same
``data/`` directory and breaks its timestamp validation.

This reader keeps the public v3 layout immutable and exposes the small Dataset
protocol consumed by ``openpi.training.data_loader``.  It intentionally only
implements the fields used by the existing ALOHA transforms.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


class AlohaV3Dataset(torch.utils.data.Dataset):
    """Read a consolidated-v3 ALOHA dataset without network access.

    Each item returns a current 14-D state, a 50 Hz camera frame, and a
    future action chunk.  Actions beyond an episode boundary are padded by
    repeating the final in-episode action, matching LeRobot's chunk semantics.
    """

    _REQUIRED_COLUMNS = (
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
    )

    def __init__(self, root: str | Path, *, action_horizon: int, frame_cache_size: int = 32) -> None:
        self.root = Path(root).expanduser().resolve()
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        if frame_cache_size <= 0:
            raise ValueError(f"frame_cache_size must be positive, got {frame_cache_size}.")
        if not self.root.is_dir():
            raise FileNotFoundError(f"ALOHA v3 dataset root does not exist: {self.root}")

        parquet_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                "No consolidated v3 parquet shards found under "
                f"{self.root / 'data'} (expected data/chunk-*/file-*.parquet)."
            )
        video_files = sorted((self.root / "videos" / "observation.images.top").glob("chunk-*/file-*.mp4"))
        if len(video_files) != 1:
            raise FileNotFoundError(
                "Expected exactly one consolidated top-camera video under "
                f"{self.root / 'videos' / 'observation.images.top'}, found {len(video_files)}."
            )

        columns: dict[str, list[Any]] = {key: [] for key in self._REQUIRED_COLUMNS}
        for parquet_path in parquet_files:
            table = pq.read_table(parquet_path, columns=list(self._REQUIRED_COLUMNS))
            data = table.to_pydict()
            for key in self._REQUIRED_COLUMNS:
                columns[key].extend(data[key])

        self._states = np.asarray(columns["observation.state"], dtype=np.float32)
        self._actions = np.asarray(columns["action"], dtype=np.float32)
        self._episode_indices = np.asarray(columns["episode_index"], dtype=np.int64)
        self._frame_indices = np.asarray(columns["frame_index"], dtype=np.int64)
        self._timestamps = np.asarray(columns["timestamp"], dtype=np.float64)
        self._global_indices = np.asarray(columns["index"], dtype=np.int64)
        self._action_horizon = int(action_horizon)
        self._video_path = video_files[0]
        self._frame_cache_size = int(frame_cache_size)
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._video_container: Any | None = None
        self._video_stream: Any | None = None

        self._validate_and_index_episodes()

    def __len__(self) -> int:
        return int(self._states.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_end = int(self._episode_ends[self._episode_positions[index]])
        action_indices = np.minimum(np.arange(index, index + self._action_horizon), episode_end - 1)
        return {
            "observation.images.top": self._decode_frame(int(self._global_indices[index])),
            "observation.state": self._states[index].copy(),
            "action": self._actions[action_indices].copy(),
        }

    def __getstate__(self) -> dict[str, Any]:
        """Make DataLoader's spawn workers reopen PyAV containers lazily."""
        state = self.__dict__.copy()
        state["_video_container"] = None
        state["_video_stream"] = None
        state["_frame_cache"] = OrderedDict()
        return state

    def _validate_and_index_episodes(self) -> None:
        if not (
            self._states.ndim == 2
            and self._actions.ndim == 2
            and self._states.shape == self._actions.shape
            and self._states.shape[1] == 14
        ):
            raise ValueError(
                "Expected ALOHA state/action arrays with shape [N, 14], got "
                f"state={self._states.shape}, action={self._actions.shape}."
            )
        n = len(self)
        if n == 0 or not (
            len(self._episode_indices)
            == len(self._frame_indices)
            == len(self._timestamps)
            == len(self._global_indices)
            == n
        ):
            raise ValueError("ALOHA v3 shard columns have inconsistent or empty lengths.")

        episode_starts = np.r_[0, np.flatnonzero(np.diff(self._episode_indices) != 0) + 1]
        episode_ends = np.r_[episode_starts[1:], n]
        self._episode_ends = episode_ends.astype(np.int64)
        self._episode_positions = np.empty(n, dtype=np.int64)
        for position, (start, end) in enumerate(zip(episode_starts, episode_ends, strict=True)):
            episode_id = self._episode_indices[start]
            if not np.all(self._episode_indices[start:end] == episode_id):
                raise ValueError(f"Episode {episode_id} is not contiguous in consolidated parquet data.")
            expected_frames = np.arange(end - start, dtype=np.int64)
            if not np.array_equal(self._frame_indices[start:end], expected_frames):
                raise ValueError(f"Episode {episode_id} frame_index is not a contiguous 0-based sequence.")
            self._episode_positions[start:end] = position

    def _decode_frame(self, global_index: int) -> np.ndarray:
        cached = self._frame_cache.get(global_index)
        if cached is not None:
            self._frame_cache.move_to_end(global_index)
            return cached.copy()

        try:
            import av
        except ImportError as exc:  # pragma: no cover - dependency is locked by the root project.
            raise RuntimeError("PyAV is required to decode the ALOHA v3 video stream.") from exc

        if self._video_container is None:
            self._video_container = av.open(str(self._video_path))
            self._video_stream = self._video_container.streams.video[0]
        assert self._video_stream is not None

        stream = self._video_stream
        # v3 records this public dataset at 50 Hz. Seek on the stream time base,
        # then select the first decoded frame at/after the target PTS.
        target_seconds = global_index / 50.0
        target_pts = int(target_seconds / float(stream.time_base))
        self._video_container.seek(max(0, target_pts), stream=stream, backward=True, any_frame=False)

        selected = None
        for frame in self._video_container.decode(stream):
            frame_time = float(frame.pts * stream.time_base)
            if frame_time + 1e-6 >= target_seconds:
                selected = frame
                break
        if selected is None:
            raise RuntimeError(f"Unable to decode ALOHA frame {global_index} from {self._video_path}.")

        image_chw = np.ascontiguousarray(selected.to_ndarray(format="rgb24").transpose(2, 0, 1))
        self._frame_cache[global_index] = image_chw
        self._frame_cache.move_to_end(global_index)
        while len(self._frame_cache) > self._frame_cache_size:
            self._frame_cache.popitem(last=False)
        return image_chw.copy()
