"""LeRobot v3 format writer for Robodata.

Writes datasets in LeRobot v3 format (chunked structure):
    dataset/
    ├── meta/
    │   ├── info.json           # Dataset metadata, features, paths
    │   ├── stats.json          # Statistics for normalization
    │   ├── tasks.parquet       # Task annotations
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet  # Episode metadata
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet  # Frame data (multiple episodes per file)
    └── videos/
        └── observation.images.{camera}/
            └── chunk-000/
                └── file-000.mp4  # Video data
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robodata.core.exceptions import ConversionError, MissingDependencyError
from robodata.core.models import CameraInfo, DatasetInfo, Episode, LazyImage
from robodata.formats.registry import FormatRegistry
from robodata.video.encoder import VideoEncoder, VideoEncoderConfig


def _check_pyarrow() -> None:
    """Check if PyArrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="pyarrow",
            feature="LeRobot v3 format writing",
            install_hint="pip install robodata[lerobot]",
        )


@dataclass
class LeRobotV3WriterConfig:
    """Configuration for LeRobot v3 writer.

    Attributes:
        fps: Frames per second (required).
        robot_type: Robot type identifier (e.g., "franka", "so100").
        video_codec: Video codec for encoding (default: "libx264").
        video_crf: Constant rate factor for video quality (default: 23).
        video_preset: FFmpeg preset for encoding speed (default: "medium").
        chunks_size: Number of episodes per chunk (default: 1000).
        repo_id: Optional HuggingFace repository ID.
        camera_name_mapping: Optional mapping from source to target camera names.
    """

    fps: float = 30.0
    robot_type: str = "unknown"
    video_codec: str = "libx264"
    video_crf: int = 23
    video_preset: str = "medium"
    chunks_size: int = 1000
    repo_id: str | None = None
    camera_name_mapping: dict[str, str] = field(default_factory=dict)


@FormatRegistry.register_writer("lerobot-v3")
class LeRobotV3Writer:
    """Writer for LeRobot v3 format.

    Converts Episode/Frame data to LeRobot v3 format with:
    - Chunked parquet files for state/action data
    - MP4 videos for each camera (chunked)
    - Parquet metadata files

    Example:
        >>> writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=30))
        >>> writer.write_dataset(episodes, Path("./output"))
    """

    def __init__(self, config: LeRobotV3WriterConfig | None = None):
        """Initialize writer with configuration.

        Args:
            config: Writer configuration. Uses defaults if None.
        """
        self.config = config or LeRobotV3WriterConfig()
        self._video_encoder = VideoEncoder(
            VideoEncoderConfig(
                codec=self.config.video_codec,
                crf=self.config.video_crf,
                preset=self.config.video_preset,
            )
        )

        # Track written episodes for metadata
        self._episode_metadata: list[dict[str, Any]] = []
        self._task_metadata: list[dict[str, Any]] = []
        self._tasks_seen: dict[str, int] = {}  # task -> task_index
        self._cameras: dict[str, CameraInfo] = {}
        self._total_frames: int = 0
        self._features: dict[str, dict[str, Any]] = {}

        # Accumulate frames for chunked writing
        self._current_chunk_frames: list[dict[str, Any]] = []
        self._current_chunk_videos: dict[str, list[LazyImage]] = {}
        self._current_chunk_index: int = 0
        self._episodes_in_current_chunk: int = 0

    @property
    def format_name(self) -> str:
        return "lerobot-v3"

    def set_shard_id(self, shard_id: int) -> None:
        """Select the shard to write to.

        For LeRobot v3, we use a convention of "one episode per chunk" during
        streaming/parallel conversion. The shard id maps directly to the chunk index.
        """
        if shard_id < 0:
            raise ValueError(f"Invalid shard_id: {shard_id}")

        # Ensure no buffered data leaks across shards.
        self._current_chunk_frames = []
        self._current_chunk_videos = {}
        self._episodes_in_current_chunk = 0
        self._current_chunk_index = shard_id

    def _map_camera_name(self, source_name: str) -> str:
        """Map source camera name to LeRobot v3 naming convention.

        LeRobot v3 uses 'observation.images.{camera}' naming.

        Args:
            source_name: Original camera name (e.g., "agentview_image").

        Returns:
            LeRobot v3 camera name (e.g., "observation.images.agentview").
        """
        if source_name in self.config.camera_name_mapping:
            return self.config.camera_name_mapping[source_name]

        # Default: strip _image suffix if present
        clean_name = source_name
        if clean_name.endswith("_image"):
            clean_name = clean_name[:-6]
        if clean_name.endswith("_rgb"):
            clean_name = clean_name[:-4]

        return f"observation.images.{clean_name}"

    def _get_chunk_file_indices(self, episode_index: int) -> tuple[int, int]:
        """Get chunk and file indices for an episode.

        Args:
            episode_index: Episode index.

        Returns:
            Tuple of (chunk_index, file_index).
        """
        chunk_index = episode_index // self.config.chunks_size
        file_index = 0  # We put all episodes in a chunk into file-000
        return chunk_index, file_index

    def _flush_chunk(self, output_path: Path) -> None:
        """Write accumulated chunk data to disk.

        Args:
            output_path: Base output directory.
        """
        if not self._current_chunk_frames:
            return

        _check_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        chunk_index = self._current_chunk_index
        file_index = 0

        # Write data parquet
        data_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"file-{file_index:03d}.parquet"

        try:
            table = pa.Table.from_pylist(self._current_chunk_frames)
            pq.write_table(table, data_path)
        except Exception as e:
            raise ConversionError("source", "lerobot-v3", f"Failed to write parquet: {e}")

        # Write videos for each camera
        fps = self.config.fps
        for cam_name, lazy_images in self._current_chunk_videos.items():
            if not lazy_images:
                continue

            video_dir = output_path / "videos" / cam_name / f"chunk-{chunk_index:03d}"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"file-{file_index:03d}.mp4"

            first_img = lazy_images[0]
            try:
                self._video_encoder.encode_frames(
                    iter(lazy_images),
                    video_path,
                    fps=fps,
                    width=first_img.width,
                    height=first_img.height,
                )
            except Exception as e:
                raise ConversionError("source", "lerobot-v3", f"Failed to encode video: {e}")

        # Clear accumulators
        self._current_chunk_frames = []
        self._current_chunk_videos = {}
        self._episodes_in_current_chunk = 0

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Write a single episode to the output directory.

        Args:
            episode: Episode to write.
            output_path: Base output directory.
            episode_index: Optional explicit episode index (auto-increment if None).
            progress_callback: Optional callback(frame_idx, total_frames) for progress.

        Raises:
            ConversionError: If writing fails.
        """
        _check_pyarrow()

        if episode_index is None:
            episode_index = len(self._episode_metadata)

        fps = episode.fps or self.config.fps

        try:
            frame_list = list(episode.frames())
        except Exception as e:
            raise ConversionError("source", "lerobot-v3", f"Failed to read frames: {e}")

        if not frame_list:
            raise ConversionError("source", "lerobot-v3", "Episode has no frames")

        # Determine task
        task = episode.language_instruction or "default"
        if task not in self._tasks_seen:
            task_idx = len(self._tasks_seen)
            self._tasks_seen[task] = task_idx
            self._task_metadata.append({"task_index": task_idx, "task": task})
        task_index = self._tasks_seen[task]

        # Check if we need to start a new chunk
        chunk_index, _ = self._get_chunk_file_indices(episode_index)
        if chunk_index != self._current_chunk_index and self._current_chunk_frames:
            self._flush_chunk(output_path)
            self._current_chunk_index = chunk_index

        # Track video frame index within the chunk's video file
        video_frame_offset = len(self._current_chunk_frames)

        # Process each frame
        for frame_idx, frame in enumerate(frame_list):
            if progress_callback:
                progress_callback(frame_idx, len(frame_list))

            global_index = self._total_frames + frame_idx

            row: dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": frame_idx,
                "index": global_index,
                "task_index": task_index,
                "timestamp": frame.timestamp if frame.timestamp is not None else frame_idx / fps,
            }

            # Add state
            if frame.state is not None:
                row["observation.state"] = frame.state.tolist()
                if "observation.state" not in self._features:
                    self._features["observation.state"] = {
                        "dtype": "float32",
                        "shape": list(frame.state.shape),
                        "names": None,
                    }

            # Add action
            if frame.action is not None:
                row["action"] = frame.action.tolist()
                if "action" not in self._features:
                    self._features["action"] = {
                        "dtype": "float32",
                        "shape": list(frame.action.shape),
                        "names": None,
                    }

            # Collect camera frames for video encoding
            # Note: Video features are NOT stored in parquet - only in the video files
            # The mapping is done via the global 'index' column
            for cam_name, lazy_img in frame.images.items():
                mapped_name = self._map_camera_name(cam_name)

                if mapped_name not in self._current_chunk_videos:
                    self._current_chunk_videos[mapped_name] = []
                self._current_chunk_videos[mapped_name].append(lazy_img)

                # Track camera info
                if cam_name not in self._cameras:
                    self._cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=lazy_img.height,
                        width=lazy_img.width,
                        channels=lazy_img.channels,
                    )

                # Add feature definition (video features go in info.json, not parquet)
                if mapped_name not in self._features:
                    self._features[mapped_name] = {
                        "dtype": "video",
                        "shape": [lazy_img.height, lazy_img.width, lazy_img.channels],
                        "names": ["height", "width", "channel"],
                        "video_info": {
                            "video.fps": float(fps),
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    }

            self._current_chunk_frames.append(row)

        # Update metadata
        self._episode_metadata.append(
            {
                "episode_index": episode_index,
                "length": len(frame_list),
                "task_index": task_index,
            }
        )
        self._total_frames += len(frame_list)
        self._episodes_in_current_chunk += 1

        # Flush after each episode to bound memory usage and show progress
        # This creates one chunk per episode (matching parallel mode behavior)
        self._flush_chunk(output_path)
        self._current_chunk_index += 1

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        """Write full dataset from episode iterator.

        Args:
            episodes: Iterator of episodes to write.
            output_path: Base output directory.
            dataset_info: Optional dataset metadata for additional info.
            progress_callback: Optional callback(episode_idx, episode_id) for progress.
        """
        output_path = Path(output_path)

        # Reset tracking state
        self._episode_metadata = []
        self._task_metadata = []
        self._tasks_seen = {}
        self._cameras = {}
        self._total_frames = 0
        self._features = {}
        self._current_chunk_frames = []
        self._current_chunk_videos = {}
        self._current_chunk_index = 0
        self._episodes_in_current_chunk = 0

        # Override config from dataset_info if provided
        if dataset_info:
            if dataset_info.inferred_fps:
                self.config.fps = dataset_info.inferred_fps
            if dataset_info.inferred_robot_type:
                self.config.robot_type = dataset_info.inferred_robot_type

        # Process each episode
        for episode_idx, episode in enumerate(episodes):
            if progress_callback:
                progress_callback(episode_idx, episode.episode_id)

            self.write_episode(episode, output_path, episode_index=episode_idx)

        # Write metadata (finalize will flush remaining data)
        if dataset_info:
            self.finalize(output_path, dataset_info)
        else:
            # Create minimal dataset info
            minimal_info = DatasetInfo(
                path=output_path,
                format="lerobot-v3",
                num_episodes=len(self._episode_metadata),
                total_frames=self._total_frames,
                inferred_fps=self.config.fps,
                inferred_robot_type=self.config.robot_type,
                cameras=self._cameras,
            )
            self.finalize(output_path, minimal_info)

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write metadata files after all episodes are written.

        Creates:
        - meta/info.json: Dataset configuration and features
        - meta/episodes/chunk-XXX/file-XXX.parquet: Episode metadata
        - meta/tasks.parquet: Task annotations

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
        """
        # Flush any remaining accumulated data
        self._flush_chunk(output_path)

        _check_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta_dir = output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        fps = int(self.config.fps or dataset_info.inferred_fps or 30)
        # Use metadata count if available, otherwise fall back to dataset_info
        # (needed for parallel processing where workers write directly)
        total_episodes = len(self._episode_metadata) if self._episode_metadata else dataset_info.num_episodes
        total_frames = self._total_frames if self._total_frames > 0 else dataset_info.total_frames

        # Add standard features
        self._features["episode_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        self._features["frame_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        self._features["timestamp"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        self._features["index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        self._features["task_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }

        # Calculate number of chunks (each episode is its own chunk)
        num_chunks = total_episodes

        # In parallel mode, infer features from the first data file
        if not self._features or len(self._features) <= 5:  # Only standard features
            first_data_file = output_path / "data" / "chunk-000" / "file-000.parquet"
            if first_data_file.exists():
                sample_table = pq.read_table(first_data_file)
                for col_name in sample_table.column_names:
                    if col_name not in self._features:
                        col = sample_table.column(col_name)
                        # Infer dtype and shape from data
                        first_val = col[0].as_py() if len(col) > 0 else None
                        if isinstance(first_val, list):
                            shape = [len(first_val)]
                            dtype = "float32"
                        elif isinstance(first_val, (int, float)):
                            shape = [1]
                            dtype = "int64" if isinstance(first_val, int) else "float32"
                        else:
                            shape = [1]
                            dtype = "float32"
                        self._features[col_name] = {
                            "dtype": dtype,
                            "shape": shape,
                            "names": None,
                        }

            # Check for video features
            videos_dir = output_path / "videos"
            if videos_dir.exists():
                for cam_dir in videos_dir.iterdir():
                    if cam_dir.is_dir():
                        cam_name = cam_dir.name
                        if cam_name not in self._features:
                            # Try to get video dimensions from first file
                            first_video = cam_dir / "chunk-000" / "file-000.mp4"
                            width, height = 96, 96  # Default
                            if first_video.exists():
                                try:
                                    import av
                                    with av.open(str(first_video)) as container:
                                        stream = container.streams.video[0]
                                        width = stream.width
                                        height = stream.height
                                except Exception:
                                    pass
                            self._features[cam_name] = {
                                "dtype": "video",
                                "shape": [height, width, 3],
                                "names": ["height", "width", "channel"],
                                "video_info": {
                                    "video.fps": float(fps),
                                    "video.codec": "h264",
                                    "video.pix_fmt": "yuv420p",
                                    "video.is_depth_map": False,
                                    "has_audio": False,
                                },
                            }

        # Build info.json
        total_tasks = len(self._task_metadata) if self._task_metadata else 1
        # Each episode is written to its own chunk (memory-efficient streaming)
        effective_chunks_size = 1
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.config.robot_type or dataset_info.inferred_robot_type or "unknown",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "chunks_size": effective_chunks_size,
            "fps": fps,
            "splits": {
                "train": f"0:{total_episodes}",
            },
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": self._features,
        }

        if self.config.repo_id:
            info["repo_id"] = self.config.repo_id

        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Write episodes metadata in chunked parquet format (one episode per chunk)
        if self._episode_metadata:
            # Sequential mode: write one episode metadata file per chunk
            for chunk_idx, ep_meta in enumerate(self._episode_metadata):
                episodes_dir = meta_dir / "episodes" / f"chunk-{chunk_idx:03d}"
                episodes_dir.mkdir(parents=True, exist_ok=True)

                table = pa.Table.from_pylist([ep_meta])
                pq.write_table(table, episodes_dir / "file-000.parquet")
        else:
            # Parallel mode: generate episode metadata
            # Use pre-computed frame counts from dataset_info if available (much faster)
            parallel_frame_counts = dataset_info.metadata.get("_parallel_episode_frame_counts", {})

            for episode_idx in range(total_episodes):
                chunk_idx = episode_idx  # In parallel mode, each episode is its own chunk

                # Use cached frame count if available, otherwise fall back to reading file
                if episode_idx in parallel_frame_counts:
                    length = parallel_frame_counts[episode_idx]
                else:
                    data_file = output_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
                    if data_file.exists():
                        ep_table = pq.read_table(data_file)
                        length = ep_table.num_rows
                    else:
                        length = 0

                episode_meta = {
                    "episode_index": episode_idx,
                    "length": length,
                    "task_index": 0,
                }

                episodes_dir = meta_dir / "episodes" / f"chunk-{chunk_idx:03d}"
                episodes_dir.mkdir(parents=True, exist_ok=True)

                ep_meta_table = pa.Table.from_pylist([episode_meta])
                pq.write_table(ep_meta_table, episodes_dir / "file-000.parquet")

        # Write tasks.parquet
        if self._task_metadata:
            tasks_table = pa.Table.from_pylist(self._task_metadata)
            pq.write_table(tasks_table, meta_dir / "tasks.parquet")
        else:
            # Parallel mode: write default task
            default_task = [{"task_index": 0, "task": "default"}]
            tasks_table = pa.Table.from_pylist(default_task)
            pq.write_table(tasks_table, meta_dir / "tasks.parquet")
