"""RoboDM format writer for Robodata.

Writes Episodes to RoboDM .vla format with efficient video compression.
See: https://github.com/BerkeleyAutomation/robodm

Output structure:
    dataset/
    ├── trajectory_000000.vla
    ├── trajectory_000001.vla
    └── ...

Each .vla file contains hierarchical keys:
    - observation/images/{camera_name} (images)
    - observation/state (state vector)
    - action (action vector)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robodata.core.exceptions import ConversionError, MissingDependencyError
from robodata.core.models import DatasetInfo, Episode
from robodata.formats.registry import FormatRegistry


def _check_robodm() -> None:
    """Check if robodm is available."""
    try:
        import robodm  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="robodm",
            feature="RoboDM format writing",
            install_hint="pip install robodm or git clone https://github.com/BerkeleyAutomation/robodm && pip install -e robodm",
        )


@dataclass
class RoboDMWriterConfig:
    """Configuration for RoboDM writer.

    Attributes:
        video_codec: Video codec for compression.
            Options: "libx264" (H.264), "libx265" (H.265), "libaom-av1" (AV1),
                    "ffv1" (lossless), "rawvideo" (uncompressed), "auto"
        codec_options: Additional codec options (e.g., {"crf": "23", "preset": "fast"}).
        fps: Frames per second (default: 30.0).
        camera_name_mapping: Optional mapping from source to target camera names.
    """

    video_codec: str = "auto"
    codec_options: dict[str, str] = field(default_factory=lambda: {"crf": "23", "preset": "medium"})
    fps: float = 30.0
    camera_name_mapping: dict[str, str] = field(default_factory=dict)


@FormatRegistry.register_writer("robodm")
class RoboDMWriter:
    """Writer for RoboDM .vla format.

    Converts Episode/Frame data to RoboDM format with:
    - Efficient video compression (up to 70x with lossy codecs)
    - Hierarchical key structure
    - One .vla file per episode

    Example:
        >>> writer = RoboDMWriter(RoboDMWriterConfig(video_codec="libx265"))
        >>> writer.write_dataset(episodes, Path("./output"))
    """

    def __init__(self, config: RoboDMWriterConfig | None = None):
        """Initialize writer with configuration.

        Args:
            config: Writer configuration. Uses defaults if None.
        """
        self.config = config or RoboDMWriterConfig()
        self._episode_count = 0

    @property
    def format_name(self) -> str:
        return "robodm"

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
    ) -> None:
        """Write a single episode as a .vla file.

        Args:
            episode: Episode to write.
            output_path: Base output directory.
            episode_index: Optional explicit episode index.
        """
        _check_robodm()
        import robodm
        import numpy as np

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        idx = episode_index if episode_index is not None else self._episode_count
        vla_path = output_path / f"trajectory_{idx:06d}.vla"

        # Collect all frame data
        data: dict[str, list[Any]] = {}

        # Initialize lists for each camera
        for cam_name in episode.cameras:
            target_name = self.config.camera_name_mapping.get(cam_name, cam_name)
            key = f"observation/images/{target_name}"
            data[key] = []

        # Initialize state and action lists
        has_state = False
        has_action = False

        # Iterate through frames
        for frame in episode.frames():
            # Collect images
            for cam_name, lazy_img in frame.images.items():
                target_name = self.config.camera_name_mapping.get(cam_name, cam_name)
                key = f"observation/images/{target_name}"
                if key not in data:
                    data[key] = []
                data[key].append(lazy_img.load())

            # Collect state
            if frame.state is not None:
                if "observation/state" not in data:
                    data["observation/state"] = []
                data["observation/state"].append(frame.state)
                has_state = True

            # Collect action
            if frame.action is not None:
                if "action" not in data:
                    data["action"] = []
                data["action"].append(frame.action)
                has_action = True

        # Convert lists to numpy arrays
        for key in data:
            if data[key]:
                data[key] = np.array(data[key])

        # Remove empty keys
        data = {k: v for k, v in data.items() if len(v) > 0}

        if not data:
            raise ConversionError(f"Episode {episode.episode_id} has no data to write")

        # Write using RoboDM
        # Note: We add frames one at a time because:
        # 1. from_dict_of_lists() has a bug that corrupts float32 images
        # 2. add() with batch arrays treats the whole array as one value
        try:
            # Suppress verbose x265 encoder output (writes to fd, not Python stderr)
            import os
            import sys
            stderr_fd = sys.stderr.fileno()
            old_stderr_fd = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)

            trajectory = robodm.Trajectory(
                path=str(vla_path),
                mode="w",
                video_codec=self.config.video_codec,
            )

            # Get number of frames from any key
            num_frames = 0
            for arr in data.values():
                if hasattr(arr, '__len__'):
                    num_frames = len(arr)
                    break

            # Add frames one at a time for proper video encoding
            for frame_idx in range(num_frames):
                for key, arr in data.items():
                    if frame_idx < len(arr):
                        frame_data = arr[frame_idx]
                        # Convert float32 images to uint8 for better codec support
                        if (hasattr(frame_data, 'ndim') and frame_data.ndim == 3
                                and frame_data.shape[2] == 3
                                and frame_data.dtype == np.float32):
                            frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)
                        trajectory.add(key, frame_data)

            trajectory.close()

            # Restore stderr
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)
        except Exception as e:
            # Restore stderr on error
            try:
                os.dup2(old_stderr_fd, stderr_fd)
                os.close(old_stderr_fd)
            except Exception:
                pass
            raise ConversionError(f"Failed to write episode {episode.episode_id}: {e}")

        self._episode_count += 1

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        """Write full dataset from episode iterator.

        Args:
            episodes: Iterator of episodes to write.
            output_path: Base output directory.
            dataset_info: Optional dataset metadata.
        """
        _check_robodm()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._episode_count = 0

        for episode in episodes:
            self.write_episode(episode, output_path)

        # Write metadata file
        if dataset_info is not None:
            self.finalize(output_path, dataset_info)

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Write metadata after all episodes.

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
        """
        import json

        output_path = Path(output_path)

        # Write a simple metadata JSON
        metadata = {
            "format": "robodm",
            "num_episodes": self._episode_count,
            "video_codec": self.config.video_codec,
            "fps": self.config.fps,
        }

        if dataset_info.cameras:
            metadata["cameras"] = {
                name: {"height": cam.height, "width": cam.width, "channels": cam.channels}
                for name, cam in dataset_info.cameras.items()
            }

        # Extract action dim from schema if available
        if dataset_info.action_schema:
            import numpy as np
            metadata["action_dim"] = int(np.prod(dataset_info.action_schema.shape))

        # Extract state dim from observation schema if available
        if dataset_info.observation_schema:
            import numpy as np
            state_dim = 0
            for name, schema in dataset_info.observation_schema.items():
                # Skip image fields
                if "image" not in name.lower() and "camera" not in name.lower():
                    state_dim += int(np.prod(schema.shape))
            if state_dim > 0:
                metadata["state_dim"] = state_dim

        meta_path = output_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
