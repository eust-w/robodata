"""Protocol interfaces for Robodata.

This module defines the abstract interfaces (protocols) that format implementations
must satisfy. Using protocols allows for type-safe duck typing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from robodata.core.models import DatasetInfo, Episode, LazyImage


@runtime_checkable
class FormatReader(Protocol):
    """Strategy interface for reading a specific format.

    Each format (RLDS, LeRobot, Zarr, etc.) implements this protocol.
    Readers are responsible for:
    - Detecting if they can read a given path
    - Inspecting dataset structure
    - Lazily iterating over episodes
    """

    @property
    def format_name(self) -> str:
        """Return identifier like 'rlds', 'lerobot-v3'."""
        ...


@runtime_checkable
class ShardedWriter(Protocol):
    """Optional capability for parallel/distributed, shard-safe writes.

    A sharded writer guarantees that after selecting a shard, subsequent writes
    are isolated to that shard (no cross-shard file conflicts).
    """

    def set_shard_id(self, shard_id: int) -> None:
        """Select the shard to write to.

        Args:
            shard_id: Stable shard identifier, typically the episode_index.
        """
        ...


@runtime_checkable
class IndexableReader(Protocol):
    """Optional capability for efficient random access by episode index.

    Readers implementing this protocol can fetch episode K without scanning
    all prior episodes. This enables scalable multi-process and distributed
    conversion on very large datasets.
    """

    def read_episode_by_index(self, path: Path, episode_index: int) -> Episode:
        """Read a specific episode by 0-based index.

        Args:
            path: Path to dataset.
            episode_index: 0-based episode index.

        Returns:
            The requested Episode.

        Raises:
            IndexError: If episode_index is out of range.
        """
        ...

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check if this reader can handle the given path.

        Args:
            path: Path to potential dataset.

        Returns:
            True if this reader can handle the path.
        """
        ...

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect specific version if applicable.

        Args:
            path: Path to dataset.

        Returns:
            Version string or None if not versioned.
        """
        ...

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze dataset structure without full loading.

        Args:
            path: Path to dataset.

        Returns:
            DatasetInfo with schema, stats, and metadata.
        """
        ...

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over episodes.

        Each Episode should support lazy frame loading for memory efficiency.

        Args:
            path: Path to dataset.

        Yields:
            Episode objects.
        """
        ...

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to dataset.
            episode_id: Unique episode identifier.

        Returns:
            The requested Episode.

        Raises:
            KeyError: If episode not found.
        """
        ...


@runtime_checkable
class FormatWriter(Protocol):
    """Strategy interface for writing a specific format.

    Writers are responsible for:
    - Writing individual episodes
    - Managing output directory structure
    - Finalizing metadata after all episodes written
    """

    @property
    def format_name(self) -> str:
        """Return identifier like 'lerobot-v3'."""
        ...

    def write_episode(
        self,
        episode: Episode,
        output_path: Path,
        episode_index: int | None = None,
    ) -> None:
        """Write a single episode to the output location.

        Args:
            episode: Episode to write.
            output_path: Base output directory.
            episode_index: Optional explicit episode index.
        """
        ...

    def write_dataset(
        self,
        episodes: Iterator[Episode],
        output_path: Path,
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        """Write full dataset from episode iterator.

        Handles metadata, directory structure, etc.

        Args:
            episodes: Iterator of episodes to write.
            output_path: Base output directory.
            dataset_info: Optional dataset metadata.
        """
        ...

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Called after all episodes written.

        Write metadata files, indexes, etc.

        Args:
            output_path: Base output directory.
            dataset_info: Dataset metadata.
        """
        ...


class VideoEncoder(Protocol):
    """Interface for video encoding backends."""

    def encode_frames(
        self,
        frames: Iterator[LazyImage],
        output_path: Path,
        fps: float,
        codec: str = "h264",
    ) -> None:
        """Encode frames to video file.

        Args:
            frames: Iterator of frames to encode.
            output_path: Output video file path.
            fps: Frames per second.
            codec: Video codec (default h264).
        """
        ...


class VideoDecoder(Protocol):
    """Interface for video decoding backends."""

    def decode_frames(
        self,
        video_path: Path,
    ) -> Iterator[LazyImage]:
        """Decode video to frames.

        Args:
            video_path: Path to video file.

        Yields:
            LazyImage for each frame.
        """
        ...

    def get_frame(self, video_path: Path, frame_index: int) -> LazyImage:
        """Get a specific frame from video.

        Args:
            video_path: Path to video file.
            frame_index: Zero-based frame index.

        Returns:
            LazyImage for the frame.
        """
        ...

    def get_frame_count(self, video_path: Path) -> int:
        """Get total number of frames in video.

        Args:
            video_path: Path to video file.

        Returns:
            Number of frames.
        """
        ...

    def get_fps(self, video_path: Path) -> float:
        """Get video frame rate.

        Args:
            video_path: Path to video file.

        Returns:
            Frames per second.
        """
        ...
