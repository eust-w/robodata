"""Base classes for format readers and writers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from robodata.core.models import DatasetInfo, Episode, Frame

logger = logging.getLogger(__name__)


@runtime_checkable
class FormatReader(Protocol):
    """Strategy interface for reading a specific format.

    Each format (RLDS, LeRobot, Zarr, etc.) implements this protocol.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier."""
        ...

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Check if this reader can handle the given path.

        Args:
            path: Path to dataset or file.

        Returns:
            True if this format can read the path.
        """
        ...

    @abstractmethod
    def inspect(self, path: Path) -> DatasetInfo:
        """Inspect dataset and return metadata.

        Args:
            path: Path to dataset.

        Returns:
            Dataset information.
        """
        ...

    @abstractmethod
    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Read episodes from dataset.

        Args:
            path: Path to dataset.

        Yields:
            Episode objects.
        """
        ...

    def get_episode_count(self, path: Path) -> int:
        """Get total number of episodes.

        Args:
            path: Path to dataset.

        Returns:
            Number of episodes, or -1 if unknown.
        """
        # Default implementation - count episodes
        try:
            count = 0
            for _ in self.read_episodes(path):
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Could not count episodes: {e}")
            return -1


@runtime_checkable
class IndexableReader(Protocol):
    """Optional capability for efficient random access by episode index.

    Readers implementing this protocol can fetch episode K without scanning
    all previous episodes.
    """

    @abstractmethod
    def get_episode(self, path: Path, episode_index: int) -> Episode:
        """Get a specific episode by index.

        Args:
            path: Path to dataset.
            episode_index: Zero-based episode index.

        Returns:
            Episode object.
        """
        ...

    @abstractmethod
    def get_episode_count(self, path: Path) -> int:
        """Get total number of episodes.

        Args:
            path: Path to dataset.

        Returns:
            Number of episodes.
        """
        ...


@runtime_checkable
class FormatWriter(Protocol):
    """Strategy interface for writing a specific format."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier."""
        ...

    @abstractmethod
    def write_episode(self, episode: Episode, output_path: Path, episode_index: int) -> None:
        """Write a single episode to the output.

        Args:
            episode: Episode data to write.
            output_path: Path to output directory.
            episode_index: Zero-based episode index.
        """
        ...

    @abstractmethod
    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Finalize the output dataset.

        Called after all episodes have been written.

        Args:
            output_path: Path to output directory.
            dataset_info: Dataset metadata.
        """
        ...


class BaseReader(ABC):
    """Abstract base class for format readers with common functionality.

    Provides shared utilities and default implementations for common operations.
    """

    def __init__(self):
        """Initialize the reader."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier."""
        pass

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Check if this reader can handle the given path."""
        pass

    @abstractmethod
    def inspect(self, path: Path) -> DatasetInfo:
        """Inspect dataset and return metadata."""
        pass

    @abstractmethod
    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Read episodes from dataset."""
        pass

    def get_episode_count(self, path: Path) -> int:
        """Get total number of episodes.

        Default implementation counts episodes by iterating.
        Subclasses should override this for more efficient implementations.
        """
        try:
            count = 0
            for _ in self.read_episodes(path):
                count += 1
            return count
        except Exception as e:
            self._logger.warning(f"Could not count episodes: {e}")
            return -1

    def validate_path(self, path: Path) -> None:
        """Validate that the path exists and is accessible.

        Args:
            path: Path to validate.

        Raises:
            FileNotFoundError: If path doesn't exist.
            PermissionError: If path is not accessible.
        """
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"Path is neither a file nor directory: {path}")

    def _log_progress(self, current: int, total: int, operation: str = "processing") -> None:
        """Log progress for long operations.

        Args:
            current: Current progress count.
            total: Total count.
            operation: Description of operation.
        """
        if total > 0 and current % max(1, total // 10) == 0:
            percentage = (current / total) * 100
            self._logger.info(f"{operation.capitalize()}: {current}/{total} ({percentage:.1f}%)")


class BaseWriter(ABC):
    """Abstract base class for format writers with common functionality."""

    def __init__(self):
        """Initialize the writer."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._episodes_written = 0

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier."""
        pass

    @abstractmethod
    def write_episode(self, episode: Episode, output_path: Path, episode_index: int) -> None:
        """Write a single episode to the output."""
        pass

    @abstractmethod
    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        """Finalize the output dataset."""
        pass

    def prepare_output(self, output_path: Path) -> None:
        """Prepare output directory.

        Args:
            output_path: Path to output directory.
        """
        output_path.mkdir(parents=True, exist_ok=True)

    def validate_episode(self, episode: Episode) -> None:
        """Validate episode data before writing.

        Args:
            episode: Episode to validate.

        Raises:
            ValueError: If episode data is invalid.
        """
        if not episode.episode_id:
            raise ValueError("Episode ID is required")
        if not episode.frames():
            raise ValueError("Episode has no frames")

    def _log_progress(self, episode_index: int, total_episodes: int) -> None:
        """Log writing progress.

        Args:
            episode_index: Current episode index.
            total_episodes: Total number of episodes.
        """
        if total_episodes > 0 and episode_index % max(1, total_episodes // 10) == 0:
            percentage = (episode_index / total_episodes) * 100
            self._logger.info(f"Writing episodes: {episode_index}/{total_episodes} ({percentage:.1f}%)")

    @property
    def episodes_written(self) -> int:
        """Get number of episodes written."""
        return self._episodes_written

    def _increment_episode_count(self) -> None:
        """Increment the episode count."""
        self._episodes_written += 1


class CachingReaderMixin:
    """Mixin class for readers that support caching.

    Provides common caching functionality for readers that need to cache
    expensive operations like file indexing or metadata loading.
    """

    def __init__(self, *args, **kwargs):
        """Initialize caching."""
        super().__init__(*args, **kwargs)
        self._cache: dict[str, Any] = {}
        self._cache_enabled = True

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching.

        Args:
            enabled: Whether to enable caching.
        """
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if not self._cache_enabled:
            return None
        return self._cache.get(key)

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if self._cache_enabled:
            self._cache[key] = value


class ValidationMixin:
    """Mixin class for readers/writers that need validation."""

    def validate_dataset_info(self, dataset_info: DatasetInfo) -> None:
        """Validate dataset information.

        Args:
            dataset_info: Dataset info to validate.

        Raises:
            ValueError: If dataset info is invalid.
        """
        if not dataset_info.total_frames and dataset_info.num_episodes == 0:
            raise ValueError("Dataset has no frames or episodes")

    def validate_frame(self, frame: Frame) -> None:
        """Validate frame data.

        Args:
            frame: Frame to validate.

        Raises:
            ValueError: If frame data is invalid.
        """
        if frame.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        if not frame.timestamp and frame.frame_index > 0:
            raise ValueError("Frame timestamp is required for non-zero indices")
