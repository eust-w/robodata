"""Main inspector module for Robodata.

Provides the Inspector facade that orchestrates format detection,
schema analysis, and statistics collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from robodata.core.exceptions import FormatDetectionError, InspectionError
from robodata.core.models import DatasetInfo
from robodata.formats.registry import FormatRegistry
from robodata.inspect.schema_analyzer import SchemaAnalyzer
from robodata.inspect.stats_collector import StatsCollector

if TYPE_CHECKING:
    from robodata.core.protocols import FormatReader


@dataclass
class InspectionOptions:
    """Options for inspection behavior.

    Attributes:
        sample_episodes: How many episodes to sample for stats.
        detect_gripper: Try to infer gripper index in state vector.
        detect_fps: Try to infer FPS from timestamps.
        deep_scan: Scan all episodes (slow but accurate).
        max_frames_per_episode: Max frames to load per episode for analysis.
    """

    sample_episodes: int = 5
    detect_gripper: bool = True
    detect_fps: bool = True
    deep_scan: bool = False
    max_frames_per_episode: int = 100


class Inspector:
    """Main entry point for dataset inspection.

    Orchestrates format detection, schema analysis, and stats collection.
    Provides a unified interface for understanding dataset structure
    before conversion.

    Example:
        >>> inspector = Inspector()
        >>> info = inspector.inspect("./my_dataset")
        >>> print(info.cameras)
        >>> print(info.missing_required)
    """

    def __init__(self, options: InspectionOptions | None = None):
        """Initialize inspector with options.

        Args:
            options: Inspection options. Uses defaults if None.
        """
        self.options = options or InspectionOptions()
        self.schema_analyzer = SchemaAnalyzer()
        self.stats_collector = StatsCollector()

    def inspect(
        self,
        path: str | Path,
        format: str | None = None,
    ) -> DatasetInfo:
        """Inspect a dataset and return structured information.

        Args:
            path: Path to dataset.
            format: Format hint (auto-detected if None).

        Returns:
            DatasetInfo with schema, stats, and conversion readiness.

        Raises:
            FormatDetectionError: If format cannot be detected.
            InspectionError: If inspection fails.
        """
        path = Path(path)

        if not path.exists():
            raise InspectionError(path, "Path does not exist")

        # 1. Detect or validate format
        if format is None:
            try:
                format = FormatRegistry.detect_format(path)
            except FormatDetectionError:
                raise

        # 2. Get format-specific reader
        reader = FormatRegistry.get_reader(format)

        # 3. Get base info from reader
        try:
            info = reader.inspect(path)
        except Exception as e:
            raise InspectionError(path, str(e))

        # 4. Enhance with deeper analysis if requested
        if self.options.deep_scan or self.options.detect_gripper or self.options.detect_fps:
            self._analyze_samples(reader, path, info)

        # 5. Determine what's missing for conversion
        info.missing_required = self._check_requirements(info)

        return info

    def _analyze_samples(
        self,
        reader: FormatReader,
        path: Path,
        info: DatasetInfo,
    ) -> None:
        """Sample episodes to infer additional properties.

        Args:
            reader: Format reader to use.
            path: Path to dataset.
            info: DatasetInfo to update.
        """
        self.stats_collector.reset()

        episodes_with_language = 0
        total_episodes = 0

        try:
            episodes = reader.read_episodes(path)

            for i, episode in enumerate(episodes):
                if i >= self.options.sample_episodes and not self.options.deep_scan:
                    break

                total_episodes += 1

                # Collect sample info (don't overwrite if already set by reader)
                if i == 0:
                    if not info.sample_episode_id:
                        info.sample_episode_id = episode.episode_id
                    if not info.sample_language and episode.language_instruction:
                        info.sample_language = episode.language_instruction

                if episode.language_instruction:
                    episodes_with_language += 1

                # Load frames for analysis
                try:
                    frames = []
                    for j, frame in enumerate(episode.frames()):
                        if j >= self.options.max_frames_per_episode:
                            break
                        frames.append(frame)

                    if i == 0:
                        info.sample_num_frames = len(frames)

                    # Analyze frames for patterns
                    if frames:
                        if self.options.detect_gripper and info.inferred_gripper_index is None:
                            gripper_idx = self.schema_analyzer.infer_gripper_index(frames)
                            if gripper_idx is not None:
                                info.inferred_gripper_index = gripper_idx

                        if self.options.detect_fps and info.inferred_fps is None:
                            fps = self.schema_analyzer.infer_fps(frames)
                            if fps is not None:
                                info.inferred_fps = fps

                        # Check for timestamps
                        if any(f.timestamp is not None for f in frames):
                            info.has_timestamps = True

                        # Check for rewards
                        if any(f.reward is not None for f in frames):
                            info.has_rewards = True

                    # Collect episode stats
                    self.stats_collector.collect_episode(episode)

                except Exception:
                    # Skip problematic episodes
                    continue

        except Exception:
            # If we can't read episodes at all (missing dependency), skip sample analysis
            return

        # Update language coverage (only if not already set by reader)
        if total_episodes > 0 and not info.has_language:
            info.language_coverage = episodes_with_language / total_episodes
            info.has_language = episodes_with_language > 0

        # Update counts from deep scan
        if self.options.deep_scan:
            stats = self.stats_collector.aggregate()
            info.num_episodes = stats.total_episodes
            info.total_frames = stats.total_frames

    def _check_requirements(self, info: DatasetInfo) -> list[str]:
        """Determine what's missing for conversion.

        Args:
            info: DatasetInfo to check.

        Returns:
            List of missing required fields.
        """
        missing = []

        # FPS is required for video-based formats
        if not info.has_timestamps and info.inferred_fps is None:
            missing.append("fps")

        # Robot type is often needed for LeRobot metadata
        if info.inferred_robot_type is None:
            missing.append("robot_type")

        return missing

    def quick_inspect(self, path: str | Path) -> DatasetInfo:
        """Quick inspection without deep analysis.

        Args:
            path: Path to dataset.

        Returns:
            Basic DatasetInfo.
        """
        saved_options = self.options
        self.options = InspectionOptions(
            sample_episodes=1,
            detect_gripper=False,
            detect_fps=False,
            deep_scan=False,
        )
        try:
            return self.inspect(path)
        finally:
            self.options = saved_options
