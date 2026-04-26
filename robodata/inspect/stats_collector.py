"""Statistics collector for Robodata.

Collects dataset statistics during inspection, including frame counts,
value distributions, and coverage metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robodata.core.models import Episode, Frame


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    episode_id: str
    num_frames: int = 0
    has_language: bool = False
    has_success_label: bool = False
    has_rewards: bool = False
    num_cameras: int = 0
    state_dim: int | None = None
    action_dim: int | None = None


@dataclass
class DatasetStats:
    """Aggregated statistics for a dataset."""

    # Counts
    total_episodes: int = 0
    total_frames: int = 0

    # Frame statistics
    min_frames_per_episode: int = 0
    max_frames_per_episode: int = 0
    mean_frames_per_episode: float = 0.0

    # Coverage
    language_coverage: float = 0.0  # Fraction with language
    success_label_coverage: float = 0.0  # Fraction with success labels
    reward_coverage: float = 0.0  # Fraction with rewards

    # Schema consistency
    consistent_state_dim: bool = True
    consistent_action_dim: bool = True
    consistent_cameras: bool = True

    # Value ranges (if collected)
    state_min: list[float] = field(default_factory=list)
    state_max: list[float] = field(default_factory=list)
    action_min: list[float] = field(default_factory=list)
    action_max: list[float] = field(default_factory=list)


class StatsCollector:
    """Collects statistics from episodes during inspection.

    Can operate in streaming mode (updating stats incrementally) or
    batch mode (collecting all data first).
    """

    def __init__(self) -> None:
        self._episode_stats: list[EpisodeStats] = []
        self._state_dims: set[int] = set()
        self._action_dims: set[int] = set()
        self._camera_sets: list[frozenset[str]] = []

    def collect_episode(self, episode: Episode) -> EpisodeStats:
        """Collect statistics from a single episode.

        Args:
            episode: Episode to analyze.

        Returns:
            Statistics for this episode.
        """
        stats = EpisodeStats(episode_id=episode.episode_id)

        # Basic metadata
        stats.has_language = episode.language_instruction is not None
        stats.has_success_label = episode.success is not None
        stats.num_cameras = len(episode.cameras)

        # Count frames and analyze content
        frame_count = 0
        has_reward = False

        for frame in episode.frames():
            frame_count += 1

            if frame.reward is not None:
                has_reward = True

            # Track dimensions from first frame
            if frame_count == 1:
                if frame.state is not None:
                    stats.state_dim = len(frame.state)
                    self._state_dims.add(stats.state_dim)
                if frame.action is not None:
                    stats.action_dim = len(frame.action)
                    self._action_dims.add(stats.action_dim)

        stats.num_frames = frame_count
        stats.has_rewards = has_reward

        # Track camera names for consistency
        if episode.cameras:
            self._camera_sets.append(frozenset(episode.cameras.keys()))

        self._episode_stats.append(stats)
        return stats

    def collect_frame(self, frame: Frame) -> None:
        """Collect statistics from a single frame (streaming mode).

        Args:
            frame: Frame to analyze.
        """
        # This is used for incremental collection without loading all episodes
        if frame.state is not None:
            self._state_dims.add(len(frame.state))
        if frame.action is not None:
            self._action_dims.add(len(frame.action))

    def aggregate(self) -> DatasetStats:
        """Aggregate collected statistics into summary.

        Returns:
            Aggregated dataset statistics.
        """
        stats = DatasetStats()

        if not self._episode_stats:
            return stats

        # Count totals
        stats.total_episodes = len(self._episode_stats)
        frame_counts = [e.num_frames for e in self._episode_stats]
        stats.total_frames = sum(frame_counts)

        # Frame statistics
        if frame_counts:
            stats.min_frames_per_episode = min(frame_counts)
            stats.max_frames_per_episode = max(frame_counts)
            stats.mean_frames_per_episode = sum(frame_counts) / len(frame_counts)

        # Coverage metrics
        with_language = sum(1 for e in self._episode_stats if e.has_language)
        with_success = sum(1 for e in self._episode_stats if e.has_success_label)
        with_rewards = sum(1 for e in self._episode_stats if e.has_rewards)

        stats.language_coverage = with_language / stats.total_episodes
        stats.success_label_coverage = with_success / stats.total_episodes
        stats.reward_coverage = with_rewards / stats.total_episodes

        # Schema consistency
        stats.consistent_state_dim = len(self._state_dims) <= 1
        stats.consistent_action_dim = len(self._action_dims) <= 1
        stats.consistent_cameras = len(set(self._camera_sets)) <= 1

        return stats

    def reset(self) -> None:
        """Reset collector state."""
        self._episode_stats.clear()
        self._state_dims.clear()
        self._action_dims.clear()
        self._camera_sets.clear()

    @property
    def episode_count(self) -> int:
        """Number of episodes collected so far."""
        return len(self._episode_stats)

    @property
    def frame_count(self) -> int:
        """Total frames collected so far."""
        return sum(e.num_frames for e in self._episode_stats)
