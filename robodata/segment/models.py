"""Data models for segmentation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Segment:
    """A single contiguous segment within an episode."""

    start: int
    end: int
    duration_frames: int
    duration_seconds: float | None = None
    label: str = ""


@dataclass
class EpisodeSegmentation:
    """Segmentation result for a single episode."""

    episode_id: str
    num_frames: int = 0
    signal_name: str = ""
    signal_dim: int = 0
    changepoints: list[int] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    num_segments: int = 0
    fps: float | None = None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "num_frames": self.num_frames,
            "signal_name": self.signal_name,
            "signal_dim": self.signal_dim,
            "changepoints": self.changepoints,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "duration_frames": s.duration_frames,
                    "duration_seconds": s.duration_seconds,
                    "label": s.label,
                }
                for s in self.segments
            ],
            "num_segments": self.num_segments,
            "fps": self.fps,
        }


@dataclass
class SegmentationReport:
    """Segmentation report for an entire dataset."""

    dataset_path: str
    num_episodes: int = 0
    config: dict = field(default_factory=dict)
    computed_at: str = ""
    per_episode: list[EpisodeSegmentation] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    def compute_summary(self) -> None:
        """Compute aggregate summary statistics from per-episode results."""
        if not self.per_episode:
            return
        counts = [ep.num_segments for ep in self.per_episode]
        self.summary = {
            "mean_segments": round(sum(counts) / len(counts), 2),
            "median_segments": sorted(counts)[len(counts) // 2],
            "min_segments": min(counts),
            "max_segments": max(counts),
            "total_changepoints": sum(len(ep.changepoints) for ep in self.per_episode),
        }

        # Add label distribution if any segments are labeled
        from collections import Counter
        label_counts: Counter[str] = Counter()
        for ep in self.per_episode:
            for seg in ep.segments:
                if seg.label:
                    label_counts[seg.label] += 1
        if label_counts:
            self.summary["label_distribution"] = dict(label_counts.most_common())

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "num_episodes": self.num_episodes,
            "config": self.config,
            "computed_at": self.computed_at,
            "summary": self.summary,
            "per_episode": [ep.to_dict() for ep in self.per_episode],
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> SegmentationReport:
        """Load a SegmentationReport from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> SegmentationReport:
        """Reconstruct a SegmentationReport from a dictionary."""
        report = cls(
            dataset_path=data["dataset_path"],
            num_episodes=data.get("num_episodes", 0),
            config=data.get("config", {}),
            computed_at=data.get("computed_at", ""),
        )
        report.summary = data.get("summary", {})

        for ep_data in data.get("per_episode", []):
            segments = [
                Segment(
                    start=s["start"],
                    end=s["end"],
                    duration_frames=s["duration_frames"],
                    duration_seconds=s.get("duration_seconds"),
                    label=s.get("label", ""),
                )
                for s in ep_data.get("segments", [])
            ]
            ep = EpisodeSegmentation(
                episode_id=ep_data["episode_id"],
                num_frames=ep_data.get("num_frames", 0),
                signal_name=ep_data.get("signal_name", ""),
                signal_dim=ep_data.get("signal_dim", 0),
                changepoints=ep_data.get("changepoints", []),
                segments=segments,
                num_segments=ep_data.get("num_segments", 0),
                fps=ep_data.get("fps"),
            )
            report.per_episode.append(ep)

        return report
