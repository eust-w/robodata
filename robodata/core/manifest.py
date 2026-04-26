"""Dataset manifest utilities.

This module provides a stable, append-only JSON schema for recording per-episode
conversion outcomes. The schema is intended to be stable during 0.x and only
extended with new optional fields.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_MANIFEST_FILENAME = "robodata_manifest.json"
_SCHEMA_VERSION = 1


def get_default_manifest_path(output_path: Path) -> Path:
    return output_path / _MANIFEST_FILENAME


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text content to a file.
    
    Creates a temporary file and atomically replaces the target.
    
    Args:
        path: Target file path.
        content: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content)
    os.replace(temp_path, path)


@dataclass
class EpisodeManifestEntry:
    episode_index: int
    status: str  # success|failed|skipped
    episode_id: str | None = None
    num_frames: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    updated_at_unix: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "status": self.status,
            "episode_id": self.episode_id,
            "num_frames": self.num_frames,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "updated_at_unix": self.updated_at_unix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeManifestEntry":
        return cls(
            episode_index=int(data["episode_index"]),
            status=str(data["status"]),
            episode_id=data.get("episode_id"),
            num_frames=data.get("num_frames"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            updated_at_unix=float(data.get("updated_at_unix", time.time())),
        )


@dataclass
class DatasetManifest:
    schema_version: int = _SCHEMA_VERSION
    created_at_unix: float = field(default_factory=lambda: time.time())
    updated_at_unix: float = field(default_factory=lambda: time.time())

    source_path: str | None = None
    source_format: str | None = None
    target_format: str | None = None

    episodes: dict[int, EpisodeManifestEntry] = field(default_factory=dict)

    def mark_success(self, episode_index: int, episode_id: str | None, num_frames: int | None) -> None:
        self.episodes[int(episode_index)] = EpisodeManifestEntry(
            episode_index=int(episode_index),
            status="success",
            episode_id=episode_id,
            num_frames=num_frames,
        )
        self.updated_at_unix = time.time()

    def mark_failed(self, episode_index: int, episode_id: str | None, error: Exception | str) -> None:
        if isinstance(error, Exception):
            error_type = type(error).__name__
            error_message = str(error)
        else:
            error_type = "Error"
            error_message = str(error)

        self.episodes[int(episode_index)] = EpisodeManifestEntry(
            episode_index=int(episode_index),
            status="failed",
            episode_id=episode_id,
            error_type=error_type,
            error_message=error_message,
        )
        self.updated_at_unix = time.time()

    def mark_skipped(self, episode_index: int) -> None:
        existing = self.episodes.get(int(episode_index))
        if existing and existing.status == "success":
            return
        self.episodes[int(episode_index)] = EpisodeManifestEntry(
            episode_index=int(episode_index),
            status="skipped",
            episode_id=existing.episode_id if existing else None,
            num_frames=existing.num_frames if existing else None,
        )
        self.updated_at_unix = time.time()

    def is_success(self, episode_index: int) -> bool:
        entry = self.episodes.get(int(episode_index))
        return bool(entry and entry.status == "success")

    def get_success_frame_count(self, episode_index: int) -> int | None:
        entry = self.episodes.get(int(episode_index))
        if not entry or entry.status != "success":
            return None
        return entry.num_frames

    def success_count(self) -> int:
        return sum(1 for e in self.episodes.values() if e.status == "success")

    def total_frames_success(self) -> int:
        total = 0
        for e in self.episodes.values():
            if e.status == "success" and isinstance(e.num_frames, int):
                total += e.num_frames
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_unix": self.created_at_unix,
            "updated_at_unix": self.updated_at_unix,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "episodes": {str(k): v.to_dict() for k, v in self.episodes.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    def merge_from(self, other: "DatasetManifest") -> None:
        for idx, entry in other.episodes.items():
            current = self.episodes.get(idx)
            if current is None or entry.updated_at_unix >= current.updated_at_unix:
                self.episodes[idx] = entry

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        manifest = cls(
            schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
            created_at_unix=float(data.get("created_at_unix", time.time())),
            updated_at_unix=float(data.get("updated_at_unix", time.time())),
            source_path=data.get("source_path"),
            source_format=data.get("source_format"),
            target_format=data.get("target_format"),
        )

        episodes = data.get("episodes", {})
        if isinstance(episodes, dict):
            for k, v in episodes.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                if isinstance(v, dict):
                    manifest.episodes[idx] = EpisodeManifestEntry.from_dict(v)

        return manifest

    @classmethod
    def load_or_create(
        cls,
        output_path: Path,
        source_path: Path | None = None,
        source_format: str | None = None,
        target_format: str | None = None,
        manifest_path: Path | None = None,
    ) -> "DatasetManifest":
        manifest_path = manifest_path or get_default_manifest_path(output_path)
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                manifest = cls.from_dict(data)
            except Exception:
                manifest = cls()
        else:
            manifest = cls()

        if source_path is not None:
            manifest.source_path = str(source_path)
        if source_format is not None:
            manifest.source_format = source_format
        if target_format is not None:
            manifest.target_format = target_format

        manifest.updated_at_unix = time.time()
        return manifest

    def save(self, output_path: Path, manifest_path: Path | None = None) -> Path:
        manifest_path = manifest_path or get_default_manifest_path(output_path)
        if manifest_path.exists():
            try:
                on_disk = DatasetManifest.from_dict(json.loads(manifest_path.read_text()))
                on_disk.merge_from(self)
                self = on_disk
            except Exception:
                pass
        self.updated_at_unix = time.time()
        _atomic_write_text(manifest_path, self.to_json() + "\n")
        return manifest_path
