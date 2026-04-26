"""Central registry for known robotics datasets.

Loads dataset metadata from a bundled JSON file and provides lookup,
filtering, and search capabilities. Used by CLI commands to resolve
dataset names to download paths and format information.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

from robodata.core.exceptions import DatasetNotFoundError
from robodata.registry.models import DatasetEntry, SourceEntry


class DatasetRegistry:
    """Registry of known robotics datasets.

    Uses class-level caching — loads from JSON on first access.
    Override the JSON path with the ROBODATA_REGISTRY_PATH env var.
    """

    _entries: dict[str, DatasetEntry] | None = None
    _loaded_path: Path | None = None

    @classmethod
    def _default_path(cls) -> Path:
        env_path = os.environ.get("ROBODATA_REGISTRY_PATH")
        if env_path:
            return Path(env_path)
        return Path(__file__).parent / "datasets.json"

    @classmethod
    def _ensure_loaded(cls) -> dict[str, DatasetEntry]:
        if cls._entries is None:
            cls.load()
        assert cls._entries is not None
        return cls._entries

    @classmethod
    def load(cls, path: Path | None = None) -> dict[str, DatasetEntry]:
        """Load the registry from JSON.

        Args:
            path: Override path to datasets.json. If None, uses default.

        Returns:
            Dict mapping dataset IDs to DatasetEntry objects.
        """
        registry_path = path or cls._default_path()
        registry_path = Path(registry_path)

        with open(registry_path) as f:
            data = json.load(f)

        entries: dict[str, DatasetEntry] = {}
        datasets = data.get("datasets", {})
        for dataset_id, dataset_data in datasets.items():
            if "id" not in dataset_data:
                dataset_data["id"] = dataset_id
            entries[dataset_id] = DatasetEntry.from_dict(dataset_data)

        cls._entries = entries
        cls._loaded_path = registry_path
        return entries

    @classmethod
    def get(cls, dataset_id: str) -> DatasetEntry:
        """Get a dataset by exact ID.

        Args:
            dataset_id: The dataset identifier (e.g., "droid", "bridge_v2").

        Returns:
            The matching DatasetEntry.

        Raises:
            DatasetNotFoundError: If the ID is not found, with suggestions.
        """
        entries = cls._ensure_loaded()
        if dataset_id in entries:
            return entries[dataset_id]

        # Find similar IDs for suggestions
        all_ids = list(entries.keys())
        suggestions = difflib.get_close_matches(dataset_id, all_ids, n=3, cutoff=0.4)
        raise DatasetNotFoundError(dataset_id, suggestions)

    @classmethod
    def list(
        cls,
        format: str | None = None,
        embodiment: str | None = None,
        tag: str | None = None,
        demo_only: bool = False,
    ) -> list[DatasetEntry]:
        """List datasets with optional filters.

        Args:
            format: Filter by dataset format (e.g., "rlds", "lerobot").
            embodiment: Filter by robot embodiment (e.g., "franka").
            tag: Filter by tag (e.g., "language_conditioned").
            demo_only: If True, only return demo-suitable datasets.

        Returns:
            List of matching DatasetEntry objects.
        """
        entries = cls._ensure_loaded()
        results = list(entries.values())

        if format:
            fmt_lower = format.lower()
            results = [e for e in results if fmt_lower in e.format.lower()]

        if embodiment:
            emb_lower = embodiment.lower()
            results = [
                e for e in results
                if any(emb_lower in emb.lower() for emb in e.embodiment)
            ]

        if tag:
            tag_lower = tag.lower()
            results = [
                e for e in results
                if any(tag_lower in t.lower() for t in e.tags)
            ]

        if demo_only:
            results = [e for e in results if e.demo_suitable]

        return results

    @classmethod
    def search(cls, query: str) -> list[DatasetEntry]:
        """Search datasets by keyword across multiple fields.

        Matches against id, name, description, tags, embodiment, and task_types.
        Tokens in the query are AND-matched.

        Args:
            query: Search string (e.g., "franka manipulation").

        Returns:
            List of matching DatasetEntry objects, sorted by relevance.
        """
        entries = cls._ensure_loaded()
        tokens = query.lower().split()
        if not tokens:
            return list(entries.values())

        scored: list[tuple[int, DatasetEntry]] = []
        for entry in entries.values():
            searchable = " ".join([
                entry.id,
                entry.name,
                entry.description,
                " ".join(entry.tags),
                " ".join(entry.embodiment),
                " ".join(entry.task_types),
                entry.format,
            ]).lower()

            # All tokens must match
            if all(t in searchable for t in tokens):
                # Score: number of field matches (more matches = more relevant)
                score = sum(
                    sum(1 for t in tokens if t in field.lower())
                    for field in [
                        entry.id, entry.name, " ".join(entry.tags),
                        " ".join(entry.embodiment),
                    ]
                )
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored]

    @classmethod
    def get_source(
        cls,
        dataset_id: str,
        split: str | None = None,
        demo: bool = False,
    ) -> SourceEntry:
        """Get the appropriate download source for a dataset.

        Args:
            dataset_id: The dataset identifier.
            split: Desired split (train, val, test).
            demo: If True, return the demo-suitable source.

        Returns:
            The matching SourceEntry.

        Raises:
            DatasetNotFoundError: If the dataset ID is not found.
            ValueError: If demo is requested but no demo source exists.
        """
        entry = cls.get(dataset_id)

        if demo:
            if not entry.demo_suitable:
                raise ValueError(
                    f"Dataset '{dataset_id}' has no demo-suitable subset. "
                    f"Use without --demo to get the full dataset."
                )
            if entry.demo_source_index is not None:
                return entry.sources[entry.demo_source_index]
            # Fall through to default source if demo_suitable but no specific index

        if split:
            for source in entry.sources:
                if source.split and source.split.lower() == split.lower():
                    return source

        # Return first source (prefer hf_hub over others)
        hf_sources = [s for s in entry.sources if s.type == "hf_hub"]
        if hf_sources:
            return hf_sources[0]
        return entry.sources[0]

    @classmethod
    def demo_datasets(cls) -> list[DatasetEntry]:
        """Return all demo-suitable datasets (<=100 episodes)."""
        return cls.list(demo_only=True)

    @classmethod
    def clear(cls) -> None:
        """Reset the registry cache. Used for testing."""
        cls._entries = None
        cls._loaded_path = None
