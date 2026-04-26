"""Tests for the dataset registry module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from robodata.core.exceptions import DatasetNotFoundError
from robodata.registry.models import (
    VALID_TAGS,
    DatasetEntry,
    ScaleInfo,
    SourceEntry,
)
from robodata.registry.registry import DatasetRegistry
from robodata.registry.validation import validate_registry


# --- Fixtures ---


SAMPLE_REGISTRY = {
    "version": "1.0.0",
    "updated_at": "2026-01-01",
    "datasets": {
        "test_dataset": {
            "id": "test_dataset",
            "name": "Test Dataset",
            "description": "A test dataset for unit tests.",
            "format": "lerobot",
            "embodiment": ["franka"],
            "sources": [
                {"type": "hf_hub", "uri": "test/test_dataset"},
                {"type": "gcs", "uri": "gs://bucket/test", "notes": "Backup"},
            ],
            "paper_url": "https://arxiv.org/abs/0000.00000",
            "license": "MIT",
            "task_types": ["pick_place"],
            "scale": {"episodes": 1000, "hours": 5.0, "approximate": True},
            "demo_suitable": True,
            "demo_episodes": 50,
            "demo_source_index": 0,
            "tags": ["manipulation", "real_world"],
            "verified": False,
            "added_at": "2026-01-01",
        },
        "sim_dataset": {
            "id": "sim_dataset",
            "name": "Sim Dataset",
            "description": "A simulated dataset with language conditioning.",
            "format": "rlds",
            "embodiment": ["sawyer"],
            "sources": [
                {"type": "gcs", "uri": "gs://bucket/sim"},
            ],
            "tags": ["simulation", "language_conditioned"],
            "verified": False,
        },
        "big_dataset": {
            "id": "big_dataset",
            "name": "Big Real Dataset",
            "description": "A large-scale real robot dataset.",
            "format": "rlds",
            "embodiment": ["franka", "ur5"],
            "sources": [
                {"type": "hf_hub", "uri": "org/big_dataset"},
            ],
            "scale": {"episodes": 100000, "approximate": True},
            "tags": ["manipulation", "real_world", "large_scale"],
            "verified": False,
        },
    },
}


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """Write a sample registry JSON and return its path."""
    p = tmp_path / "datasets.json"
    p.write_text(json.dumps(SAMPLE_REGISTRY))
    return p


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is cleared before and after each test."""
    DatasetRegistry.clear()
    yield
    DatasetRegistry.clear()
    # Clean up env var if set
    os.environ.pop("ROBODATA_REGISTRY_PATH", None)


# --- Model Tests ---


class TestScaleInfo:
    def test_create(self):
        s = ScaleInfo(episodes=100, hours=2.5, approximate=False)
        assert s.episodes == 100
        assert s.hours == 2.5
        assert s.approximate is False

    def test_from_dict(self):
        s = ScaleInfo.from_dict({"episodes": 50})
        assert s.episodes == 50
        assert s.hours is None
        assert s.approximate is True

    def test_to_dict_roundtrip(self):
        s = ScaleInfo(episodes=200, hours=10.0, approximate=False)
        d = s.to_dict()
        s2 = ScaleInfo.from_dict(d)
        assert s2.episodes == s.episodes
        assert s2.hours == s.hours


class TestSourceEntry:
    def test_create_valid(self):
        s = SourceEntry(type="hf_hub", uri="lerobot/pusht")
        assert s.type == "hf_hub"
        assert s.uri == "lerobot/pusht"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid source type"):
            SourceEntry(type="ftp", uri="ftp://example.com")

    def test_empty_uri(self):
        with pytest.raises(ValueError, match="URI cannot be empty"):
            SourceEntry(type="hf_hub", uri="")

    def test_from_dict(self):
        s = SourceEntry.from_dict({
            "type": "gcs",
            "uri": "gs://bucket/data",
            "split": "train",
            "notes": "Requires auth",
        })
        assert s.type == "gcs"
        assert s.split == "train"
        assert s.notes == "Requires auth"


class TestDatasetEntry:
    def test_create_minimal(self):
        e = DatasetEntry(
            id="test",
            name="Test",
            description="Test dataset",
            format="lerobot",
            embodiment=["franka"],
            sources=[SourceEntry(type="hf_hub", uri="test/test")],
        )
        assert e.id == "test"
        assert e.demo_suitable is False

    def test_empty_id(self):
        with pytest.raises(ValueError, match="id cannot be empty"):
            DatasetEntry(
                id="", name="Test", description="Test",
                format="lerobot", embodiment=[],
                sources=[SourceEntry(type="hf_hub", uri="x/y")],
            )

    def test_no_sources(self):
        with pytest.raises(ValueError, match="must have at least one source"):
            DatasetEntry(
                id="test", name="Test", description="Test",
                format="lerobot", embodiment=[], sources=[],
            )

    def test_invalid_tags(self):
        with pytest.raises(ValueError, match="invalid tags"):
            DatasetEntry(
                id="test", name="Test", description="Test",
                format="lerobot", embodiment=[],
                sources=[SourceEntry(type="hf_hub", uri="x/y")],
                tags=["manipulation", "bogus_tag"],
            )

    def test_bad_demo_source_index(self):
        with pytest.raises(ValueError, match="demo_source_index"):
            DatasetEntry(
                id="test", name="Test", description="Test",
                format="lerobot", embodiment=[],
                sources=[SourceEntry(type="hf_hub", uri="x/y")],
                demo_source_index=5,
            )

    def test_from_dict_roundtrip(self):
        data = SAMPLE_REGISTRY["datasets"]["test_dataset"]
        entry = DatasetEntry.from_dict(data)
        d = entry.to_dict()
        entry2 = DatasetEntry.from_dict(d)
        assert entry2.id == entry.id
        assert entry2.name == entry.name
        assert len(entry2.sources) == len(entry.sources)
        assert entry2.scale.episodes == entry.scale.episodes


# --- Registry Tests ---


class TestDatasetRegistry:
    def test_load(self, registry_path: Path):
        entries = DatasetRegistry.load(registry_path)
        assert len(entries) == 3
        assert "test_dataset" in entries

    def test_get_existing(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        entry = DatasetRegistry.get("test_dataset")
        assert entry.name == "Test Dataset"

    def test_get_nonexistent(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        with pytest.raises(DatasetNotFoundError, match="not found"):
            DatasetRegistry.get("nonexistent")

    def test_get_with_suggestions(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        with pytest.raises(DatasetNotFoundError) as exc_info:
            DatasetRegistry.get("test_datase")  # typo
        assert len(exc_info.value.suggestions) > 0

    def test_list_all(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.list()
        assert len(results) == 3

    def test_list_filter_by_format(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.list(format="rlds")
        assert len(results) == 2
        assert all("rlds" in e.format for e in results)

    def test_list_filter_by_embodiment(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.list(embodiment="franka")
        assert len(results) == 2

    def test_list_filter_by_tag(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.list(tag="simulation")
        assert len(results) == 1
        assert results[0].id == "sim_dataset"

    def test_list_demo_only(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.list(demo_only=True)
        assert len(results) == 1
        assert results[0].id == "test_dataset"

    def test_search(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.search("franka manipulation")
        assert len(results) >= 1
        # Franka datasets should be in results
        ids = [e.id for e in results]
        assert "test_dataset" in ids

    def test_search_empty_query(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.search("")
        assert len(results) == 3

    def test_search_no_results(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        results = DatasetRegistry.search("nonexistent_thing_xyz")
        assert len(results) == 0

    def test_get_source_default(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        source = DatasetRegistry.get_source("test_dataset")
        # Should prefer hf_hub
        assert source.type == "hf_hub"

    def test_get_source_demo(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        source = DatasetRegistry.get_source("test_dataset", demo=True)
        assert source.type == "hf_hub"
        assert source.uri == "test/test_dataset"

    def test_get_source_demo_not_available(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        with pytest.raises(ValueError, match="no demo-suitable subset"):
            DatasetRegistry.get_source("sim_dataset", demo=True)

    def test_demo_datasets(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        demos = DatasetRegistry.demo_datasets()
        assert len(demos) == 1

    def test_env_var_override(self, registry_path: Path):
        os.environ["ROBODATA_REGISTRY_PATH"] = str(registry_path)
        DatasetRegistry.clear()
        # Should load from env var path
        entries = DatasetRegistry._ensure_loaded()
        assert len(entries) == 3

    def test_clear(self, registry_path: Path):
        DatasetRegistry.load(registry_path)
        assert DatasetRegistry._entries is not None
        DatasetRegistry.clear()
        assert DatasetRegistry._entries is None


# --- Validation Tests ---


class TestRegistryValidation:
    def test_valid_registry(self, registry_path: Path):
        result = validate_registry(registry_path)
        assert result.ok

    def test_missing_file(self, tmp_path: Path):
        result = validate_registry(tmp_path / "nonexistent.json")
        assert not result.ok
        assert any("not found" in e for e in result.errors)

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json")
        result = validate_registry(p)
        assert not result.ok

    def test_missing_datasets_key(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text('{"version": "1.0.0"}')
        result = validate_registry(p)
        assert not result.ok
        assert any("datasets" in e for e in result.errors)

    def test_duplicate_ids(self, tmp_path: Path):
        data = {
            "version": "1.0.0",
            "datasets": {
                "a": {"id": "same_id", "name": "A", "description": "A", "format": "rlds",
                       "sources": [{"type": "hf_hub", "uri": "x/y"}]},
                "b": {"id": "same_id", "name": "B", "description": "B", "format": "rlds",
                       "sources": [{"type": "hf_hub", "uri": "x/z"}]},
            },
        }
        p = tmp_path / "dup.json"
        p.write_text(json.dumps(data))
        result = validate_registry(p)
        assert any("Duplicate" in e for e in result.errors)

    def test_invalid_source_type(self, tmp_path: Path):
        data = {
            "version": "1.0.0",
            "datasets": {
                "bad": {"id": "bad", "name": "Bad", "description": "Bad", "format": "rlds",
                        "sources": [{"type": "ftp", "uri": "ftp://x"}]},
            },
        }
        p = tmp_path / "bad_source.json"
        p.write_text(json.dumps(data))
        result = validate_registry(p)
        assert any("invalid type" in e for e in result.errors)

    def test_invalid_tags(self, tmp_path: Path):
        data = {
            "version": "1.0.0",
            "datasets": {
                "bad": {"id": "bad", "name": "Bad", "description": "Bad", "format": "rlds",
                        "sources": [{"type": "hf_hub", "uri": "x/y"}],
                        "tags": ["bogus_tag"]},
            },
        }
        p = tmp_path / "bad_tags.json"
        p.write_text(json.dumps(data))
        result = validate_registry(p)
        assert any("Invalid tag" in e for e in result.errors)

    def test_bad_demo_source_index(self, tmp_path: Path):
        data = {
            "version": "1.0.0",
            "datasets": {
                "bad": {"id": "bad", "name": "Bad", "description": "Bad", "format": "rlds",
                        "sources": [{"type": "hf_hub", "uri": "x/y"}],
                        "demo_source_index": 5},
            },
        }
        p = tmp_path / "bad_idx.json"
        p.write_text(json.dumps(data))
        result = validate_registry(p)
        assert any("demo_source_index" in e for e in result.errors)

    def test_bundled_registry_valid(self):
        """Validate the actual bundled datasets.json."""
        result = validate_registry()
        assert result.ok, f"Bundled registry has errors: {result.errors}"
