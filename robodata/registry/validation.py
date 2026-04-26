"""Registry validation — schema checks, duplicate detection, and optional source probing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robodata.registry.models import VALID_SOURCE_TYPES, VALID_TAGS


@dataclass
class ValidationResult:
    """Result of validating the registry."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        if not parts:
            return "Registry is valid."
        return f"Validation: {', '.join(parts)}"


REQUIRED_DATASET_FIELDS = {"id", "name", "description", "format", "sources"}
RECOMMENDED_FIELDS = {"paper_url", "license", "embodiment", "tags"}


def validate_registry(
    path: Path | None = None,
    probe: bool = False,
) -> ValidationResult:
    """Validate the registry JSON file.

    Args:
        path: Path to datasets.json. If None, uses the bundled default.
        probe: If True, check that source URIs are reachable (requires network).

    Returns:
        ValidationResult with errors and warnings.
    """
    if path is None:
        path = Path(__file__).parent / "datasets.json"

    result = ValidationResult()

    # Load and parse JSON
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        result.errors.append(f"Registry file not found: {path}")
        return result
    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON: {e}")
        return result

    # Check top-level structure
    if "version" not in data:
        result.warnings.append("Missing top-level 'version' field")
    if "datasets" not in data:
        result.errors.append("Missing top-level 'datasets' field")
        return result

    datasets = data["datasets"]
    if not isinstance(datasets, dict):
        result.errors.append("'datasets' must be an object/dict")
        return result

    seen_ids: set[str] = set()

    for key, entry in datasets.items():
        if not isinstance(entry, dict):
            result.errors.append(f"[{key}] Entry must be an object/dict")
            continue

        # Check required fields
        for req in REQUIRED_DATASET_FIELDS:
            if req not in entry:
                result.errors.append(f"[{key}] Missing required field: '{req}'")

        # Check recommended fields
        for rec in RECOMMENDED_FIELDS:
            if rec not in entry or not entry.get(rec):
                result.warnings.append(f"[{key}] Missing recommended field: '{rec}'")

        # Duplicate ID check
        entry_id = entry.get("id", key)
        if entry_id in seen_ids:
            result.errors.append(f"[{key}] Duplicate dataset ID: '{entry_id}'")
        seen_ids.add(entry_id)

        # Key vs id mismatch
        if "id" in entry and entry["id"] != key:
            result.warnings.append(
                f"[{key}] Key '{key}' does not match 'id' field '{entry['id']}'"
            )

        # Validate sources
        sources = entry.get("sources", [])
        if not isinstance(sources, list):
            result.errors.append(f"[{key}] 'sources' must be a list")
        else:
            for source_idx, source in enumerate(sources):
                if not isinstance(source, dict):
                    result.errors.append(f"[{key}] sources[{source_idx}] must be an object")
                    continue
                source_type = source.get("type")
                if source_type and source_type not in VALID_SOURCE_TYPES:
                    result.errors.append(
                        f"[{key}] sources[{source_idx}] invalid type '{source_type}'"
                    )
                if not source.get("uri"):
                    result.errors.append(f"[{key}] sources[{source_idx}] missing 'uri'")

        # Validate tags
        tags = entry.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if tag not in VALID_TAGS:
                    result.errors.append(f"[{key}] Invalid tag: '{tag}'")

        # Validate demo_source_index
        demo_idx = entry.get("demo_source_index")
        if demo_idx is not None and isinstance(sources, list):
            if not isinstance(demo_idx, int) or demo_idx < 0 or demo_idx >= len(sources):
                result.errors.append(
                    f"[{key}] demo_source_index {demo_idx} out of range "
                    f"(0-{len(sources) - 1})"
                )

    # Optional: probe sources
    if probe:
        _probe_sources(data["datasets"], result)

    return result


def _probe_sources(datasets: dict[str, Any], result: ValidationResult) -> None:
    """Check that source URIs are reachable."""
    try:
        import urllib.request
    except ImportError:
        result.warnings.append("Cannot probe sources: urllib not available")
        return

    for key, entry in datasets.items():
        for i, source in enumerate(entry.get("sources", [])):
            uri = source.get("uri", "")
            stype = source.get("type", "")

            if stype == "hf_hub":
                url = f"https://huggingface.co/datasets/{uri}"
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    urllib.request.urlopen(req, timeout=10)
                except Exception as e:
                    result.warnings.append(
                        f"[{key}] sources[{i}] HF Hub probe failed: {e}"
                    )
            elif stype == "http":
                try:
                    req = urllib.request.Request(uri, method="HEAD")
                    urllib.request.urlopen(req, timeout=10)
                except Exception as e:
                    result.warnings.append(
                        f"[{key}] sources[{i}] HTTP probe failed: {e}"
                    )
            # GCS and rsync probing skipped (requires gcloud/rsync CLI)
