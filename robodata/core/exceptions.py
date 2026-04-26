"""Custom exceptions for Robodata.

All Robodata-specific exceptions inherit from RobodataError, allowing users
to catch all Robodata errors with a single except clause if desired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RobodataError(Exception):
    """Base exception for all Robodata errors."""

    pass


class UnsupportedFormatError(RobodataError):
    """Raised when a format is not supported.

    Attributes:
        format_name: The unsupported format identifier.
        available_formats: List of available format names.
    """

    def __init__(
        self,
        format_name: str,
        available_formats: list[str] | None = None,
        message: str | None = None,
    ):
        self.format_name = format_name
        self.available_formats = available_formats or []

        if message:
            super().__init__(message)
        elif available_formats:
            super().__init__(
                f"Unsupported format: '{format_name}'. "
                f"Available formats: {', '.join(available_formats)}"
            )
        else:
            super().__init__(f"Unsupported format: '{format_name}'")


class FormatDetectionError(RobodataError):
    """Raised when format cannot be detected.

    Attributes:
        path: Path that was being inspected.
    """

    def __init__(self, path: Path | str, message: str | None = None):
        self.path = Path(path)

        if message:
            super().__init__(message)
        else:
            super().__init__(f"Could not detect format for: {self.path}")


class InspectionError(RobodataError):
    """Raised when dataset inspection fails.

    Attributes:
        path: Path to the dataset.
        reason: Specific reason for failure.
    """

    def __init__(self, path: Path | str, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to inspect '{self.path}': {reason}")


class ValidationError(RobodataError):
    """Raised when data validation fails.

    Attributes:
        errors: List of validation error messages.
        episode_id: Optional episode where error occurred.
    """

    def __init__(
        self,
        errors: list[str],
        episode_id: str | None = None,
    ):
        self.errors = errors
        self.episode_id = episode_id

        if episode_id:
            message = f"Validation failed for episode '{episode_id}': {'; '.join(errors)}"
        else:
            message = f"Validation failed: {'; '.join(errors)}"

        super().__init__(message)


class ConversionError(RobodataError):
    """Raised when conversion fails.

    Attributes:
        source_format: Source format identifier.
        target_format: Target format identifier.
        reason: Specific reason for failure.
    """

    def __init__(
        self,
        source_format: str,
        target_format: str,
        reason: str,
    ):
        self.source_format = source_format
        self.target_format = target_format
        self.reason = reason
        super().__init__(f"Failed to convert from '{source_format}' to '{target_format}': {reason}")


class MissingDependencyError(RobodataError):
    """Raised when an optional dependency is not installed.

    Attributes:
        dependency: Name of the missing dependency.
        feature: Feature that requires the dependency.
        install_hint: pip install command hint.
    """

    def __init__(
        self,
        dependency: str,
        feature: str,
        install_hint: str | None = None,
    ):
        self.dependency = dependency
        self.feature = feature
        self.install_hint = install_hint or f"pip install {dependency}"

        super().__init__(
            f"Missing dependency '{dependency}' for {feature}. Install with: {self.install_hint}"
        )


class SchemaError(RobodataError):
    """Raised when there's a schema mismatch or incompatibility.

    Attributes:
        field_name: Name of the problematic field.
        expected: Expected schema/type.
        actual: Actual schema/type found.
    """

    def __init__(
        self,
        field_name: str,
        expected: Any,
        actual: Any,
    ):
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Schema mismatch for field '{field_name}': expected {expected}, got {actual}"
        )


class DatasetNotFoundError(RobodataError):
    """Raised when a dataset is not found in the registry.

    Attributes:
        dataset_id: The requested dataset ID.
        suggestions: Similar dataset IDs that might be what the user meant.
    """

    def __init__(
        self,
        dataset_id: str,
        suggestions: list[str] | None = None,
    ):
        self.dataset_id = dataset_id
        self.suggestions = suggestions or []

        msg = f"Dataset '{dataset_id}' not found in registry."
        if self.suggestions:
            msg += f" Did you mean: {', '.join(self.suggestions)}?"
        super().__init__(msg)


class EpisodeNotFoundError(RobodataError):
    """Raised when a specific episode cannot be found.

    Attributes:
        episode_id: The requested episode ID.
        path: Path to the dataset.
    """

    def __init__(self, episode_id: str, path: Path | str | None = None):
        self.episode_id = episode_id
        self.path = Path(path) if path else None

        if path:
            super().__init__(f"Episode '{episode_id}' not found in {self.path}")
        else:
            super().__init__(f"Episode '{episode_id}' not found")
