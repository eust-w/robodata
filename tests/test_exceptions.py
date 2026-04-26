"""Tests for custom exceptions."""

from pathlib import Path

from robodata.core.exceptions import (
    ConversionError,
    EpisodeNotFoundError,
    RobodataError,
    FormatDetectionError,
    InspectionError,
    MissingDependencyError,
    SchemaError,
    UnsupportedFormatError,
    ValidationError,
)


class TestExceptions:
    """Tests for Robodata exceptions."""

    def test_error_base(self):
        """Test RobodataError base class."""
        error = RobodataError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError."""
        error = UnsupportedFormatError("hdf5")
        assert "hdf5" in str(error)
        assert isinstance(error, RobodataError)

    def test_format_detection_error(self):
        """Test FormatDetectionError."""
        error = FormatDetectionError(Path("/data/unknown"))
        assert "/data/unknown" in str(error)
        assert isinstance(error, RobodataError)

    def test_inspection_error(self):
        """Test InspectionError."""
        error = InspectionError(Path("/data/broken"), "Corrupt file")
        assert "/data/broken" in str(error)
        assert "Corrupt file" in str(error)
        assert isinstance(error, RobodataError)

    def test_missing_dependency_error(self):
        """Test MissingDependencyError."""
        error = MissingDependencyError(
            dependency="tensorflow",
            feature="RLDS support",
            install_hint="pip install robodata[rlds]",
        )
        assert "tensorflow" in str(error)
        assert "RLDS support" in str(error)
        assert "pip install" in str(error)
        assert isinstance(error, RobodataError)

    def test_episode_not_found_error(self):
        """Test EpisodeNotFoundError."""
        error = EpisodeNotFoundError("ep_999", Path("/data/test"))
        assert "ep_999" in str(error)
        assert "/data/test" in str(error)
        assert isinstance(error, RobodataError)

    def test_validation_error(self):
        """Test ValidationError with list of errors."""
        error = ValidationError(["Missing required field: fps", "Invalid action shape"])
        assert "fps" in str(error)
        assert "Invalid action shape" in str(error)
        assert isinstance(error, RobodataError)

    def test_validation_error_with_episode(self):
        """Test ValidationError with episode ID."""
        error = ValidationError(["Bad data"], episode_id="ep_001")
        assert "ep_001" in str(error)
        assert isinstance(error, RobodataError)

    def test_schema_error(self):
        """Test SchemaError."""
        error = SchemaError(
            field_name="action",
            expected="(7,)",
            actual="(6,)",
        )
        assert "action" in str(error)
        assert "(7,)" in str(error)
        assert "(6,)" in str(error)
        assert isinstance(error, RobodataError)

    def test_conversion_error(self):
        """Test ConversionError."""
        error = ConversionError(
            source_format="rlds",
            target_format="lerobot-v3",
            reason="Missing camera data",
        )
        assert "rlds" in str(error)
        assert "lerobot-v3" in str(error)
        assert "Missing camera data" in str(error)
        assert isinstance(error, RobodataError)
