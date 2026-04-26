"""Tests for format registry."""

from pathlib import Path

import pytest

from robodata.formats.registry import FormatRegistry


class TestFormatRegistry:
    """Tests for FormatRegistry."""

    def test_list_formats(self):
        """Test listing available formats."""
        formats = FormatRegistry.list_formats()

        # Should have at least the core formats
        assert "lerobot-v2" in formats
        assert "lerobot-v3" in formats
        assert "zarr" in formats
        assert "rosbag" in formats

    def test_format_capabilities(self):
        """Test format capabilities reporting."""
        formats = FormatRegistry.list_formats()

        # All implemented formats should have read capability
        for name, caps in formats.items():
            assert "can_read" in caps
            assert "can_write" in caps

        # Core formats should have read capability
        assert formats["zarr"]["can_read"] is True
        assert formats["lerobot-v3"]["can_read"] is True
        assert formats["rosbag"]["can_read"] is True

        # lerobot-v3 now has write capability
        assert formats["lerobot-v3"]["can_write"] is True

    def test_get_reader(self):
        """Test getting a reader by format name."""
        reader = FormatRegistry.get_reader("zarr")
        assert reader is not None
        assert reader.format_name == "zarr"

    def test_get_reader_unknown_format(self):
        """Test getting reader for unknown format raises error."""
        from robodata.core.exceptions import UnsupportedFormatError

        with pytest.raises(UnsupportedFormatError):
            FormatRegistry.get_reader("unknown_format")

    def test_detect_format_zarr(self, temp_zarr_dataset: Path):
        """Test format detection for Zarr dataset."""
        detected = FormatRegistry.detect_format(temp_zarr_dataset)
        assert detected == "zarr"

    def test_detect_format_lerobot_v3(self, temp_lerobot_v3_dataset: Path):
        """Test format detection for LeRobot v3 dataset."""
        detected = FormatRegistry.detect_format(temp_lerobot_v3_dataset)
        assert detected == "lerobot-v3"

    def test_detect_format_rosbag(self, temp_rosbag_metadata: Path):
        """Test format detection for rosbag."""
        detected = FormatRegistry.detect_format(temp_rosbag_metadata)
        assert detected == "rosbag"

    def test_detect_format_nonexistent(self, tmp_path: Path):
        """Test format detection for non-existent path."""
        from robodata.core.exceptions import FormatDetectionError

        with pytest.raises(FormatDetectionError):
            FormatRegistry.detect_format(tmp_path / "nonexistent")

    def test_detect_format_unknown(self, tmp_path: Path):
        """Test format detection for unknown format."""
        from robodata.core.exceptions import FormatDetectionError

        # Create an empty directory
        unknown_dir = tmp_path / "unknown"
        unknown_dir.mkdir()

        with pytest.raises(FormatDetectionError):
            FormatRegistry.detect_format(unknown_dir)

    def test_detection_priority(self):
        """Test that v3 is detected before v2 for LeRobot."""
        # v3 should come before v2 in priority
        priority = FormatRegistry._detection_priority
        v3_idx = priority.index("lerobot-v3")
        v2_idx = priority.index("lerobot-v2")
        assert v3_idx < v2_idx
