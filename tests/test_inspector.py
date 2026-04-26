"""Tests for the Inspector facade."""

from pathlib import Path

import pytest

from robodata.inspect.inspector import InspectionOptions, Inspector


class TestInspectionOptions:
    """Tests for InspectionOptions."""

    def test_default_options(self):
        """Test default inspection options."""
        opts = InspectionOptions()
        assert opts.sample_episodes == 5  # Default is 5
        assert opts.max_frames_per_episode == 100
        assert opts.detect_gripper is True
        assert opts.detect_fps is True
        assert opts.deep_scan is False

    def test_custom_options(self):
        """Test custom inspection options."""
        opts = InspectionOptions(
            sample_episodes=10,
            max_frames_per_episode=50,
            detect_gripper=False,
            deep_scan=True,
        )
        assert opts.sample_episodes == 10
        assert opts.max_frames_per_episode == 50
        assert opts.detect_gripper is False
        assert opts.deep_scan is True


class TestInspector:
    """Tests for Inspector facade."""

    def test_inspect_zarr(self, temp_zarr_dataset: Path):
        """Test inspecting a Zarr dataset."""
        inspector = Inspector()
        info = inspector.inspect(str(temp_zarr_dataset))

        assert info.format == "zarr"
        assert info.num_episodes == 2
        assert "camera0" in info.cameras

    def test_inspect_with_format_hint(self, temp_zarr_dataset: Path):
        """Test inspecting with format hint."""
        inspector = Inspector()
        info = inspector.inspect(str(temp_zarr_dataset), format="zarr")

        assert info.format == "zarr"

    def test_inspect_nonexistent_path(self):
        """Test inspecting non-existent path raises error."""
        from robodata.core.exceptions import InspectionError

        inspector = Inspector()
        with pytest.raises(InspectionError):
            inspector.inspect("/nonexistent/path")

    def test_inspect_with_options(self, temp_zarr_dataset: Path):
        """Test inspecting with custom options."""
        opts = InspectionOptions(sample_episodes=1, detect_gripper=False)
        inspector = Inspector(opts)
        info = inspector.inspect(str(temp_zarr_dataset))

        assert info.format == "zarr"


class TestPublicAPI:
    """Tests for the public robodata.inspect() API."""

    def test_inspect_function(self, temp_zarr_dataset: Path):
        """Test the robodata.inspect() function."""
        from robodata import inspect

        info = inspect(str(temp_zarr_dataset))

        assert info.format == "zarr"
        assert info.num_episodes == 2

    def test_inspect_with_kwargs(self, temp_zarr_dataset: Path):
        """Test robodata.inspect() with options as kwargs."""
        from robodata import inspect

        info = inspect(str(temp_zarr_dataset), sample_episodes=1)

        assert info.format == "zarr"
