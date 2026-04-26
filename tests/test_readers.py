"""Tests for format readers."""

from pathlib import Path

import pytest


class TestZarrReader:
    """Tests for ZarrReader."""

    def test_can_read_zarr_directory(self, temp_zarr_dataset: Path):
        """Test that Zarr reader can detect Zarr directories."""
        from robodata.formats.zarr.reader import ZarrReader

        assert ZarrReader.can_read(temp_zarr_dataset) is True

    def test_cannot_read_non_zarr(self, tmp_path: Path):
        """Test that Zarr reader rejects non-Zarr paths."""
        from robodata.formats.zarr.reader import ZarrReader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert ZarrReader.can_read(empty_dir) is False

    def test_inspect_zarr(self, temp_zarr_dataset: Path):
        """Test inspecting a Zarr dataset."""
        from robodata.formats.zarr.reader import ZarrReader

        reader = ZarrReader()
        info = reader.inspect(temp_zarr_dataset)

        assert info.format == "zarr"
        assert info.format_version == "1.0"
        assert info.num_episodes == 2
        assert info.total_frames == 100
        assert info.inferred_fps == 30
        assert info.inferred_robot_type == "test_robot"

        # Should detect camera
        assert "camera0" in info.cameras
        assert info.cameras["camera0"].height == 32
        assert info.cameras["camera0"].width == 32

        # Should detect action
        assert info.action_schema is not None
        assert info.action_schema.shape == (7,)


class TestLeRobotV3Reader:
    """Tests for LeRobotV3Reader."""

    def test_can_read_lerobot_v3(self, temp_lerobot_v3_dataset: Path):
        """Test that LeRobot v3 reader can detect v3 datasets."""
        from robodata.formats.lerobot_v3.reader import LeRobotV3Reader

        assert LeRobotV3Reader.can_read(temp_lerobot_v3_dataset) is True

    def test_cannot_read_v2(self, tmp_path: Path):
        """Test that v3 reader rejects v2 datasets."""
        import json

        from robodata.formats.lerobot_v3.reader import LeRobotV3Reader

        # Create v2-style dataset (no info.json with codebase_version)
        v2_dataset = tmp_path / "v2_dataset"
        v2_dataset.mkdir()
        (v2_dataset / "data").mkdir()
        (v2_dataset / "videos").mkdir()

        # v2 has meta/info.json, not info.json at root
        (v2_dataset / "meta").mkdir()
        with open(v2_dataset / "meta" / "info.json", "w") as f:
            json.dump({"fps": 30}, f)

        assert LeRobotV3Reader.can_read(v2_dataset) is False

    def test_detect_version(self, temp_lerobot_v3_dataset: Path):
        """Test version detection for LeRobot v3."""
        from robodata.formats.lerobot_v3.reader import LeRobotV3Reader

        version = LeRobotV3Reader.detect_version(temp_lerobot_v3_dataset)
        assert version == "3.0"  # Reader strips the "v" prefix

    def test_inspect_lerobot_v3(self, temp_lerobot_v3_dataset: Path):
        """Test inspecting a LeRobot v3 dataset."""
        from robodata.formats.lerobot_v3.reader import LeRobotV3Reader

        reader = LeRobotV3Reader()
        info = reader.inspect(temp_lerobot_v3_dataset)

        assert info.format == "lerobot-v3"
        assert info.format_version == "3.0"  # Reader strips the "v" prefix
        assert info.num_episodes == 2
        assert info.total_frames == 100
        assert info.inferred_fps == 30
        assert info.inferred_robot_type == "test_robot"


class TestRosbagReader:
    """Tests for RosbagReader."""

    def test_can_read_rosbag_directory(self, temp_rosbag_metadata: Path):
        """Test that rosbag reader can detect ROS2 bag directories."""
        from robodata.formats.rosbag.reader import RosbagReader

        assert RosbagReader.can_read(temp_rosbag_metadata) is True

    def test_cannot_read_non_rosbag(self, tmp_path: Path):
        """Test that rosbag reader rejects non-rosbag paths."""
        from robodata.formats.rosbag.reader import RosbagReader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert RosbagReader.can_read(empty_dir) is False

    def test_detect_version_ros2_sqlite(self, temp_rosbag_metadata: Path):
        """Test version detection for ROS2 bags via metadata.yaml."""
        from robodata.formats.rosbag.reader import RosbagReader

        # With only metadata.yaml (no actual .db3 files), returns "ros2"
        version = RosbagReader.detect_version(temp_rosbag_metadata)
        assert version == "ros2"

    def test_inspect_rosbag_metadata(self, temp_rosbag_metadata: Path):
        """Test inspecting a rosbag via metadata.yaml."""
        from robodata.formats.rosbag.reader import RosbagReader

        reader = RosbagReader()
        info = reader.inspect(temp_rosbag_metadata)

        assert info.format == "rosbag"
        assert info.format_version == "ros2-sqlite3"  # From metadata.yaml storage_identifier
        assert info.num_episodes == 1
        assert info.total_frames == 300  # Image message count

        # Should detect camera from Image topic
        # _extract_camera_name("/camera/rgb/image_raw") -> "camera"
        assert len(info.cameras) == 1
        assert "camera" in info.cameras

        # Should detect JointState as observation
        assert "/robot/joint_states" in info.observation_schema

        # Should infer FPS (300 frames / 10 seconds = 30)
        assert info.inferred_fps == 30.0


class TestReaderIntegration:
    """Integration tests for readers with real sample data."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "sample_data" / "zarr" / "pusht").exists(),
        reason="Sample data not available",
    )
    def test_inspect_pusht_dataset(self, sample_data_dir: Path):
        """Test inspecting the pusht Zarr dataset."""
        zarr = pytest.importorskip("zarr")
        from robodata import inspect

        pusht_path = sample_data_dir / "zarr" / "pusht" / "pusht_cchi_v7_replay.zarr"
        if not pusht_path.exists():
            pytest.skip("pusht dataset not available")

        info = inspect(str(pusht_path))

        assert info.format == "zarr"
        assert info.num_episodes > 0
        assert "img" in info.cameras

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "sample_data" / "rosbag" / "r2b_storage").exists(),
        reason="Sample data not available",
    )
    def test_inspect_r2b_dataset(self, sample_data_dir: Path):
        """Test inspecting the R2B rosbag dataset."""
        from robodata import inspect

        r2b_path = sample_data_dir / "rosbag" / "r2b_storage"
        if not r2b_path.exists():
            pytest.skip("R2B dataset not available")

        info = inspect(str(r2b_path))

        assert info.format == "rosbag"
        assert info.format_version == "ros2-sqlite3"
        assert len(info.cameras) > 0
