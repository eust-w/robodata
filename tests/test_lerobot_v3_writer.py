"""Tests for LeRobot V3 writer."""

import json
from pathlib import Path

import numpy as np
import pytest

from robodata.core.models import CameraInfo, Episode, Frame, LazyImage
from robodata.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample episode for testing."""

    def create_frame(idx: int) -> Frame:
        def make_image_loader(index: int):
            def loader():
                # Create a simple gradient image
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (index * 10) % 256
                img[:, :, 1] = np.arange(64).reshape(-1, 1)
                img[:, :, 2] = np.arange(64).reshape(1, -1)
                return img

            return loader

        return Frame(
            index=idx,
            timestamp=idx / 30.0,
            images={
                "camera0": LazyImage(
                    loader=make_image_loader(idx),
                    height=64,
                    width=64,
                    channels=3,
                ),
            },
            state=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            action=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], dtype=np.float32),
        )

    def frame_loader():
        for i in range(30):  # 1 second at 30 fps
            yield create_frame(i)

    return Episode(
        episode_id="ep_001",
        language_instruction="Pick up the red block",
        cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
        fps=30.0,
        _frame_loader=frame_loader,
    )


@pytest.fixture
def sample_episodes(sample_episode: Episode) -> list[Episode]:
    """Create multiple sample episodes."""

    def create_episode(ep_idx: int) -> Episode:
        def create_frame(idx: int) -> Frame:
            def make_image_loader(index: int):
                def loader():
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    img[:, :, 0] = (index * 10) % 256
                    return img

                return loader

            return Frame(
                index=idx,
                timestamp=idx / 30.0,
                images={
                    "camera0": LazyImage(
                        loader=make_image_loader(idx),
                        height=64,
                        width=64,
                        channels=3,
                    ),
                },
                state=np.array([0.1] * 7, dtype=np.float32),
                action=np.array([0.01] * 7, dtype=np.float32),
            )

        def frame_loader():
            for i in range(20 + ep_idx * 5):  # Variable length episodes
                yield create_frame(i)

        return Episode(
            episode_id=f"ep_{ep_idx:03d}",
            language_instruction=f"Task {ep_idx % 2}",  # Two different tasks
            cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

    return [create_episode(i) for i in range(3)]


class TestLeRobotV3WriterConfig:
    """Tests for LeRobotV3WriterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LeRobotV3WriterConfig()
        assert config.fps == 30.0
        assert config.robot_type == "unknown"
        assert config.video_codec == "libx264"
        assert config.video_crf == 23
        assert config.chunks_size == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LeRobotV3WriterConfig(
            fps=60.0,
            robot_type="franka",
            video_crf=18,
            chunks_size=500,
            camera_name_mapping={"cam0": "observation.images.front"},
        )
        assert config.fps == 60.0
        assert config.robot_type == "franka"
        assert config.video_crf == 18
        assert config.chunks_size == 500
        assert config.camera_name_mapping == {"cam0": "observation.images.front"}


class TestLeRobotV3Writer:
    """Tests for LeRobotV3Writer."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_single_episode(self, tmp_path: Path, sample_episode: Episode):
        """Test writing a single episode."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)  # Flush to write data

        # Check parquet file exists (chunked structure)
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        assert parquet_path.exists()

        # Check video file exists (chunked structure)
        video_path = output_dir / "videos" / "observation.images.camera0" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_write_dataset(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test writing full dataset with multiple episodes."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Check structure
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
        assert (output_dir / "meta" / "tasks.parquet").exists()
        assert (output_dir / "data" / "chunk-000").exists()
        assert (output_dir / "videos").exists()

        # Check data parquet exists
        parquet_files = list((output_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(parquet_files) == 1

        # Check metadata content
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["fps"] == 30
        assert info["total_episodes"] == 3
        assert info["splits"]["train"] == "0:3"
        assert "data_path" in info
        assert "video_path" in info

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_episodes_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test episodes parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read episodes parquet
        episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(episodes_path)
        episodes_data = table.to_pylist()

        assert len(episodes_data) == 3
        assert episodes_data[0]["episode_index"] == 0
        assert episodes_data[1]["episode_index"] == 1
        assert episodes_data[2]["episode_index"] == 2
        assert all("length" in ep for ep in episodes_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_tasks_parquet_content(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test tasks.parquet has correct content."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        # Read tasks.parquet
        tasks_path = output_dir / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_path)
        tasks_data = table.to_pylist()

        # Should have 2 unique tasks (Task 0 and Task 1)
        assert len(tasks_data) == 2
        assert any(t["task"] == "Task 0" for t in tasks_data)
        assert any(t["task"] == "Task 1" for t in tasks_data)

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_camera_name_mapping(self, tmp_path: Path, sample_episode: Episode):
        """Test custom camera name mapping."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(
            fps=30.0,
            camera_name_mapping={"camera0": "observation.images.front_cam"},
        )
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Check video is in mapped directory
        video_path = output_dir / "videos" / "observation.images.front_cam" / "chunk-000" / "file-000.mp4"
        assert video_path.exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_parquet_has_required_columns(self, tmp_path: Path, sample_episode: Episode):
        """Test parquet file has required columns."""
        import pyarrow.parquet as pq

        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0)
        writer = LeRobotV3Writer(config)

        writer.write_episode(sample_episode, output_dir, episode_index=0)
        writer._flush_chunk(output_dir)

        # Read parquet and check columns
        parquet_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(parquet_path)
        columns = set(table.column_names)

        assert "episode_index" in columns
        assert "frame_index" in columns
        assert "index" in columns
        assert "task_index" in columns
        assert "timestamp" in columns
        assert "observation.state" in columns
        assert "action" in columns

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_default_camera_name_stripping(self, tmp_path: Path):
        """Test that _image and _rgb suffixes are stripped by default."""
        config = LeRobotV3WriterConfig()
        writer = LeRobotV3Writer(config)

        assert writer._map_camera_name("agentview_image") == "observation.images.agentview"
        assert writer._map_camera_name("front_rgb") == "observation.images.front"
        assert writer._map_camera_name("wrist") == "observation.images.wrist"

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_info_json_structure(self, tmp_path: Path, sample_episodes: list[Episode]):
        """Test info.json has correct LeRobot v3 structure."""
        output_dir = tmp_path / "output"
        config = LeRobotV3WriterConfig(fps=30.0, robot_type="test_robot")
        writer = LeRobotV3Writer(config)

        writer.write_dataset(iter(sample_episodes), output_dir)

        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)

        # Check required fields
        assert "codebase_version" in info
        assert "robot_type" in info
        assert "total_episodes" in info
        assert "total_frames" in info
        assert "total_tasks" in info
        assert "chunks_size" in info
        assert "fps" in info
        assert "splits" in info
        assert "data_path" in info
        assert "video_path" in info
        assert "features" in info

        # Check features have correct structure
        features = info["features"]
        assert "episode_index" in features
        assert "frame_index" in features
        assert "timestamp" in features
        assert "action" in features

        # Check video features have video_info
        for key, feat in features.items():
            if feat.get("dtype") == "video":
                assert "video_info" in feat
                assert "video.fps" in feat["video_info"]
