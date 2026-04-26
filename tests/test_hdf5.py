"""Tests for HDF5 format reader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from robodata.formats.hdf5.reader import HDF5Reader


@pytest.fixture
def robomimic_hdf5(tmp_path: Path) -> Path:
    """Create a mock robomimic-style HDF5 file."""
    h5py = pytest.importorskip("h5py")

    hdf5_path = tmp_path / "robomimic_test.hdf5"

    with h5py.File(hdf5_path, "w") as f:
        # Create data group
        data_group = f.create_group("data")
        data_group.attrs["total"] = 100  # 2 demos * 50 frames
        data_group.attrs["env_args"] = '{"env_kwargs": {"control_freq": 20, "robots": ["Panda"]}}'

        # Create two demo groups
        for demo_idx in range(2):
            demo = data_group.create_group(f"demo_{demo_idx}")
            demo.attrs["num_samples"] = 50

            # Actions
            demo.create_dataset("actions", data=np.random.randn(50, 7).astype(np.float64))

            # Observations
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_image",
                data=np.random.randint(0, 255, (50, 84, 84, 3), dtype=np.uint8),
            )
            obs.create_dataset("robot0_joint_pos", data=np.random.randn(50, 7).astype(np.float64))
            obs.create_dataset("robot0_eef_pos", data=np.random.randn(50, 3).astype(np.float64))

            # Rewards
            demo.create_dataset("rewards", data=np.random.randn(50).astype(np.float64))

        # Create mask group
        mask = f.create_group("mask")
        mask.create_dataset("train", data=[b"demo_0"])
        mask.create_dataset("valid", data=[b"demo_1"])

    return hdf5_path


@pytest.fixture
def aloha_hdf5(tmp_path: Path) -> Path:
    """Create a mock ALOHA-style HDF5 file."""
    h5py = pytest.importorskip("h5py")

    hdf5_path = tmp_path / "aloha_test.hdf5"

    with h5py.File(hdf5_path, "w") as f:
        # Action at root level
        f.create_dataset("action", data=np.random.randn(30, 14).astype(np.float64))

        # State at root level
        f.create_dataset("qpos", data=np.random.randn(30, 14).astype(np.float64))
        f.create_dataset("qvel", data=np.random.randn(30, 14).astype(np.float64))

        # Images group
        images = f.create_group("images")
        images.create_dataset(
            "cam_high",
            data=np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8),
        )
        images.create_dataset(
            "cam_left_wrist",
            data=np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8),
        )

    return hdf5_path


class TestHDF5Reader:
    """Test HDF5Reader class."""

    def test_can_read_hdf5_file(self, robomimic_hdf5: Path):
        """Test detection of HDF5 files."""
        assert HDF5Reader.can_read(robomimic_hdf5) is True

    def test_can_read_h5_file(self, tmp_path: Path):
        """Test detection of .h5 files."""
        h5py = pytest.importorskip("h5py")
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        assert HDF5Reader.can_read(h5_path) is True

    def test_can_read_directory_with_hdf5(self, robomimic_hdf5: Path):
        """Test detection of directories containing HDF5 files."""
        assert HDF5Reader.can_read(robomimic_hdf5.parent) is True

    def test_cannot_read_non_hdf5(self, tmp_path: Path):
        """Test rejection of non-HDF5 paths."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not hdf5")
        assert HDF5Reader.can_read(txt_file) is False
        assert HDF5Reader.can_read(tmp_path / "nonexistent") is False

    def test_detect_layout_robomimic(self, robomimic_hdf5: Path):
        """Test layout detection for robomimic style."""
        h5py = pytest.importorskip("h5py")
        with h5py.File(robomimic_hdf5, "r") as f:
            layout = HDF5Reader._detect_layout(f)
            assert layout == "robomimic"

    def test_detect_layout_aloha(self, aloha_hdf5: Path):
        """Test layout detection for ALOHA style."""
        h5py = pytest.importorskip("h5py")
        with h5py.File(aloha_hdf5, "r") as f:
            layout = HDF5Reader._detect_layout(f)
            assert layout == "aloha"


class TestRobomimicInspect:
    """Test inspection of robomimic-style HDF5."""

    def test_inspect_basic(self, robomimic_hdf5: Path):
        """Test basic inspection."""
        reader = HDF5Reader()
        info = reader.inspect(robomimic_hdf5)

        assert info.format == "hdf5"
        assert info.format_version == "robomimic"
        assert info.num_episodes == 2
        assert info.total_frames == 100

    def test_inspect_schema(self, robomimic_hdf5: Path):
        """Test schema inference."""
        reader = HDF5Reader()
        info = reader.inspect(robomimic_hdf5)

        # Check action schema
        assert info.action_schema is not None
        assert info.action_schema.shape == (7,)

        # Check cameras
        assert "agentview" in info.cameras
        assert info.cameras["agentview"].height == 84
        assert info.cameras["agentview"].width == 84

        # Check observations
        assert "robot0_joint_pos" in info.observation_schema
        assert "robot0_eef_pos" in info.observation_schema

    def test_inspect_metadata(self, robomimic_hdf5: Path):
        """Test metadata extraction."""
        reader = HDF5Reader()
        info = reader.inspect(robomimic_hdf5)

        assert info.inferred_fps == 20
        assert info.inferred_robot_type == "Panda"


class TestAlohaInspect:
    """Test inspection of ALOHA-style HDF5."""

    def test_inspect_basic(self, aloha_hdf5: Path):
        """Test basic inspection."""
        reader = HDF5Reader()
        info = reader.inspect(aloha_hdf5)

        assert info.format == "hdf5"
        assert info.format_version == "aloha"
        assert info.num_episodes == 1  # Single file
        assert info.total_frames == 30

    def test_inspect_schema(self, aloha_hdf5: Path):
        """Test schema inference."""
        reader = HDF5Reader()
        info = reader.inspect(aloha_hdf5)

        # Check action
        assert info.action_schema is not None
        assert info.action_schema.shape == (14,)

        # Check cameras
        assert "cam_high" in info.cameras
        assert "cam_left_wrist" in info.cameras
        assert info.cameras["cam_high"].height == 480
        assert info.cameras["cam_high"].width == 640

        # Check observations
        assert "qpos" in info.observation_schema
        assert "qvel" in info.observation_schema


class TestRobomimicReadEpisodes:
    """Test episode reading for robomimic-style HDF5."""

    def test_read_episodes_count(self, robomimic_hdf5: Path):
        """Test reading correct number of episodes."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(robomimic_hdf5))
        assert len(episodes) == 2

    def test_read_episodes_ids(self, robomimic_hdf5: Path):
        """Test episode IDs."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(robomimic_hdf5))
        assert episodes[0].episode_id == "0"
        assert episodes[1].episode_id == "1"

    def test_read_frames(self, robomimic_hdf5: Path):
        """Test frame loading."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(robomimic_hdf5))

        frames = list(episodes[0].frames())
        assert len(frames) == 50

        # Check first frame
        frame = frames[0]
        assert frame.index == 0
        assert frame.is_first is True
        assert frame.action is not None
        assert frame.action.shape == (7,)

    def test_read_images(self, robomimic_hdf5: Path):
        """Test image loading."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(robomimic_hdf5))

        frames = list(episodes[0].frames())
        frame = frames[0]

        assert "agentview" in frame.images
        img = frame.images["agentview"].load()
        assert img.shape == (84, 84, 3)
        assert img.dtype == np.uint8


class TestAlohaReadEpisodes:
    """Test episode reading for ALOHA-style HDF5."""

    def test_read_single_episode(self, aloha_hdf5: Path):
        """Test reading single file as one episode."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(aloha_hdf5))
        assert len(episodes) == 1

    def test_read_frames(self, aloha_hdf5: Path):
        """Test frame loading."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(aloha_hdf5))

        frames = list(episodes[0].frames())
        assert len(frames) == 30

        # Check frame content
        frame = frames[0]
        assert frame.action is not None
        assert frame.action.shape == (14,)
        assert frame.state is not None
        assert frame.state.shape == (14,)  # qpos

    def test_read_images(self, aloha_hdf5: Path):
        """Test image loading."""
        reader = HDF5Reader()
        episodes = list(reader.read_episodes(aloha_hdf5))

        frames = list(episodes[0].frames())
        frame = frames[0]

        assert "cam_high" in frame.images
        assert "cam_left_wrist" in frame.images

        img = frame.images["cam_high"].load()
        assert img.shape == (480, 640, 3)


class TestMultiFileAloha:
    """Test reading multiple ALOHA HDF5 files."""

    def test_read_multiple_files(self, tmp_path: Path):
        """Test reading directory with multiple episode files."""
        h5py = pytest.importorskip("h5py")

        # Create multiple episode files
        for ep_idx in range(3):
            hdf5_path = tmp_path / f"episode_{ep_idx}.hdf5"
            with h5py.File(hdf5_path, "w") as f:
                f.create_dataset("action", data=np.random.randn(10, 7).astype(np.float64))
                f.create_dataset("qpos", data=np.random.randn(10, 7).astype(np.float64))

        reader = HDF5Reader()
        episodes = list(reader.read_episodes(tmp_path))

        assert len(episodes) == 3
        assert episodes[0].episode_id == "0"
        assert episodes[1].episode_id == "1"
        assert episodes[2].episode_id == "2"


class TestReadEpisodeById:
    """Test read_episode method."""

    def test_read_specific_episode(self, robomimic_hdf5: Path):
        """Test reading a specific episode by ID."""
        reader = HDF5Reader()
        episode = reader.read_episode(robomimic_hdf5, "1")
        assert episode.episode_id == "1"

    def test_read_nonexistent_episode(self, robomimic_hdf5: Path):
        """Test error on nonexistent episode."""
        from robodata.core.exceptions import EpisodeNotFoundError

        reader = HDF5Reader()
        with pytest.raises(EpisodeNotFoundError):
            reader.read_episode(robomimic_hdf5, "999")
