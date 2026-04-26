"""Tests for RLDS format writer."""

import json
from pathlib import Path

import numpy as np
import pytest

from robodata.core.models import (
    CameraInfo,
    DatasetInfo,
    Episode,
    Frame,
    LazyImage,
)


def _check_tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except ImportError:
        return False


def _create_mock_episode(episode_id: str, num_frames: int = 10) -> Episode:
    """Create a mock episode for testing."""
    def load_frames():
        for i in range(num_frames):
            # Create a lazy image loader
            def make_loader(idx=i):
                return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

            images = {
                "camera": LazyImage(
                    loader=make_loader,
                    height=64,
                    width=64,
                    channels=3,
                )
            }

            yield Frame(
                index=i,
                images=images,
                state=np.random.randn(7).astype(np.float32),
                action=np.random.randn(7).astype(np.float32),
                is_first=(i == 0),
                is_last=(i == num_frames - 1),
            )

    return Episode(
        episode_id=episode_id,
        language_instruction="Pick up the object",
        _frame_loader=load_frames,
    )


@pytest.mark.skipif(not _check_tensorflow_available(), reason="TensorFlow not installed")
class TestRLDSWriter:
    """Test RLDSWriter class."""

    def test_write_single_episode(self, tmp_path: Path):
        """Test writing a single episode."""
        from robodata.formats.rlds.writer import RLDSWriter, RLDSWriterConfig

        config = RLDSWriterConfig(
            dataset_name="test_dataset",
            fps=30.0,
            episodes_per_shard=5,
        )
        writer = RLDSWriter(config)

        episode = _create_mock_episode("ep_0", num_frames=5)
        writer.write_episode(episode, tmp_path, episode_index=0)

        # Finalize
        dataset_info = DatasetInfo(
            path=tmp_path,
            format="rlds",
            num_episodes=1,
            total_frames=5,
        )
        writer.finalize(tmp_path, dataset_info)

        # Check output files exist
        assert (tmp_path / "dataset_info.json").exists()
        assert (tmp_path / "features.json").exists()

        # Check TFRecord files
        tfrecord_files = list(tmp_path.glob("*.tfrecord*"))
        assert len(tfrecord_files) >= 1

    def test_write_dataset(self, tmp_path: Path):
        """Test writing multiple episodes."""
        from robodata.formats.rlds.writer import RLDSWriter, RLDSWriterConfig

        config = RLDSWriterConfig(
            dataset_name="test_multi",
            fps=30.0,
            episodes_per_shard=2,
        )
        writer = RLDSWriter(config)

        def episodes():
            for i in range(5):
                yield _create_mock_episode(f"ep_{i}", num_frames=3)

        writer.write_dataset(episodes(), tmp_path)

        # Check metadata
        assert (tmp_path / "dataset_info.json").exists()
        with open(tmp_path / "dataset_info.json") as f:
            info = json.load(f)
        
        assert info["name"] == "test_multi"
        assert "splits" in info
        assert len(info["splits"]) == 1
        assert info["splits"][0]["name"] == "train"

    def test_features_json_structure(self, tmp_path: Path):
        """Test that features.json has correct structure."""
        from robodata.formats.rlds.writer import RLDSWriter, RLDSWriterConfig

        config = RLDSWriterConfig(dataset_name="test_features")
        writer = RLDSWriter(config)

        episode = _create_mock_episode("ep_0", num_frames=3)
        writer.write_episode(episode, tmp_path, episode_index=0)

        dataset_info = DatasetInfo(
            path=tmp_path,
            format="rlds",
            num_episodes=1,
            total_frames=3,
        )
        writer.finalize(tmp_path, dataset_info)

        # Check features.json structure
        with open(tmp_path / "features.json") as f:
            features = json.load(f)

        assert "featuresDict" in features
        assert "features" in features["featuresDict"]
        assert "steps" in features["featuresDict"]["features"]
        assert "episode_metadata" in features["featuresDict"]["features"]

    @pytest.mark.skipif(not _check_tensorflow_available(), reason="TensorFlow not installed")
    def test_tfrecord_readable(self, tmp_path: Path):
        """Test that output TFRecords can be read back."""
        import tensorflow as tf
        from robodata.formats.rlds.writer import RLDSWriter, RLDSWriterConfig

        config = RLDSWriterConfig(dataset_name="test_readable")
        writer = RLDSWriter(config)

        episode = _create_mock_episode("ep_0", num_frames=5)
        writer.write_episode(episode, tmp_path, episode_index=0)

        dataset_info = DatasetInfo(
            path=tmp_path,
            format="rlds",
            num_episodes=1,
            total_frames=5,
        )
        writer.finalize(tmp_path, dataset_info)

        # Read TFRecord back
        tfrecord_files = list(tmp_path.glob("*.tfrecord*"))
        assert len(tfrecord_files) >= 1

        ds = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
        records = list(ds)
        assert len(records) >= 1

        # Parse first record
        example = tf.train.Example()
        example.ParseFromString(records[0].numpy())

        # Check expected keys
        keys = list(example.features.feature.keys())
        assert "steps/action" in keys
        assert "episode_metadata/file_path" in keys
