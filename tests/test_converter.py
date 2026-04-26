"""Tests for the Converter facade."""

import json
from pathlib import Path

import numpy as np
import pytest

from robodata.convert.converter import (
    ConversionConfig,
    ConversionResult,
    Converter,
    convert,
)
from robodata.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
from robodata.formats.registry import FormatRegistry


def _check_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class MockReader:
    """Mock reader for testing."""

    def __init__(self):
        self._episodes = []

    @property
    def format_name(self) -> str:
        return "mock"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return (path / ".mock_dataset").exists()

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        return "1.0"

    def inspect(self, path: Path) -> DatasetInfo:
        return DatasetInfo(
            path=path,
            format="mock",
            format_version="1.0",
            num_episodes=3,
            total_frames=90,
            inferred_fps=30.0,
            inferred_robot_type="mock_robot",
            cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
        )

    def read_episodes(self, path: Path):
        """Yield mock episodes."""
        for i in range(3):
            yield self._create_episode(i)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode."""
        idx = int(episode_id.split("_")[1])
        return self._create_episode(idx)

    def _create_episode(self, idx: int) -> Episode:
        """Create a mock episode."""

        def frame_loader():
            for j in range(30):
                yield self._create_frame(j)

        return Episode(
            episode_id=f"ep_{idx:03d}",
            language_instruction=f"Task {idx % 2}",
            cameras={"camera0": CameraInfo(name="camera0", height=64, width=64)},
            fps=30.0,
            _frame_loader=frame_loader,
        )

    def _create_frame(self, idx: int) -> Frame:
        """Create a mock frame."""

        def img_loader():
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = (idx * 10) % 256
            return img

        return Frame(
            index=idx,
            timestamp=idx / 30.0,
            images={
                "camera0": LazyImage(
                    loader=img_loader,
                    height=64,
                    width=64,
                    channels=3,
                ),
            },
            state=np.array([0.1] * 7, dtype=np.float32),
            action=np.array([0.01] * 7, dtype=np.float32),
        )


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock dataset directory."""
    dataset_path = tmp_path / "mock_dataset"
    dataset_path.mkdir()
    (dataset_path / ".mock_dataset").touch()  # Marker file
    return dataset_path


@pytest.fixture
def register_mock_reader():
    """Register mock reader temporarily."""
    FormatRegistry._readers["mock"] = MockReader
    yield
    del FormatRegistry._readers["mock"]


class TestConversionConfig:
    """Tests for ConversionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()
        assert config.target_format is None  # Default is None (must be specified)
        assert config.fps is None
        assert config.robot_type is None
        assert config.fail_on_error is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            target_format="lerobot-v3",
            fps=60.0,
            robot_type="franka",
            camera_mapping={"cam0": "observation.images.front"},
            fail_on_error=True,
        )
        assert config.fps == 60.0
        assert config.robot_type == "franka"
        assert config.fail_on_error is True


class TestConversionResult:
    """Tests for ConversionResult."""

    def test_success_result(self):
        """Test successful conversion result."""
        result = ConversionResult(
            success=True,
            source_format="mock",
            target_format="lerobot-v3",
            episodes_converted=10,
            episodes_failed=0,
            total_frames=300,
            output_path=Path("/output"),
        )
        assert result.success is True
        assert result.episodes_converted == 10
        assert result.episodes_failed == 0

    def test_partial_failure_result(self):
        """Test partial failure result."""
        result = ConversionResult(
            success=False,
            source_format="mock",
            target_format="lerobot-v3",
            episodes_converted=8,
            episodes_failed=2,
            errors=["Episode 3: error", "Episode 7: error"],
        )
        assert result.success is False
        assert result.episodes_converted == 8
        assert result.episodes_failed == 2
        assert len(result.errors) == 2


class TestConverter:
    """Tests for Converter class."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_convert_mock_to_lerobot_v3(
        self, tmp_path: Path, mock_dataset: Path, register_mock_reader
    ):
        """Test converting mock dataset to LeRobot v3."""
        output_path = tmp_path / "output"

        config = ConversionConfig(target_format="lerobot-v3", fps=30.0)
        converter = Converter(config)

        result = converter.convert(
            mock_dataset,
            output_path,
            source_format="mock",
        )

        assert result.success is True
        assert result.source_format == "mock"
        assert result.target_format == "lerobot-v3"
        assert result.episodes_converted == 3
        assert result.episodes_failed == 0

        # Check output structure (LeRobot v3 chunked format)
        assert (output_path / "meta" / "info.json").exists()
        assert (output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
        assert (output_path / "meta" / "tasks.parquet").exists()
        assert (output_path / "data" / "chunk-000").exists()
        assert (output_path / "videos").exists()

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_convert_with_progress_callback(
        self, tmp_path: Path, mock_dataset: Path, register_mock_reader
    ):
        """Test conversion with progress callback."""
        output_path = tmp_path / "output"

        progress_calls = []

        def progress_callback(stage: str, current: int, total: int):
            progress_calls.append((stage, current, total))

        config = ConversionConfig(target_format="lerobot-v3")
        converter = Converter(config)

        result = converter.convert(
            mock_dataset,
            output_path,
            source_format="mock",
            progress_callback=progress_callback,
        )

        assert result.success is True

        # Check progress callbacks were called
        stages = [call[0] for call in progress_calls]
        assert "inspect" in stages
        assert "episode" in stages
        assert "finalize" in stages

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_convert_with_config_overrides(
        self, tmp_path: Path, mock_dataset: Path, register_mock_reader
    ):
        """Test conversion with FPS and robot type overrides."""
        output_path = tmp_path / "output"

        config = ConversionConfig(
            target_format="lerobot-v3",
            fps=60.0,
            robot_type="custom_robot",
        )
        converter = Converter(config)

        result = converter.convert(
            mock_dataset,
            output_path,
            source_format="mock",
        )

        assert result.success is True

        # Check metadata has overridden values
        with open(output_path / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["fps"] == 60.0
        assert info["robot_type"] == "custom_robot"

    def test_convert_unsupported_format(self, tmp_path: Path):
        """Test conversion with unsupported format raises error when fail_on_error=True."""
        from robodata.core.exceptions import UnsupportedFormatError

        config = ConversionConfig(target_format="unsupported-format", fail_on_error=True)
        converter = Converter(config)

        with pytest.raises(UnsupportedFormatError):
            converter.convert(
                tmp_path,
                tmp_path / "output",
                source_format="mock",
                target_format="unsupported-format",
            )

    def test_convert_unsupported_format_no_raise(self, tmp_path: Path, register_mock_reader):
        """Test conversion with unsupported writer returns error result when fail_on_error=False."""
        config = ConversionConfig(target_format="unsupported-format", fail_on_error=False)
        converter = Converter(config)

        result = converter.convert(
            tmp_path,
            tmp_path / "output",
            source_format="mock",
            target_format="unsupported-format",
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "No writer for format" in result.errors[0]

    def test_convert_nonexistent_source(self, tmp_path: Path, register_mock_reader):
        """Test conversion with nonexistent source returns error result."""
        config = ConversionConfig(fail_on_error=False)
        converter = Converter(config)

        result = converter.convert(
            tmp_path / "nonexistent",
            tmp_path / "output",
        )

        assert result.success is False
        assert len(result.errors) > 0


class TestConvertFunction:
    """Tests for the convert convenience function."""

    @pytest.mark.skipif(
        not _check_dependencies_available(),
        reason="PyAV or PyArrow not installed",
    )
    def test_convert_function(
        self, tmp_path: Path, mock_dataset: Path, register_mock_reader
    ):
        """Test the convert convenience function."""
        output_path = tmp_path / "output"

        result = convert(
            mock_dataset,
            output_path,
            target_format="lerobot-v3",
            source_format="mock",
            fps=30.0,
        )

        assert result.success is True
        assert result.episodes_converted == 3
