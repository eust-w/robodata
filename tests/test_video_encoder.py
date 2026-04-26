"""Tests for video encoder."""

from pathlib import Path

import numpy as np
import pytest

from robodata.core.models import LazyImage
from robodata.video.encoder import VideoEncoder, VideoEncoderConfig, encode_video


def _check_av_available() -> bool:
    """Check if PyAV is available."""
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_frames():
    """Create sample LazyImage frames for testing."""

    def create_frame(index: int, height: int = 64, width: int = 64):
        """Create a LazyImage with gradient pattern."""

        def loader():
            # Create a gradient image with some variation based on index
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 10) % 256  # R
            frame[:, :, 1] = np.arange(height).reshape(-1, 1)  # G gradient
            frame[:, :, 2] = np.arange(width).reshape(1, -1)  # B gradient
            return frame

        return LazyImage(loader=loader, height=height, width=width, channels=3)

    return [create_frame(i) for i in range(30)]  # 1 second at 30fps


@pytest.fixture
def sample_numpy_frames():
    """Create sample numpy arrays for testing."""
    frames = []
    for i in range(30):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 10) % 256
        frame[:, :, 1] = np.arange(64).reshape(-1, 1)
        frame[:, :, 2] = np.arange(64).reshape(1, -1)
        frames.append(frame)
    return frames


class TestVideoEncoderConfig:
    """Tests for VideoEncoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VideoEncoderConfig()
        assert config.codec == "libx264"
        assert config.crf == 23
        assert config.preset == "medium"
        assert config.pixel_format == "yuv420p"
        assert config.gop_size == 12

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VideoEncoderConfig(
            codec="libx265",
            crf=18,
            preset="fast",
            pixel_format="yuv444p",
            gop_size=24,
        )
        assert config.codec == "libx265"
        assert config.crf == 18
        assert config.preset == "fast"
        assert config.pixel_format == "yuv444p"
        assert config.gop_size == 24


class TestVideoEncoder:
    """Tests for VideoEncoder."""

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_encode_lazy_images(self, tmp_path: Path, sample_frames):
        """Test encoding LazyImage frames to video."""
        output_path = tmp_path / "test_output.mp4"
        encoder = VideoEncoder()

        frame_count = encoder.encode_frames(
            iter(sample_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        assert frame_count == 30
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_encode_numpy_arrays(self, tmp_path: Path, sample_numpy_frames):
        """Test encoding numpy arrays to video."""
        output_path = tmp_path / "test_numpy_output.mp4"
        encoder = VideoEncoder()

        frame_count = encoder.encode_from_arrays(
            iter(sample_numpy_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        assert frame_count == 30
        assert output_path.exists()

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_encode_with_custom_config(self, tmp_path: Path, sample_frames):
        """Test encoding with custom configuration."""
        output_path = tmp_path / "test_custom.mp4"
        config = VideoEncoderConfig(
            crf=28,  # Lower quality, smaller file
            preset="ultrafast",  # Faster encoding
        )
        encoder = VideoEncoder(config)

        frame_count = encoder.encode_frames(
            iter(sample_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        assert frame_count == 30
        assert output_path.exists()

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_encode_empty_frames_raises(self, tmp_path: Path):
        """Test that encoding empty frames raises ValueError."""
        output_path = tmp_path / "empty.mp4"
        encoder = VideoEncoder()

        with pytest.raises(ValueError, match="No frames provided"):
            encoder.encode_frames(
                iter([]),
                output_path,
                fps=30.0,
                width=64,
                height=64,
            )

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_creates_parent_directories(self, tmp_path: Path, sample_frames):
        """Test that encoder creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dirs" / "video.mp4"
        encoder = VideoEncoder()

        encoder.encode_frames(
            iter(sample_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        assert output_path.exists()

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_clears_lazy_image_cache(self, tmp_path: Path, sample_frames):
        """Test that encoder clears LazyImage cache after encoding each frame."""
        output_path = tmp_path / "cache_test.mp4"
        encoder = VideoEncoder()

        encoder.encode_frames(
            iter(sample_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        # After encoding, all caches should be cleared
        for frame in sample_frames:
            assert not frame.is_loaded


class TestEncodeVideoFunction:
    """Tests for the encode_video convenience function."""

    @pytest.mark.skipif(
        not _check_av_available(),
        reason="PyAV not installed",
    )
    def test_encode_video_function(self, tmp_path: Path, sample_frames):
        """Test the encode_video convenience function."""
        output_path = tmp_path / "convenience.mp4"

        frame_count = encode_video(
            iter(sample_frames),
            output_path,
            fps=30.0,
            width=64,
            height=64,
        )

        assert frame_count == 30
        assert output_path.exists()
