"""Video encoding/decoding module for Robodata.

Provides PyAV-based video encoding for converting robotics image
sequences to MP4 format.
"""

from robodata.video.encoder import VideoEncoder, VideoEncoderConfig, encode_video

__all__ = ["VideoEncoder", "VideoEncoderConfig", "encode_video"]
