"""Core domain models and protocols for Robodata."""

from robodata.core.exceptions import (
    RobodataError,
    FormatDetectionError,
    InspectionError,
    UnsupportedFormatError,
    ValidationError,
)
from robodata.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyArray,
    LazyImage,
)
from robodata.core.protocols import FormatReader, FormatWriter, VideoDecoder, VideoEncoder

__all__ = [
    # Models
    "CameraInfo",
    "DatasetInfo",
    "Dtype",
    "Episode",
    "FieldSchema",
    "Frame",
    "LazyArray",
    "LazyImage",
    # Protocols
    "FormatReader",
    "FormatWriter",
    "VideoDecoder",
    "VideoEncoder",
    # Exceptions
    "RobodataError",
    "FormatDetectionError",
    "InspectionError",
    "UnsupportedFormatError",
    "ValidationError",
]
