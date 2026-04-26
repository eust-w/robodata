"""Common utilities for format readers and writers."""

from __future__ import annotations

from robodata.core.typing_utils import ensure_numpy
from robodata.core.models import Dtype


def numpy_dtype_to_dtype(dtype: Any) -> Dtype:
    """Convert numpy dtype to Robodata Dtype.
    
    Args:
        dtype: numpy dtype object
        
    Returns:
        Corresponding Robodata Dtype enum value
    """
    np = ensure_numpy()

    if dtype == np.float32:
        return Dtype.FLOAT32
    elif dtype == np.float64:
        return Dtype.FLOAT64
    elif dtype == np.int8:
        return Dtype.INT8
    elif dtype == np.int16:
        return Dtype.INT16
    elif dtype == np.int32:
        return Dtype.INT32
    elif dtype == np.int64:
        return Dtype.INT64
    elif dtype == np.uint8:
        return Dtype.UINT8
    elif dtype == np.uint16:
        return Dtype.UINT16
    elif dtype == np.uint32:
        return Dtype.UINT32
    elif dtype == np.uint64:
        return Dtype.UINT64
    elif dtype == np.bool_:
        return Dtype.BOOL
    else:
        return Dtype.UNKNOWN


def extract_camera_name_from_key(key: str) -> str:
    """Extract camera name from array key or topic.
    
    Common patterns:
    - camera0_rgb, observation.images.camera0 -> camera0
    - /camera/color/image_raw -> camera_color
    
    Args:
        key: Array key or topic path
        
    Returns:
        Normalized camera name
    """
    # Handle ROS topics
    if key.startswith("/"):
        parts = key.strip("/").split("/")
        # Remove common suffixes
        if parts and parts[-1] in ["image_raw", "compressed", "image_rect"]:
            parts = parts[:-1]
        if parts and parts[-1] in ["color", "mono", "depth"]:
            return f"{parts[0]}_{parts[-1]}"
        return parts[0] if parts else key
    
    # Handle array keys
    key_lower = key.lower()
    
    # Remove common prefixes
    for prefix in ["observation.images.", "observation/", "steps/observation/", "camera_"]:
        if key_lower.startswith(prefix):
            key = key[len(prefix):]
            break
    
    # Remove common suffixes
    for suffix in ["_rgb", "_image", "_img", "_frame"]:
        if key_lower.endswith(suffix):
            key = key[:-len(suffix)]
            break
    
    # Clean up separators
    return key.replace("/", "_").replace(".", "_")
