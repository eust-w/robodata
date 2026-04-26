"""Format readers and writers for Robodata.

This module provides the format registry and auto-import of format implementations.
"""

from robodata.formats.registry import FormatRegistry

__all__ = ["FormatRegistry"]


def _register_formats() -> None:
    """Import format modules to trigger registration decorators."""
    # Import format modules - this triggers @FormatRegistry.register_* decorators
    try:
        from robodata.formats import rlds  # noqa: F401
    except ImportError:
        pass  # RLDS dependencies not installed

    try:
        from robodata.formats import lerobot_v2  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import lerobot_v3  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import zarr  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import rosbag  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import hdf5  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import robodm  # noqa: F401
    except ImportError:
        pass

    try:
        from robodata.formats import groot  # noqa: F401
    except ImportError:
        pass


# Register formats on module import
_register_formats()
