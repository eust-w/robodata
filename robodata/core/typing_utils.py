"""Common typing utilities and imports for Robodata."""

from __future__ import annotations

from typing import Any

# Common imports used throughout the codebase
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


def ensure_numpy() -> Any:
    """Ensure numpy is available and return the module.
    
    Raises:
        ImportError: If numpy is not available.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required but not installed. "
            "Install with: pip install numpy"
        )
    return np


# Common type aliases
try:
    from numpy.typing import NDArray
    ArrayLike = NDArray[Any]
except ImportError:
    # Fallback type hints when numpy is not available
    ArrayLike = Any
