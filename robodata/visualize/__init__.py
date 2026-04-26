"""Visualization tools for Robodata."""

from robodata.visualize.unified_viewer import UnifiedBackend, UnifiedViewer, unified_visualize
from robodata.visualize.web_viewer import WebBackend, WebViewer, web_visualize

__all__ = [
    "UnifiedBackend",
    "UnifiedViewer",
    "unified_visualize",
    "WebBackend",
    "WebViewer",
    "web_visualize",
]

# Optional fast viewer (requires opencv-python)
try:
    from robodata.visualize.cv_viewer import CVBackend, CVViewer, cv_visualize
    __all__.extend(["CVBackend", "CVViewer", "cv_visualize"])
except ImportError:
    pass
