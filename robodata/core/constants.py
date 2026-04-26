"""Constants and configuration values for Robodata."""

from __future__ import annotations

# Image processing constants
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_CHANNELS = 3
IMAGE_UINT8_SCALE = 255

# Visualization constants
DEFAULT_MAX_EPISODES = 50
DEFAULT_PLOT_HEIGHT = 120
DEFAULT_DPI = 100
PLOT_DPI_HIGH = 150
DEFAULT_FPS = 10.0
MAX_SPEED_MULTIPLIER = 8.0
SPEED_MULTIPLIER_STEP = 1.5

# Web viewer constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0  # Auto-assign
CACHE_MAX_AGE = 3600  # seconds
HTTP_STATUS_OK = 200
HTTP_STATUS_NOT_FOUND = 404

# Cache and memory constants
DEFAULT_CACHE_SIZE = 100
DEFAULT_CACHE_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_IMAGE_CACHE_SIZE = 100
DEFAULT_IMAGE_CACHE_SIZE_BYTES = 512 * 1024 * 1024  # 512MB
DEFAULT_MEMORY_LIMIT_MB = 4096  # 4GB
MAX_WORKERS_CPU = 8
MAX_WORKERS_IO = 16

# Parallel processing constants
DEFAULT_CHUNK_SIZE = 1
DEFAULT_TIMEOUT = None
DEFAULT_MAX_RETRIES = 0

# Font and UI constants
FONT_SIZE_SMALL = 0.4
FONT_SIZE_NORMAL = 0.6
FONT_SIZE_LARGE = 1.1
FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_BOLD = 600

# Color constants (RGB)
COLOR_RED = (0, 0, 255)  # OpenCV uses BGR
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (200, 200, 200)
COLOR_BLACK = (0, 0, 0)

# File size limits
MAX_FILE_SIZE_MB = 1000
MAX_MEMORY_USAGE_RATIO = 0.8

# Quality thresholds
DEFAULT_MIN_QUALITY = 0.5
MAX_QUALITY = 1.0
MIN_QUALITY = 0.0

# Dataset limits
MAX_EPISODES_FOR_PREVIEW = 50
MAX_FRAMES_FOR_PREVIEW = 1000

# Logging and debugging
DEFAULT_LOG_LEVEL = "INFO"
DEBUG_MODE = False

# Version compatibility
MIN_PYTHON_VERSION = (3, 8)
SUPPORTED_FORMATS = [
    "rlds",
    "lerobot-v2", 
    "lerobot-v3",
    "zarr",
    "rosbag",
    "hdf5",
    "robodm",
    "groot"
]

# Export all constants for easy import
__all__ = [
    # Image processing
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH", 
    "DEFAULT_IMAGE_CHANNELS",
    "IMAGE_UINT8_SCALE",
    
    # Visualization
    "DEFAULT_MAX_EPISODES",
    "DEFAULT_PLOT_HEIGHT",
    "DEFAULT_DPI",
    "PLOT_DPI_HIGH",
    "DEFAULT_FPS",
    "MAX_SPEED_MULTIPLIER",
    "SPEED_MULTIPLIER_STEP",
    
    # Web viewer
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "CACHE_MAX_AGE",
    "HTTP_STATUS_OK",
    "HTTP_STATUS_NOT_FOUND",
    
    # Cache and memory
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_CACHE_SIZE_BYTES",
    "DEFAULT_IMAGE_CACHE_SIZE",
    "DEFAULT_IMAGE_CACHE_SIZE_BYTES",
    "DEFAULT_MEMORY_LIMIT_MB",
    "MAX_WORKERS_CPU",
    "MAX_WORKERS_IO",
    
    # Parallel processing
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    
    # Font and UI
    "FONT_SIZE_SMALL",
    "FONT_SIZE_NORMAL", 
    "FONT_SIZE_LARGE",
    "FONT_WEIGHT_NORMAL",
    "FONT_WEIGHT_BOLD",
    
    # Colors
    "COLOR_RED",
    "COLOR_WHITE",
    "COLOR_GRAY",
    "COLOR_BLACK",
    
    # File size limits
    "MAX_FILE_SIZE_MB",
    "MAX_MEMORY_USAGE_RATIO",
    
    # Quality thresholds
    "DEFAULT_MIN_QUALITY",
    "MAX_QUALITY",
    "MIN_QUALITY",
    
    # Dataset limits
    "MAX_EPISODES_FOR_PREVIEW",
    "MAX_FRAMES_FOR_PREVIEW",
    
    # Logging
    "DEFAULT_LOG_LEVEL",
    "DEBUG_MODE",
    
    # Version compatibility
    "MIN_PYTHON_VERSION",
    "SUPPORTED_FORMATS",
]
