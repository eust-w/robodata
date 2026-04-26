"""Centralized configuration management for Robodata."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robodata.core.constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_SIZE_BYTES,
    DEFAULT_IMAGE_CACHE_SIZE,
    DEFAULT_IMAGE_CACHE_SIZE_BYTES,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_MAX_EPISODES,
    MAX_WORKERS_CPU,
    MAX_WORKERS_IO,
    DEFAULT_FPS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_CHANNELS,
)


@dataclass
class CacheConfig:
    """Configuration for caching systems."""
    max_size: int = DEFAULT_CACHE_SIZE
    max_size_bytes: int = DEFAULT_CACHE_SIZE_BYTES
    image_max_size: int = DEFAULT_IMAGE_CACHE_SIZE
    image_max_size_bytes: int = DEFAULT_IMAGE_CACHE_SIZE_BYTES
    enable_compression: bool = True
    cleanup_interval: int = 300  # seconds


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    max_workers_cpu: int = MAX_WORKERS_CPU
    max_workers_io: int = MAX_WORKERS_IO
    chunk_size: int = 1
    enable_parallel: bool = True
    auto_cleanup: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    max_episodes: int = DEFAULT_MAX_EPISODES
    default_fps: float = DEFAULT_FPS
    default_height: int = DEFAULT_IMAGE_HEIGHT
    default_width: int = DEFAULT_IMAGE_WIDTH
    default_channels: int = DEFAULT_IMAGE_CHANNELS
    enable_caching: bool = True
    plot_quality: str = "medium"  # low, medium, high


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Path | None = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class RobodataConfig:
    """Main configuration container for Robodata."""
    
    # Sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    debug_mode: bool = False
    temp_dir: Path | None = None
    data_dir: Path | None = None
    
    @classmethod
    def from_env(cls) -> "RobodataConfig":
        """Create configuration from environment variables.
        
        Returns:
            RobodataConfig with values loaded from environment
        """
        config = cls()
        
        # Cache settings
        if cache_size := os.getenv("ROBODATA_CACHE_SIZE"):
            config.cache.max_size = int(cache_size)
        if cache_bytes := os.getenv("ROBODATA_CACHE_SIZE_BYTES"):
            config.cache.max_size_bytes = int(cache_bytes)
        if compression := os.getenv("ROBODATA_CACHE_COMPRESSION"):
            config.cache.enable_compression = compression.lower() in ("true", "1", "yes")
        
        # Performance settings
        if memory_limit := os.getenv("ROBODATA_MEMORY_LIMIT_MB"):
            config.performance.memory_limit_mb = int(memory_limit)
        if max_workers := os.getenv("ROBODATA_MAX_WORKERS"):
            config.performance.max_workers_cpu = min(int(max_workers), MAX_WORKERS_CPU)
        if parallel := os.getenv("ROBODATA_ENABLE_PARALLEL"):
            config.performance.enable_parallel = parallel.lower() in ("true", "1", "yes")
        
        # Visualization settings
        if max_episodes := os.getenv("ROBODATA_MAX_EPISODES"):
            config.visualization.max_episodes = int(max_episodes)
        if fps := os.getenv("ROBODATA_DEFAULT_FPS"):
            config.visualization.default_fps = float(fps)
        
        # Logging settings
        if log_level := os.getenv("ROBODATA_LOG_LEVEL"):
            config.logging.level = log_level.upper()
        if log_file := os.getenv("ROBODATA_LOG_FILE"):
            config.logging.file_path = Path(log_file)
        
        # Global settings
        if debug := os.getenv("ROBODATA_DEBUG"):
            config.debug_mode = debug.lower() in ("true", "1", "yes")
        if temp_dir := os.getenv("ROBODATA_TEMP_DIR"):
            config.temp_dir = Path(temp_dir)
        if data_dir := os.getenv("ROBODATA_DATA_DIR"):
            config.data_dir = Path(data_dir)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> "RobodataConfig":
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            RobodataConfig loaded from file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            return cls.from_dict(data or {})
        except ImportError:
            raise ValueError("PyYAML is required for configuration file support")
        except Exception as e:
            raise ValueError(f"Invalid configuration file: {e}")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobodataConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            RobodataConfig created from dictionary
        """
        # Extract sub-configurations
        cache_data = data.get('cache', {})
        performance_data = data.get('performance', {})
        viz_data = data.get('visualization', {})
        logging_data = data.get('logging', {})
        
        return cls(
            cache=CacheConfig(**cache_data),
            performance=PerformanceConfig(**performance_data),
            visualization=VisualizationConfig(**viz_data),
            logging=LoggingConfig(**logging_data),
            debug_mode=data.get('debug_mode', False),
            temp_dir=Path(data['temp_dir']) if data.get('temp_dir') else None,
            data_dir=Path(data['data_dir']) if data.get('data_dir') else None,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'cache': {
                'max_size': self.cache.max_size,
                'max_size_bytes': self.cache.max_size_bytes,
                'image_max_size': self.cache.image_max_size,
                'image_max_size_bytes': self.cache.image_max_size_bytes,
                'enable_compression': self.cache.enable_compression,
                'cleanup_interval': self.cache.cleanup_interval,
            },
            'performance': {
                'memory_limit_mb': self.performance.memory_limit_mb,
                'max_workers_cpu': self.performance.max_workers_cpu,
                'max_workers_io': self.performance.max_workers_io,
                'chunk_size': self.performance.chunk_size,
                'enable_parallel': self.performance.enable_parallel,
                'auto_cleanup': self.performance.auto_cleanup,
            },
            'visualization': {
                'max_episodes': self.visualization.max_episodes,
                'default_fps': self.visualization.default_fps,
                'default_height': self.visualization.default_height,
                'default_width': self.visualization.default_width,
                'default_channels': self.visualization.default_channels,
                'enable_caching': self.visualization.enable_caching,
                'plot_quality': self.visualization.plot_quality,
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': str(self.logging.file_path) if self.logging.file_path else None,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count,
                'enable_console': self.logging.enable_console,
            },
            'debug_mode': self.debug_mode,
            'temp_dir': str(self.temp_dir) if self.temp_dir else None,
            'data_dir': str(self.data_dir) if self.data_dir else None,
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        try:
            import yaml
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except ImportError:
            raise ValueError("PyYAML is required for configuration file support")
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory for Robodata.
        
        Returns:
            Path to temporary directory
        """
        if self.temp_dir:
            return self.temp_dir
        else:
            return Path.home() / ".robodata" / "temp"
    
    def get_data_dir(self) -> Path:
        """Get data directory for Robodata.
        
        Returns:
            Path to data directory
        """
        if self.data_dir:
            return self.data_dir
        else:
            return Path.home() / ".robodata" / "data"


# Global configuration instance
_global_config: RobodataConfig | None = None


def get_config() -> RobodataConfig:
    """Get the global configuration instance.
    
    Returns:
        Global RobodataConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = RobodataConfig.from_env()
    return _global_config


def set_config(config: RobodataConfig) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Configuration to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = RobodataConfig.from_env()


# Context manager for temporary configuration changes
class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, **kwargs):
        """Initialize config context.
        
        Args:
            **kwargs: Configuration changes to apply temporarily
        """
        self.changes = kwargs
        self.original_config: RobodataConfig | None = None
    
    def __enter__(self) -> RobodataConfig:
        """Apply temporary configuration changes."""
        self.original_config = get_config()
        
        # Create new config with changes
        current_dict = self.original_config.to_dict()
        self._deep_update(current_dict, self.changes)
        
        new_config = RobodataConfig.from_dict(current_dict)
        set_config(new_config)
        return new_config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original configuration."""
        if self.original_config:
            set_config(self.original_config)
        return False
    
    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
