"""Enhanced configuration management system for Robodata.

Provides hierarchical configuration, validation, environment variable support,
and configuration merging capabilities.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from robodata.core.exceptions import RobodataError
from robodata.core.logging_system import get_logger

T = TypeVar("T")


class ConfigSource(Enum):
    """Sources of configuration values."""
    DEFAULT = "default"
    FILE = "file"
    ENV = "environment"
    CLI = "command_line"


@dataclass
class ConfigField:
    """Metadata for a configuration field."""
    name: str
    type: Type
    default: Any
    description: str
    env_var: Optional[str] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    required: bool = False
    secret: bool = False  # For sensitive values like API keys


class ConfigValidationError(RobodataError):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.config_file = config_file or self._get_default_config_file()
        self._config_data: Dict[str, Any] = {}
        self._sources: Dict[str, ConfigSource] = {}
        self._load_config()
    
    def _get_default_config_file(self) -> Path:
        """Get default configuration file path."""
        return Path.home() / ".robodata" / "config.yaml"
    
    def _load_config(self) -> None:
        """Load configuration from all sources."""
        # 1. Load defaults
        self._load_defaults()
        
        # 2. Load from file
        if self.config_file.exists():
            self._load_from_file()
        
        # 3. Load from environment variables
        self._load_from_env()
        
        # 4. Validate final configuration
        self._validate_config()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        defaults = {
            "conversion": {
                "target_format": "lerobot-v3",
                "num_workers": 1,
                "fail_on_error": False,
                "fps": None,
                "robot_type": None,
                "camera_mapping": {},
                "field_mapping": {},
            },
            "logging": {
                "level": "INFO",
                "enable_rich": True,
                "enable_file_logging": False,
                "log_file": None,
                "max_file_size_mb": 100,
                "backup_count": 3,
            },
            "cache": {
                "max_size": 1000,
                "strategy": "lru",
                "ttl_seconds": None,
                "enable_disk_cache": False,
                "disk_cache_dir": None,
                "max_disk_size_mb": 1024,
            },
            "performance": {
                "memory_limit_mb": 2048,
                "chunk_size": 1000,
                "enable_parallel": True,
                "ray_address": None,
            },
            "quality": {
                "enable_scoring": True,
                "export_format": "json",
                "thresholds": {
                    "smoothness": 5.0,
                    "dead_actions": 0.1,
                    "gripper_chatter": 0.2,
                    "static_detection": 0.3,
                },
            },
        }
        
        self._merge_dict(defaults, ConfigSource.DEFAULT)
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            self._merge_dict(file_config, ConfigSource.FILE)
            self.logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "ROBODATA_CONVERSION_TARGET_FORMAT": "conversion.target_format",
            "ROBODATA_CONVERSION_NUM_WORKERS": "conversion.num_workers",
            "ROBODATA_CONVERSION_FAIL_ON_ERROR": "conversion.fail_on_error",
            "ROBODATA_LOGGING_LEVEL": "logging.level",
            "ROBODATA_LOGGING_ENABLE_FILE": "logging.enable_file_logging",
            "ROBODATA_LOGGING_FILE": "logging.log_file",
            "ROBODATA_CACHE_MAX_SIZE": "cache.max_size",
            "ROBODATA_CACHE_STRATEGY": "cache.strategy",
            "ROBODATA_CACHE_ENABLE_DISK": "cache.enable_disk_cache",
            "ROBODATA_PERFORMANCE_MEMORY_LIMIT": "performance.memory_limit_mb",
            "ROBODATA_PERFORMANCE_CHUNK_SIZE": "performance.chunk_size",
            "ROBODATA_PERFORMANCE_ENABLE_PARALLEL": "performance.enable_parallel",
            "ROBODATA_RAY_ADDRESS": "performance.ray_address",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value, ConfigSource.ENV)
    
    def _merge_dict(self, data: Dict[str, Any], source: ConfigSource) -> None:
        """Merge dictionary data into configuration."""
        def merge_recursive(target: Dict, source: Dict, path: str = ""):
            for key, value in source.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge_recursive(target[key], value, current_path)
                else:
                    target[key] = value
                    self._sources[current_path] = source
        
        merge_recursive(self._config_data, data)
    
    def _set_nested_value(self, path: str, value: Any, source: ConfigSource) -> None:
        """Set a nested configuration value."""
        keys = path.split('.')
        current = self._config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion for environment variables
        if source == ConfigSource.ENV:
            value = self._convert_env_value(value)
        
        current[keys[-1]] = value
        self._sources[path] = source
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self) -> None:
        """Validate the complete configuration."""
        validators = {
            "conversion.num_workers": self._validate_positive_int,
            "conversion.fail_on_error": self._validate_bool,
            "logging.level": self._validate_log_level,
            "logging.max_file_size_mb": self._validate_positive_int,
            "logging.backup_count": self._validate_positive_int,
            "cache.max_size": self._validate_positive_int,
            "cache.strategy": self._validate_cache_strategy,
            "cache.ttl_seconds": self._validate_positive_int,
            "cache.max_disk_size_mb": self._validate_positive_int,
            "performance.memory_limit_mb": self._validate_positive_int,
            "performance.chunk_size": self._validate_positive_int,
            "performance.enable_parallel": self._validate_bool,
        }
        
        for path, validator in validators.items():
            value = self.get(path)
            if value is not None:
                try:
                    validator(value, path)
                except ConfigValidationError as e:
                    self.logger.error(f"Configuration validation failed for {path}: {e}")
                    raise
    
    def _validate_positive_int(self, value: Any, path: str) -> None:
        """Validate positive integer value."""
        if not isinstance(value, int) or value <= 0:
            raise ConfigValidationError(f"{path} must be a positive integer, got {value}")
    
    def _validate_bool(self, value: Any, path: str) -> None:
        """Validate boolean value."""
        if not isinstance(value, bool):
            raise ConfigValidationError(f"{path} must be a boolean, got {value}")
    
    def _validate_log_level(self, value: Any, path: str) -> None:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value not in valid_levels:
            raise ConfigValidationError(f"{path} must be one of {valid_levels}, got {value}")
    
    def _validate_cache_strategy(self, value: Any, path: str) -> None:
        """Validate cache strategy."""
        valid_strategies = ["lru", "lfu", "fifo", "ttl"]
        if value not in valid_strategies:
            raise ConfigValidationError(f"{path} must be one of {valid_strategies}, got {value}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path."""
        keys = path.split('.')
        current = self._config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any, source: ConfigSource = ConfigSource.CLI) -> None:
        """Set a configuration value by path."""
        self._set_nested_value(path, value, source)
        self._validate_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self.get(section, {})
    
    def save(self, file_path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        save_path = file_path or self.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise ConfigValidationError(f"Failed to save configuration: {e}")
    
    def get_source(self, path: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value."""
        return self._sources.get(path)
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all configuration values with their sources."""
        result = {}
        
        def flatten_dict(data: Dict, prefix: str = ""):
            for key, value in data.items():
                current_path = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_dict(value, current_path)
                else:
                    result[current_path] = {
                        "value": value,
                        "source": self._sources.get(current_path, ConfigSource.DEFAULT)
                    }
        
        flatten_dict(self._config_data)
        return result
    
    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """Reset configuration to defaults."""
        if section:
            # Reset specific section
            self._load_defaults()
            section_data = self.get(section)
            self._config_data[section] = section_data
        else:
            # Reset all configuration
            self._config_data.clear()
            self._sources.clear()
            self._load_defaults()
        
        self._validate_config()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(path: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return get_config_manager().get(path, default)


def set_config(path: str, value: Any) -> None:
    """Set a configuration value."""
    get_config_manager().set(path, value)


def save_config(file_path: Optional[Path] = None) -> None:
    """Save configuration to file."""
    get_config_manager().save(file_path)


# Configuration classes for type safety
@dataclass
class ConversionConfig:
    """Configuration for conversion operations."""
    target_format: str = "lerobot-v3"
    num_workers: int = 1
    fail_on_error: bool = False
    fps: Optional[float] = None
    robot_type: Optional[str] = None
    camera_mapping: Dict[str, str] = field(default_factory=dict)
    field_mapping: Dict[str, str] = field(default_factory=dict)
    writer_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_global_config(cls) -> ConversionConfig:
        """Create from global configuration."""
        config_manager = get_config_manager()
        return cls(
            target_format=config_manager.get("conversion.target_format", "lerobot-v3"),
            num_workers=config_manager.get("conversion.num_workers", 1),
            fail_on_error=config_manager.get("conversion.fail_on_error", False),
            fps=config_manager.get("conversion.fps"),
            robot_type=config_manager.get("conversion.robot_type"),
            camera_mapping=config_manager.get("conversion.camera_mapping", {}),
            field_mapping=config_manager.get("conversion.field_mapping", {}),
            writer_config=config_manager.get_section("conversion.writer_config"),
        )


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    enable_rich: bool = True
    enable_file_logging: bool = False
    log_file: Optional[Path] = None
    max_file_size_mb: int = 100
    backup_count: int = 3
    
    @classmethod
    def from_global_config(cls) -> LoggingConfig:
        """Create from global configuration."""
        config_manager = get_config_manager()
        log_file = config_manager.get("logging.log_file")
        return cls(
            level=config_manager.get("logging.level", "INFO"),
            enable_rich=config_manager.get("logging.enable_rich", True),
            enable_file_logging=config_manager.get("logging.enable_file_logging", False),
            log_file=Path(log_file) if log_file else None,
            max_file_size_mb=config_manager.get("logging.max_file_size_mb", 100),
            backup_count=config_manager.get("logging.backup_count", 3),
        )


@dataclass
class CacheConfig:
    """Configuration for caching."""
    max_size: int = 1000
    strategy: str = "lru"
    ttl_seconds: Optional[int] = None
    enable_disk_cache: bool = False
    disk_cache_dir: Optional[Path] = None
    max_disk_size_mb: int = 1024
    
    @classmethod
    def from_global_config(cls) -> CacheConfig:
        """Create from global configuration."""
        config_manager = get_config_manager()
        cache_dir = config_manager.get("cache.disk_cache_dir")
        return cls(
            max_size=config_manager.get("cache.max_size", 1000),
            strategy=config_manager.get("cache.strategy", "lru"),
            ttl_seconds=config_manager.get("cache.ttl_seconds"),
            enable_disk_cache=config_manager.get("cache.enable_disk_cache", False),
            disk_cache_dir=Path(cache_dir) if cache_dir else None,
            max_disk_size_mb=config_manager.get("cache.max_disk_size_mb", 1024),
        )


# Context manager for temporary configuration changes
class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, changes: Dict[str, Any]):
        self.changes = changes
        self.original_values: Dict[str, Any] = {}
        self.config_manager = get_config_manager()
    
    def __enter__(self):
        # Store original values
        for path in self.changes:
            self.original_values[path] = self.config_manager.get(path)
        
        # Apply changes
        for path, value in self.changes.items():
            self.config_manager.set(path, value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for path, value in self.original_values.items():
            if value is not None:
                self.config_manager.set(path, value)


def temporary_config(**changes):
    """Create a temporary configuration context."""
    return ConfigContext(changes)
