"""Tests for the enhanced configuration management system."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from robodata.core.config_manager_v2 import (
    ConfigManager,
    ConfigSource,
    ConfigValidationError,
    ConversionConfig,
    LoggingConfig,
    CacheConfig,
    get_config_manager,
    get_config,
    set_config,
    temporary_config,
)


class TestConfigManager:
    """Test ConfigManager class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create a temporary config file
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yaml"
        self.config_file.write_text("""
conversion:
  target_format: "lerobot-v3"
  num_workers: 4
  fail_on_error: true

logging:
  level: "DEBUG"
  enable_rich: false
  enable_file_logging: true
  log_file: "test.log"

cache:
  max_size: 500
  strategy: "lfu"
  ttl_seconds: 3600
""")
    
    def teardown_method(self):
        """Cleanup after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        manager = ConfigManager(self.config_file)
        
        assert manager.get("conversion.target_format") == "lerobot-v3"
        assert manager.get("conversion.num_workers") == 4
        assert manager.get("conversion.fail_on_error") is True
        assert manager.get("logging.level") == "DEBUG"
        assert manager.get("logging.enable_rich") is False
        assert manager.get("cache.max_size") == 500
        assert manager.get("cache.strategy") == "lfu"
    
    def test_default_values(self):
        """Test default configuration values."""
        manager = ConfigManager(Path("nonexistent.yaml"))
        
        # Should have defaults
        assert manager.get("conversion.target_format") == "lerobot-v3"
        assert manager.get("conversion.num_workers") == 1
        assert manager.get("logging.level") == "INFO"
        assert manager.get("cache.max_size") == 1000
    
    def test_config_get_with_defaults(self):
        """Test getting configuration with default values."""
        manager = ConfigManager(self.config_file)
        
        # Existing value
        assert manager.get("conversion.num_workers") == 4
        
        # Non-existent value with default
        assert manager.get("nonexistent.key", "default") == "default"
        
        # Non-existent value without default
        assert manager.get("nonexistent.key") is None
    
    def test_config_set(self):
        """Test setting configuration values."""
        manager = ConfigManager(self.config_file)
        
        # Set new value
        manager.set("conversion.num_workers", 8)
        assert manager.get("conversion.num_workers") == 8
        
        # Set nested value
        manager.set("new.nested.value", "test")
        assert manager.get("new.nested.value") == "test"
        
        # Check source
        assert manager.get_source("conversion.num_workers") == ConfigSource.CLI
        assert manager.get_source("new.nested.value") == ConfigSource.CLI
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'ROBODATA_CONVERSION_TARGET_FORMAT': 'rlds',
            'ROBODATA_LOGGING_LEVEL': 'ERROR',
            'ROBODATA_CACHE_MAX_SIZE': '2000',
        }):
            manager = ConfigManager(Path("nonexistent.yaml"))
            
            assert manager.get("conversion.target_format") == "rlds"
            assert manager.get("logging.level") == "ERROR"
            assert manager.get("cache.max_workers") == 2000  # Note: typo in original env var
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager(self.config_file)
        
        # Valid values should not raise
        manager.set("conversion.num_workers", 4)
        manager.set("logging.level", "INFO")
        manager.set("cache.strategy", "lru")
        
        # Invalid values should raise
        with pytest.raises(ConfigValidationError):
            manager.set("conversion.num_workers", -1)
        
        with pytest.raises(ConfigValidationError):
            manager.set("logging.level", "INVALID")
        
        with pytest.raises(ConfigValidationError):
            manager.set("cache.strategy", "invalid")
    
    def test_config_save(self):
        """Test saving configuration to file."""
        manager = ConfigManager(self.config_file)
        
        # Modify configuration
        manager.set("conversion.num_workers", 8)
        manager.set("new.value", "test")
        
        # Save to new file
        new_config_file = self.temp_dir / "new_config.yaml"
        manager.save(new_config_file)
        
        # Load and verify
        new_manager = ConfigManager(new_config_file)
        assert new_manager.get("conversion.num_workers") == 8
        assert new_manager.get("new.value") == "test"
    
    def test_get_section(self):
        """Test getting entire configuration sections."""
        manager = ConfigManager(self.config_file)
        
        conversion_section = manager.get_section("conversion")
        assert isinstance(conversion_section, dict)
        assert conversion_section["target_format"] == "lerobot-v3"
        assert conversion_section["num_workers"] == 4
        
        # Non-existent section
        nonexistent = manager.get_section("nonexistent")
        assert nonexistent == {}
    
    def test_list_all(self):
        """Test listing all configuration values."""
        manager = ConfigManager(self.config_file)
        
        all_config = manager.list_all()
        assert isinstance(all_config, dict)
        
        # Check some expected keys
        assert "conversion.target_format" in all_config
        assert "logging.level" in all_config
        assert "cache.max_size" in all_config
        
        # Check structure
        for key, value_info in all_config.items():
            assert "value" in value_info
            assert "source" in value_info
            assert isinstance(value_info["source"], ConfigSource)
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        manager = ConfigManager(self.config_file)
        
        # Modify some values
        manager.set("conversion.num_workers", 8)
        manager.set("new.value", "test")
        
        # Reset all
        manager.reset_to_defaults()
        
        # Should be back to defaults
        assert manager.get("conversion.num_workers") == 1
        assert manager.get("new.value") is None
        
        # Reset specific section
        manager.set("conversion.num_workers", 8)
        manager.reset_to_defaults("conversion")
        assert manager.get("conversion.num_workers") == 1


class TestConfigClasses:
    """Test configuration data classes."""
    
    def test_conversion_config_from_global(self):
        """Test ConversionConfig creation from global config."""
        with patch('robodata.core.config_manager_v2.get_config_manager') as mock_get_manager:
            mock_manager = ConfigManager()
            mock_manager.set("conversion.target_format", "rlds")
            mock_manager.set("conversion.num_workers", 6)
            mock_manager.set("conversion.camera_mapping", {"cam1": "camera_1"})
            mock_get_manager.return_value = mock_manager
            
            config = ConversionConfig.from_global_config()
            
            assert config.target_format == "rlds"
            assert config.num_workers == 6
            assert config.camera_mapping == {"cam1": "camera_1"}
    
    def test_logging_config_from_global(self):
        """Test LoggingConfig creation from global config."""
        with patch('robodata.core.config_manager_v2.get_config_manager') as mock_get_manager:
            mock_manager = ConfigManager()
            mock_manager.set("logging.level", "ERROR")
            mock_manager.set("logging.enable_file_logging", True)
            mock_manager.set("logging.log_file", "/tmp/test.log")
            mock_get_manager.return_value = mock_manager
            
            config = LoggingConfig.from_global_config()
            
            assert config.level == "ERROR"
            assert config.enable_file_logging is True
            assert config.log_file == Path("/tmp/test.log")
    
    def test_cache_config_from_global(self):
        """Test CacheConfig creation from global config."""
        with patch('robodata.core.config_manager_v2.get_config_manager') as mock_get_manager:
            mock_manager = ConfigManager()
            mock_manager.set("cache.max_size", 2000)
            mock_manager.set("cache.strategy", "lfu")
            mock_manager.set("cache.enable_disk_cache", True)
            mock_manager.set("cache.disk_cache_dir", "/tmp/cache")
            mock_get_manager.return_value = mock_manager
            
            config = CacheConfig.from_global_config()
            
            assert config.max_size == 2000
            assert config.strategy == "lfu"
            assert config.enable_disk_cache is True
            assert config.disk_cache_dir == Path("/tmp/cache")


class TestConfigContext:
    """Test configuration context manager."""
    
    def test_temporary_config_changes(self):
        """Test temporary configuration changes."""
        manager = ConfigManager()
        manager.set("test.value", "original")
        
        with temporary_config(test_value="temporary"):
            assert manager.get("test.value") == "temporary"
        
        # Should be restored
        assert manager.get("test.value") == "original"
    
    def test_nested_temporary_config(self):
        """Test nested temporary configuration contexts."""
        manager = ConfigManager()
        manager.set("test.value", "original")
        
        with temporary_config(test_value="outer"):
            assert manager.get("test.value") == "outer"
            
            with temporary_config(test_value="inner"):
                assert manager.get("test.value") == "inner"
            
            assert manager.get("test.value") == "outer"
        
        assert manager.get("test.value") == "original"
    
    def test_temporary_config_with_exception(self):
        """Test temporary configuration with exception."""
        manager = ConfigManager()
        manager.set("test.value", "original")
        
        try:
            with temporary_config(test_value="temporary"):
                assert manager.get("test.value") == "temporary"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be restored
        assert manager.get("test.value") == "original"


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2
    
    def test_global_get_set_config(self):
        """Test global get/set config functions."""
        # Reset global manager for test
        import robodata.core.config_manager_v2 as config_module
        config_module._config_manager = None
        
        set_config("test.global_value", "test")
        assert get_config("test.global_value") == "test"
        
        # Test with default
        assert get_config("nonexistent.key", "default") == "default"


if __name__ == "__main__":
    pytest.main([__file__])
