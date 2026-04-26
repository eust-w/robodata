"""Structured logging system for Robodata.

Provides consistent, configurable logging across all modules with proper
level handling, formatting, and output destinations.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


class LogLevel(Enum):
    """Log levels with semantic meaning."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Configuration for the logging system."""
    level: LogLevel = LogLevel.INFO
    enable_rich: bool = True
    enable_file_logging: bool = False
    log_file: Path | None = None
    max_file_size_mb: int = 100
    backup_count: int = 3
    enable_progress: bool = True
    format_string: str | None = None


class RobodataFormatter(logging.Formatter):
    """Custom formatter for Robodata log messages."""
    
    def __init__(self, include_rich: bool = True):
        self.include_rich = include_rich
        if include_rich:
            super().__init__(
                fmt="%(message)s",
                datefmt="[%X]"
            )
        else:
            super().__init__(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        # Add module information for better debugging
        if not hasattr(record, 'module'):
            record.module = record.name.split('.')[-1] if '.' in record.name else record.name
        
        return super().format(record)


class ProgressLogger:
    """Specialized logger for progress tracking."""
    
    def __init__(self, logger: logging.Logger, total: int, description: str = "Processing"):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.last_reported = 0
        self.report_interval = max(1, total // 100)  # Report every 1%
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        
        # Report progress at intervals
        if (self.current - self.last_reported) >= self.report_interval:
            percentage = (self.current / self.total) * 100
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"
            )
            self.last_reported = self.current
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.logger.info(f"{self.description}: Complete ({self.total}/{self.total})")


class StructuredLogger:
    """Structured logger with contextual information."""
    
    def __init__(self, name: str, config: LogConfig | None = None):
        self.name = name
        self.config = config or LogConfig()
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the logger with appropriate handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config.enable_rich:
            console = Console(stderr=True)
            handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True
            )
        else:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(RobodataFormatter(include_rich=False))
        
        logger.addHandler(handler)
        
        # File handler
        if self.config.enable_file_logging and self.config.log_file:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(RobodataFormatter(include_rich=False))
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with optional context."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method with context support."""
        # Add contextual information
        extra = {}
        
        if 'episode_id' in kwargs:
            extra['episode_id'] = kwargs['episode_id']
            message = f"[Episode {kwargs['episode_id']}] {message}"
        
        if 'format' in kwargs:
            extra['format'] = kwargs['format']
            message = f"[{kwargs['format']}] {message}"
        
        if 'stage' in kwargs:
            extra['stage'] = kwargs['stage']
            message = f"[{kwargs['stage']}] {message}"
        
        # Log the message
        self._logger.log(getattr(logging, level.value), message, extra=extra)
    
    @contextmanager
    def log_context(self, **context):
        """Context manager for adding temporary context to log messages."""
        original_context = {}
        
        # Store original context if it exists
        for key, value in context.items():
            if hasattr(self._logger, key):
                original_context[key] = getattr(self._logger, key)
            setattr(self._logger, key, value)
        
        try:
            yield self
        finally:
            # Restore original context
            for key in context:
                if key in original_context:
                    setattr(self._logger, key, original_context[key])
                else:
                    delattr(self._logger, key)
    
    def create_progress_logger(self, total: int, description: str = "Processing") -> ProgressLogger:
        """Create a progress logger."""
        return ProgressLogger(self._logger, total, description)


class LoggingManager:
    """Global logging manager for Robodata."""
    
    _instance: LoggingManager | None = None
    _config: LogConfig = LogConfig()
    _loggers: dict[str, StructuredLogger] = {}
    
    def __new__(cls) -> LoggingManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def configure(cls, config: LogConfig) -> None:
        """Configure global logging settings."""
        cls._config = config
        # Reconfigure existing loggers
        for logger in cls._loggers.values():
            logger.config = config
            logger._logger = logger._setup_logger()
    
    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(name, cls._config)
        return cls._loggers[name]
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown all loggers."""
        for logger in cls._loggers.values():
            for handler in logger._logger.handlers:
                handler.close()
        cls._loggers.clear()
        logging.shutdown()


# Convenience functions
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return LoggingManager.get_logger(name)


def configure_logging(config: LogConfig) -> None:
    """Configure global logging."""
    LoggingManager.configure(config)


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    enable_rich: bool = True,
    log_file: Path | None = None,
    **kwargs
) -> None:
    """Setup logging with common parameters."""
    config = LogConfig(
        level=level,
        enable_rich=enable_rich,
        log_file=log_file,
        **kwargs
    )
    configure_logging(config)


# Context manager for temporary logging configuration
@contextmanager
def temporary_logging_config(config: LogConfig):
    """Temporarily change logging configuration."""
    original_config = LoggingManager._config
    LoggingManager.configure(config)
    try:
        yield
    finally:
        LoggingManager.configure(original_config)


# Performance logging decorator
def log_performance(logger_name: str = __name__):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            import time
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator


# Exception logging decorator
def log_exceptions(logger_name: str = __name__, reraise: bool = True):
    """Decorator to log exceptions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator
