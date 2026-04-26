"""Comprehensive error handling system for Robodata.

Provides structured exception handling, error recovery, and reporting
mechanisms for robust operation.
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from robodata.core.logging_system import get_logger


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling."""
    FORMAT = "format"
    IO = "io"
    VALIDATION = "validation"
    CONVERSION = "conversion"
    MEMORY = "memory"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    source_path: Path | None = None
    target_path: Path | None = None
    format_name: str | None = None
    episode_index: int | None = None
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorReport:
    """Structured error report."""
    exception: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback: str | None = None
    timestamp: str = field(default_factory=lambda: "")
    recovery_suggestions: list[str] = field(default_factory=list)


class ErrorHandler:
    """Centralized error handler for Robodata."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = get_logger(logger_name)
        self._error_handlers: dict[ErrorCategory, list[Callable]] = {}
        self._recovery_strategies: dict[ErrorCategory, list[Callable]] = {}
    
    def register_handler(self, category: ErrorCategory, handler: Callable) -> None:
        """Register a handler for a specific error category."""
        if category not in self._error_handlers:
            self._error_handlers[category] = []
        self._error_handlers[category].append(handler)
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable) -> None:
        """Register a recovery strategy for a specific error category."""
        if category not in self._recovery_strategies:
            self._recovery_strategies[category] = []
        self._recovery_strategies[category].append(strategy)
    
    def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        reraise: bool = True
    ) -> ErrorReport:
        """Handle an error with appropriate strategies."""
        # Categorize the error
        category = self._categorize_error(exception)
        severity = self._determine_severity(exception, context)
        
        # Create error report
        report = ErrorReport(
            exception=exception,
            severity=severity,
            category=category,
            context=context,
            traceback=traceback.format_exc(),
            recovery_suggestions=self._get_recovery_suggestions(category)
        )
        
        # Log the error
        self._log_error(report)
        
        # Apply handlers
        self._apply_handlers(report)
        
        # Attempt recovery if appropriate
        if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            self._attempt_recovery(report)
        
        if reraise:
            raise exception
        
        return report
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize an exception based on its type and message."""
        exception_type = type(exception).__name__
        message = str(exception).lower()
        
        # Format-related errors
        if any(keyword in message for keyword in ["format", "unsupported", "invalid format"]):
            return ErrorCategory.FORMAT
        
        # IO-related errors
        if any(keyword in message for keyword in ["file", "path", "directory", "permission"]):
            return ErrorCategory.IO
        
        # Validation errors
        if any(keyword in message for keyword in ["validation", "invalid", "schema"]):
            return ErrorCategory.VALIDATION
        
        # Memory errors
        if exception_type in ["MemoryError", "OutOfMemoryError"]:
            return ErrorCategory.MEMORY
        
        # Network errors
        if any(keyword in message for keyword in ["network", "connection", "timeout"]):
            return ErrorCategory.NETWORK
        
        # Dependency errors
        if any(keyword in message for keyword in ["import", "module", "dependency"]):
            return ErrorCategory.DEPENDENCY
        
        # Configuration errors
        if any(keyword in message for keyword in ["config", "setting", "option"]):
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on exception and context."""
        exception_type = type(exception).__name__
        
        # Critical errors
        if exception_type in ["SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL
        
        # High severity
        if exception_type in ["MemoryError", "OSError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if exception_type in ["ValueError", "TypeError"]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        return ErrorSeverity.LOW
    
    def _get_recovery_suggestions(self, category: ErrorCategory) -> list[str]:
        """Get recovery suggestions for an error category."""
        suggestions = {
            ErrorCategory.FORMAT: [
                "Check if the dataset format is supported",
                "Verify the dataset path is correct",
                "Try specifying the format explicitly"
            ],
            ErrorCategory.IO: [
                "Check file permissions",
                "Verify disk space is available",
                "Ensure the path exists"
            ],
            ErrorCategory.VALIDATION: [
                "Check dataset schema",
                "Verify data integrity",
                "Review format specifications"
            ],
            ErrorCategory.MEMORY: [
                "Reduce batch size",
                "Close other applications",
                "Use streaming mode if available"
            ],
            ErrorCategory.NETWORK: [
                "Check internet connection",
                "Try again later",
                "Use local dataset if available"
            ],
            ErrorCategory.DEPENDENCY: [
                "Install missing dependencies",
                "Update package versions",
                "Check Python version compatibility"
            ],
            ErrorCategory.CONFIGURATION: [
                "Review configuration file",
                "Check parameter values",
                "Use default settings"
            ]
        }
        
        return suggestions.get(category, ["Contact support for assistance"])
    
    def _log_error(self, report: ErrorReport) -> None:
        """Log the error with appropriate level."""
        log_method = {
            ErrorSeverity.LOW: self.logger.warning,
            ErrorSeverity.MEDIUM: self.logger.error,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
        }[report.severity]
        
        log_method(
            f"{report.category.value.upper()} error in {report.context.operation}: "
            f"{report.exception}",
            format=report.context.format_name,
            episode_id=report.context.episode_index
        )
        
        # Log detailed info for debugging
        self.logger.debug(f"Error details: {report.traceback}")
        
        # Log recovery suggestions
        if report.recovery_suggestions:
            self.logger.info("Recovery suggestions:")
            for suggestion in report.recovery_suggestions:
                self.logger.info(f"  - {suggestion}")
    
    def _apply_handlers(self, report: ErrorReport) -> None:
        """Apply registered handlers for the error category."""
        handlers = self._error_handlers.get(report.category, [])
        for handler in handlers:
            try:
                handler(report)
            except Exception as e:
                self.logger.error(f"Error handler failed: {e}")
    
    def _attempt_recovery(self, report: ErrorReport) -> None:
        """Attempt recovery using registered strategies."""
        strategies = self._recovery_strategies.get(report.category, [])
        for strategy in strategies:
            try:
                if strategy(report):
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    break
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.__name__} failed: {e}")


class ErrorCollector:
    """Collects and manages multiple errors during operations."""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.errors: list[ErrorReport] = []
        self.error_handler = ErrorHandler()
    
    def add_error(
        self,
        exception: Exception,
        context: ErrorContext,
        reraise: bool = False
    ) -> ErrorReport:
        """Add an error to the collection."""
        report = self.error_handler.handle_error(exception, context, reraise=False)
        
        if len(self.errors) < self.max_errors:
            self.errors.append(report)
        else:
            # Replace oldest error if at capacity
            self.errors.pop(0)
            self.errors.append(report)
        
        return report
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def get_errors_by_category(self, category: ErrorCategory) -> list[ErrorReport]:
        """Get errors filtered by category."""
        return [error for error in self.errors if error.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[ErrorReport]:
        """Get errors filtered by severity."""
        return [error for error in self.errors if error.severity == severity]
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected errors."""
        summary = {
            "total_errors": len(self.errors),
            "by_category": {},
            "by_severity": {},
            "most_common": []
        }
        
        # Count by category
        for error in self.errors:
            category = error.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        
        # Count by severity
        for error in self.errors:
            severity = error.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        return summary
    
    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()


# Context managers for error handling
@contextmanager
def error_context(
    operation: str,
    source_path: Path | None = None,
    target_path: Path | None = None,
    format_name: str | None = None,
    reraise: bool = True
):
    """Context manager for error handling with automatic context creation."""
    context = ErrorContext(
        operation=operation,
        source_path=source_path,
        target_path=target_path,
        format_name=format_name
    )
    
    try:
        yield
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle_error(e, context, reraise=reraise)


@contextmanager
def error_collection(max_errors: int = 100):
    """Context manager that collects errors without raising them."""
    collector = ErrorCollector(max_errors)
    
    try:
        yield collector
    except Exception as e:
        # Add the error to collector but don't raise
        context = ErrorContext(operation="unspecified")
        collector.add_error(e, context, reraise=False)


# Decorator for automatic error handling
def handle_errors(
    operation: str,
    reraise: bool = True,
    format_name: str | None = None
):
    """Decorator for automatic error handling in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with error_context(operation, reraise=reraise, format_name=format_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    return _global_error_handler
