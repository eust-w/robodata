"""Enhanced error handling utilities for Robodata."""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from robodata.core.exceptions import RobodataError

T = TypeVar("T")


class ErrorContext:
    """Context manager for consistent error handling and logging."""
    
    def __init__(
        self,
        operation: str,
        logger: logging.Logger | None = None,
        reraise: type[Exception] | bool = True,
        default: Any = None,
    ):
        """Initialize error context.
        
        Args:
            operation: Description of the operation being performed
            logger: Logger instance for error reporting
            reraise: Exception type to reraise, True for original, False to suppress
            default: Default value to return on error (if reraise=False)
        """
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.reraise = reraise
        self.default = default
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Error in {self.operation}: {exc_val}", exc_info=True)
            
            if self.reraise is False:
                # Suppress the exception
                return True
            elif self.reraise is not True and issubclass(exc_type, self.reraise):
                # Reraise as specified type
                raise self.reraise(f"Failed to {self.operation}: {exc_val}") from exc_val
            # Otherwise let the original exception propagate
        return False


def safe_execute(
    func: Callable[..., T],
    *args,
    operation: str = "operation",
    default: T | None = None,
    logger: logging.Logger | None = None,
    **kwargs
) -> T:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        operation: Description of the operation for error messages
        default: Default value to return on error
        logger: Logger instance for error reporting
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error in {operation}: {e}", exc_info=True)
        return default  # type: ignore


def handle_conversion_error(
    error: Exception,
    source_format: str,
    target_format: str,
    operation: str,
    logger: logging.Logger | None = None,
) -> RobodataError:
    """Handle conversion errors consistently.
    
    Args:
        error: Original exception
        source_format: Source format name
        target_format: Target format name
        operation: Operation being performed
        logger: Logger instance for error reporting
        
    Returns:
        RobodataError with consistent formatting
    """
    message = f"{operation} failed: {error}"
    if logger:
        logger.error(f"Conversion error ({source_format} -> {target_format}): {message}", exc_info=True)
    
    if isinstance(error, RobodataError):
        return error
    else:
        return RobodataError(message)
