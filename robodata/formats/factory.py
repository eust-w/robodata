"""Factory pattern for format readers and writers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union

from robodata.core.exceptions import UnsupportedFormatError
from robodata.core.models import DatasetInfo, Episode
from robodata.formats.base import BaseReader, BaseWriter, FormatReader, FormatWriter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Union[FormatReader, FormatWriter])


class FormatFactory:
    """Factory for creating format readers and writers.

    Provides a centralized way to create and manage format handlers
    with support for registration, discovery, and validation.
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._readers: Dict[str, Type[FormatReader]] = {}
        self._writers: Dict[str, Type[FormatWriter]] = {}
        self._reader_instances: Dict[str, FormatReader] = {}
        self._writer_instances: Dict[str, FormatWriter] = {}

    def register_reader(self, format_name: str, reader_class: Type[FormatReader]) -> None:
        """Register a format reader.

        Args:
            format_name: Format identifier.
            reader_class: Reader class to register.
        """
        if format_name in self._readers:
            logger.warning(f"Overriding existing reader for format: {format_name}")
        
        self._readers[format_name] = reader_class
        logger.debug(f"Registered reader for format: {format_name}")

    def register_writer(self, format_name: str, writer_class: Type[FormatWriter]) -> None:
        """Register a format writer.

        Args:
            format_name: Format identifier.
            writer_class: Writer class to register.
        """
        if format_name in self._writers:
            logger.warning(f"Overriding existing writer for format: {format_name}")
        
        self._writers[format_name] = writer_class
        logger.debug(f"Registered writer for format: {format_name}")

    def get_reader(self, format_name: str) -> FormatReader:
        """Get a reader instance for the specified format.

        Args:
            format_name: Format identifier.

        Returns:
            Reader instance.

        Raises:
            UnsupportedFormatError: If format is not supported.
        """
        if format_name not in self._readers:
            raise UnsupportedFormatError(
                f"Unsupported format: '{format_name}'. Available formats: {', '.join(self._readers.keys())}"
            )
        
        # Return cached instance if available
        if format_name in self._reader_instances:
            return self._reader_instances[format_name]
        
        # Create new instance
        reader_class = self._readers[format_name]
        reader = reader_class()
        self._reader_instances[format_name] = reader
        
        return reader

    def get_writer(self, format_name: str) -> FormatWriter:
        """Get a writer instance for the specified format.

        Args:
            format_name: Format identifier.

        Returns:
            Writer instance.

        Raises:
            UnsupportedFormatError: If format is not supported.
        """
        if format_name not in self._writers:
            raise UnsupportedFormatError(
                f"Unsupported format: '{format_name}'. Available formats: {', '.join(self._writers.keys())}"
            )
        
        # Return cached instance if available
        if format_name in self._writer_instances:
            return self._writer_instances[format_name]
        
        # Create new instance
        writer_class = self._writers[format_name]
        writer = writer_class()
        self._writer_instances[format_name] = writer
        
        return writer

    def detect_format(self, path: Path) -> str:
        """Detect the format of the dataset at the given path.

        Args:
            path: Path to dataset.

        Returns:
            Detected format name.

        Raises:
            UnsupportedFormatError: If format cannot be detected.
        """
        path = Path(path)
        
        # Try each registered reader
        for format_name, reader_class in self._readers.items():
            try:
                if reader_class.can_read(path):
                    logger.debug(f"Detected format: {format_name} for path: {path}")
                    return format_name
            except Exception as e:
                logger.debug(f"Error testing {format_name} reader: {e}")
                continue
        
        raise UnsupportedFormatError(f"Could not detect format for: {path}")

    def get_supported_formats(self) -> Dict[str, Dict[str, bool]]:
        """Get information about supported formats.

        Returns:
            Dictionary mapping format names to capabilities.
        """
        formats = {}
        
        # Collect all format names
        all_formats = set(self._readers.keys()) | set(self._writers.keys())
        
        for format_name in sorted(all_formats):
            formats[format_name] = {
                "can_read": format_name in self._readers,
                "can_write": format_name in self._writers,
            }
        
        return formats

    def validate_format_support(self, source_format: str, target_format: str) -> None:
        """Validate that both source and target formats are supported.

        Args:
            source_format: Source format name.
            target_format: Target format name.

        Raises:
            UnsupportedFormatError: If either format is not supported.
        """
        if source_format not in self._readers:
            raise UnsupportedFormatError(
                f"Source format '{source_format}' is not supported for reading"
            )
        
        if target_format not in self._writers:
            raise UnsupportedFormatError(
                f"Target format '{target_format}' is not supported for writing"
            )

    def clear_cache(self) -> None:
        """Clear cached reader and writer instances."""
        self._reader_instances.clear()
        self._writer_instances.clear()
        logger.debug("Cleared format handler cache")

    def get_reader_info(self, format_name: str) -> Dict[str, Any]:
        """Get information about a reader.

        Args:
            format_name: Format name.

        Returns:
            Reader information.
        """
        if format_name not in self._readers:
            raise UnsupportedFormatError(f"Format not supported: {format_name}")
        
        reader_class = self._readers[format_name]
        return {
            "format_name": format_name,
            "class_name": reader_class.__name__,
            "module": reader_class.__module__,
            "doc": reader_class.__doc__,
        }

    def get_writer_info(self, format_name: str) -> Dict[str, Any]:
        """Get information about a writer.

        Args:
            format_name: Format name.

        Returns:
            Writer information.
        """
        if format_name not in self._writers:
            raise UnsupportedFormatError(f"Format not supported: {format_name}")
        
        writer_class = self._writers[format_name]
        return {
            "format_name": format_name,
            "class_name": writer_class.__name__,
            "module": writer_class.__module__,
            "doc": writer_class.__doc__,
        }


class ReaderDecorator:
    """Decorator for registering readers with the factory."""

    def __init__(self, format_name: str, factory: FormatFactory | None = None) -> None:
        """Initialize decorator.

        Args:
            format_name: Format identifier.
            factory: Factory instance to use. If None, uses global factory.
        """
        self.format_name = format_name
        self.factory = factory or get_global_factory()

    def __call__(self, reader_class: Type[FormatReader]) -> Type[FormatReader]:
        """Register the reader class.

        Args:
            reader_class: Reader class to register.

        Returns:
            The same reader class, now registered.
        """
        self.factory.register_reader(self.format_name, reader_class)
        return reader_class


class WriterDecorator:
    """Decorator for registering writers with the factory."""

    def __init__(self, format_name: str, factory: FormatFactory | None = None) -> None:
        """Initialize decorator.

        Args:
            format_name: Format identifier.
            factory: Factory instance to use. If None, uses global factory.
        """
        self.format_name = format_name
        self.factory = factory or get_global_factory()

    def __call__(self, writer_class: Type[FormatWriter]) -> Type[FormatWriter]:
        """Register the writer class.

        Args:
            writer_class: Writer class to register.

        Returns:
            The same writer class, now registered.
        """
        self.factory.register_writer(self.format_name, writer_class)
        return writer_class


# Global factory instance
_global_factory: FormatFactory | None = None


def get_global_factory() -> FormatFactory:
    """Get the global format factory.

    Returns:
        Global FormatFactory instance.
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = FormatFactory()
    return _global_factory


def register_reader(format_name: str, factory: FormatFactory | None = None) -> ReaderDecorator:
    """Decorator for registering readers.

    Args:
        format_name: Format identifier.
        factory: Factory instance to use.

    Returns:
        Decorator function.

    Example:
        @register_reader("myformat")
        class MyFormatReader:
            ...
    """
    return ReaderDecorator(format_name, factory)


def register_writer(format_name: str, factory: FormatFactory | None = None) -> WriterDecorator:
    """Decorator for registering writers.

    Args:
        format_name: Format identifier.
        factory: Factory instance to use.

    Returns:
        Decorator function.

    Example:
        @register_writer("myformat")
        class MyFormatWriter:
            ...
    """
    return WriterDecorator(format_name, factory)


def get_reader(format_name: str) -> FormatReader:
    """Get a reader for the specified format.

    Args:
        format_name: Format identifier.

    Returns:
        Reader instance.
    """
    return get_global_factory().get_reader(format_name)


def get_writer(format_name: str) -> FormatWriter:
    """Get a writer for the specified format.

    Args:
        format_name: Format identifier.

    Returns:
        Writer instance.
    """
    return get_global_factory().get_writer(format_name)


def detect_format(path: Path) -> str:
    """Detect the format of the dataset at the given path.

    Args:
        path: Path to dataset.

    Returns:
        Detected format name.
    """
    return get_global_factory().detect_format(path)


def get_supported_formats() -> Dict[str, Dict[str, bool]]:
    """Get information about supported formats.

    Returns:
        Dictionary mapping format names to capabilities.
    """
    return get_global_factory().get_supported_formats()
