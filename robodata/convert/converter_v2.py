"""Simplified and optimized converter implementation.

This module provides a clean, efficient conversion facade that focuses on
the core conversion logic while delegating specialized tasks to dedicated
components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from robodata.config.models import ConversionConfig
from robodata.core.exceptions import ConversionError, UnsupportedFormatError
from robodata.core.models import DatasetInfo
from robodata.core.pipeline import (
    ProcessingConfig,
    ProcessingMode,
    ProcessingPipeline,
    ProcessingResult,
)
from robodata.formats.registry import FormatRegistry


class ConversionProcessor:
    """Core conversion processor that handles the transformation logic."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def setup(self) -> None:
        """Setup the conversion processor."""
        self.logger.debug("Setting up conversion processor")
    
    def cleanup(self) -> None:
        """Cleanup the conversion processor."""
        self.logger.debug("Cleaning up conversion processor")
    
    def process_item(self, episode_data: Any) -> Any:
        """Process a single episode for conversion."""
        # The actual processing logic would be implemented here
        # For now, we delegate to the format-specific handlers
        return episode_data


class ConversionOrchestrator:
    """High-level conversion orchestrator that manages the entire conversion process."""
    
    def __init__(self, config: ConversionConfig | None = None):
        self.config = config or ConversionConfig()
        self.logger = logging.getLogger(__name__)
    
    def convert(
        self,
        source: str | Path,
        output: str | Path,
        target_format: str | None = None,
        source_format: str | None = None,
        progress_callback: callable | None = None,
    ) -> ConversionResult:
        """Convert a dataset from source to target format.
        
        Args:
            source: Path to source dataset.
            output: Path for output dataset.
            target_format: Target format (uses config default if None).
            source_format: Source format hint (auto-detect if None).
            progress_callback: Optional progress callback.
            
        Returns:
            ConversionResult with success/failure information.
        """
        source = Path(source)
        output = Path(output)
        target_format = target_format or self.config.target_format
        
        # Initialize result
        result = ConversionResult(
            success=False,
            source_format="unknown",
            target_format=target_format,
            output_path=output,
        )
        
        try:
            # 1. Detect source format
            source_format = self._detect_source_format(source, result)
            if source_format is None:
                return result
            
            # 2. Get format handlers
            reader, writer = self._get_format_handlers(source_format, target_format, result)
            if reader is None or writer is None:
                return result
            
            # 3. Inspect source dataset
            dataset_info = self._inspect_dataset(reader, source)
            
            # 4. Configure processing
            processing_config = self._create_processing_config(progress_callback)
            
            # 5. Execute conversion
            result = self._execute_conversion(
                reader, writer, source, output, dataset_info, processing_config, result
            )
            
            # 6. Finalize output
            self._finalize_output(writer, output, dataset_info, result)
            
        except Exception as e:
            self._handle_conversion_error(e, result)
        
        return result
    
    def _detect_source_format(self, source: Path, result: ConversionResult) -> str | None:
        """Detect the source format."""
        try:
            source_format = FormatRegistry.detect_format(source)
            if source_format is None:
                error_msg = "Unable to detect source format"
                result.errors.append(error_msg)
                if self.config.fail_on_error:
                    raise ConversionError("unknown", result.target_format, error_msg)
                return None
            
            result.source_format = source_format
            return source_format
            
        except Exception as e:
            error_msg = f"Format detection failed: {e}"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise ConversionError("unknown", result.target_format, error_msg)
            return None
    
    def _get_format_handlers(
        self, source_format: str, target_format: str, result: ConversionResult
    ) -> tuple[Any, Any]:
        """Get reader and writer for the specified formats."""
        try:
            reader = FormatRegistry.get_reader(source_format)
        except UnsupportedFormatError as e:
            result.errors.append(str(e))
            if self.config.fail_on_error:
                raise
            return None, None
        
        try:
            writer = FormatRegistry.get_writer(target_format)
        except UnsupportedFormatError as e:
            error_msg = f"No writer for format: {target_format}"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise
            return None, None
        
        return reader, writer
    
    def _inspect_dataset(self, reader: Any, source: Path) -> DatasetInfo:
        """Inspect the source dataset."""
        try:
            dataset_info = reader.inspect(source)
            self._apply_config_overrides(dataset_info)
            return dataset_info
        except Exception as e:
            raise ConversionError("unknown", "unknown", f"Dataset inspection failed: {e}")
    
    def _apply_config_overrides(self, dataset_info: DatasetInfo) -> None:
        """Apply configuration overrides to dataset info."""
        if self.config.fps is not None:
            dataset_info.inferred_fps = self.config.fps
        
        if self.config.robot_type is not None:
            dataset_info.inferred_robot_type = self.config.robot_type
    
    def _create_processing_config(self, progress_callback: callable | None) -> ProcessingConfig:
        """Create processing configuration based on conversion config."""
        # Determine processing mode
        if self.config.num_workers > 1:
            mode = ProcessingMode.PARALLEL
        else:
            mode = ProcessingMode.SEQUENTIAL
        
        return ProcessingConfig(
            mode=mode,
            num_workers=self.config.num_workers,
            fail_fast=self.config.fail_on_error,
            progress_callback=progress_callback,
        )
    
    def _execute_conversion(
        self,
        reader: Any,
        writer: Any,
        source: Path,
        output: Path,
        dataset_info: DatasetInfo,
        processing_config: ProcessingConfig,
        result: ConversionResult,
    ) -> ConversionResult:
        """Execute the actual conversion process."""
        # Configure writer
        self._configure_writer(writer, dataset_info)
        
        # Create processor and pipeline
        processor = ConversionProcessor(self.config)
        pipeline = ProcessingPipeline(processor, processing_config)
        
        # Create data source and sink adapters
        source_adapter = self._create_source_adapter(reader, source)
        sink_adapter = self._create_sink_adapter(writer, output)
        
        # Execute pipeline
        processing_result = pipeline.execute(source_adapter, sink_adapter)
        
        # Update conversion result
        result.episodes_converted = processing_result.items_processed
        result.episodes_failed = processing_result.items_failed
        result.errors.extend(processing_result.errors)
        
        return result
    
    def _configure_writer(self, writer: Any, dataset_info: DatasetInfo) -> None:
        """Configure writer with dataset-specific options."""
        if hasattr(writer, "config"):
            config = writer.config
            
            # Apply dataset info
            if hasattr(config, "fps") and dataset_info.inferred_fps:
                config.fps = dataset_info.inferred_fps
            
            if hasattr(config, "robot_type") and dataset_info.inferred_robot_type:
                config.robot_type = dataset_info.inferred_robot_type
            
            # Apply conversion config
            if hasattr(config, "camera_name_mapping") and self.config.camera_mapping:
                config.camera_name_mapping = self.config.camera_mapping
            
            if hasattr(config, "field_mapping") and self.config.field_mapping:
                config.field_mapping = self.config.field_mapping
    
    def _create_source_adapter(self, reader: Any, source: Path) -> Any:
        """Create a data source adapter for the reader."""
        # This would implement the DataSource protocol
        # For now, return the reader directly
        return reader
    
    def _create_sink_adapter(self, writer: Any, output: Path) -> Any:
        """Create a data sink adapter for the writer."""
        # This would implement the DataSink protocol
        # For now, return the writer directly
        return writer
    
    def _finalize_output(
        self, writer: Any, output: Path, dataset_info: DatasetInfo, result: ConversionResult
    ) -> None:
        """Finalize the output dataset."""
        try:
            writer.finalize(output, dataset_info)
        except Exception as e:
            error_msg = f"Finalization failed: {e}"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise ConversionError(result.source_format, result.target_format, error_msg)
    
    def _handle_conversion_error(self, error: Exception, result: ConversionResult) -> None:
        """Handle conversion errors appropriately."""
        error_msg = f"Conversion failed: {error}"
        result.errors.append(error_msg)
        
        if self.config.fail_on_error:
            if isinstance(error, ConversionError):
                raise
            else:
                raise ConversionError(result.source_format, result.target_format, error_msg)


# Simplified converter class that delegates to the orchestrator
class Converter:
    """Simplified converter facade."""
    
    def __init__(self, config: ConversionConfig | None = None):
        self.config = config or ConversionConfig()
        self.orchestrator = ConversionOrchestrator(config)
    
    def convert(
        self,
        source: str | Path,
        output: str | Path,
        target_format: str | None = None,
        source_format: str | None = None,
        progress_callback: callable | None = None,
    ) -> ConversionResult:
        """Convert a dataset using the orchestrator."""
        return self.orchestrator.convert(
            source=source,
            output=output,
            target_format=target_format,
            source_format=source_format,
            progress_callback=progress_callback,
        )


# Convenience function
def convert(
    source: str | Path,
    output: str | Path,
    target_format: str = "lerobot-v3",
    source_format: str | None = None,
    fps: float | None = None,
    robot_type: str | None = None,
    camera_mapping: dict[str, str] | None = None,
    fail_on_error: bool = False,
    **kwargs: Any,
) -> ConversionResult:
    """Convenience function to convert a dataset."""
    config = ConversionConfig(
        target_format=target_format,
        fps=fps,
        robot_type=robot_type,
        camera_mapping=camera_mapping or {},
        fail_on_error=fail_on_error,
        writer_config=kwargs,
    )
    
    converter = Converter(config)
    return converter.convert(source, output, target_format, source_format)
