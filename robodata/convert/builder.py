"""Builder pattern for complex conversion operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from robodata.config.models import ConversionConfig
from robodata.convert.commands import Command, CommandInvoker, ConvertDatasetCommand, InspectDatasetCommand, ValidateDatasetCommand
from robodata.convert.observers import ConversionObserver, LoggingObserver, MetricsObserver, ProgressObserver
from robodata.convert.strategies import ConversionStrategy, ParallelConfig, ConversionStrategySelector
from robodata.core.config_manager import get_config, ConfigContext

logger = logging.getLogger(__name__)


class ConversionBuilder:
    """Builder for creating and configuring conversion operations."""
    
    def __init__(self):
        """Initialize builder."""
        self.logger = logging.getLogger(__name__)
        
        # Command parameters
        self._source_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._source_format: Optional[str] = None
        self._target_format: Optional[str] = None
        self._config: Optional[ConversionConfig] = None
        
        # Strategy and observers
        self._strategy: Optional[ConversionStrategy] = None
        self._parallel_config: Optional[ParallelConfig] = None
        self._observers: List[ConversionObserver] = []
        
        # Additional options
        self._auto_detect_format: bool = True
        self._validate_before_convert: bool = False
        self._inspect_before_convert: bool = False
        self._fail_on_validation_error: bool = True
        
        # Builder state
        self._built = False
    
    def source(self, path: Union[str, Path]) -> "ConversionBuilder":
        """Set source dataset path.
        
        Args:
            path: Source dataset path.
            
        Returns:
            Builder instance for chaining.
        """
        self._source_path = Path(path)
        return self
    
    def output(self, path: Union[str, Path]) -> "ConversionBuilder":
        """Set output dataset path.
        
        Args:
            path: Output dataset path.
            
        Returns:
            Builder instance for chaining.
        """
        self._output_path = Path(path)
        return self
    
    def source_format(self, format_name: str) -> "ConversionBuilder":
        """Set source format.
        
        Args:
            format_name: Source format name.
            
        Returns:
            Builder instance for chaining.
        """
        self._source_format = format_name
        self._auto_detect_format = False
        return self
    
    def target_format(self, format_name: str) -> "ConversionBuilder":
        """Set target format.
        
        Args:
            format_name: Target format name.
            
        Returns:
            Builder instance for chaining.
        """
        self._target_format = format_name
        return self
    
    def config(self, config: ConversionConfig) -> "ConversionBuilder":
        """Set conversion configuration.
        
        Args:
            config: Conversion configuration.
            
        Returns:
            Builder instance for chaining.
        """
        self._config = config
        return self
    
    def strategy(self, strategy: ConversionStrategy) -> "ConversionBuilder":
        """Set conversion strategy.
        
        Args:
            strategy: Conversion strategy to use.
            
        Returns:
            Builder instance for chaining.
        """
        self._strategy = strategy
        return self
    
    def parallel(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: int = 1,
    ) -> "ConversionBuilder":
        """Configure parallel processing.
        
        Args:
            num_workers: Number of worker processes/threads.
            use_processes: Whether to use processes (True) or threads (False).
            chunk_size: Size of work chunks.
            
        Returns:
            Builder instance for chaining.
        """
        self._parallel_config = ParallelConfig(
            num_workers=num_workers,
            use_processes=use_processes,
            chunk_size=chunk_size,
        )
        self._strategy = ConversionStrategy.PARALLEL
        return self
    
    def sequential(self) -> "ConversionBuilder":
        """Use sequential processing.
        
        Returns:
            Builder instance for chaining.
        """
        self._strategy = ConversionStrategy.SEQUENTIAL
        return self
    
    def streaming(self) -> "ConversionBuilder":
        """Use streaming processing.
        
        Returns:
            Builder instance for chaining.
        """
        self._strategy = ConversionStrategy.STREAMING
        return self
    
    def batch(self, batch_size: int = 10) -> "ConversionBuilder":
        """Use batch processing.
        
        Args:
            batch_size: Number of episodes per batch.
            
        Returns:
            Builder instance for chaining.
        """
        self._strategy = ConversionStrategy.BATCH
        # Store batch_size in parallel_config for the batch strategy
        self._parallel_config = ParallelConfig()
        self._parallel_config.chunk_size = batch_size
        return self
    
    def add_observer(self, observer: ConversionObserver) -> "ConversionBuilder":
        """Add an observer to the conversion.
        
        Args:
            observer: Observer to add.
            
        Returns:
            Builder instance for chaining.
        """
        self._observers.append(observer)
        return self
    
    def with_logging(self, log_level: str = "INFO") -> "ConversionBuilder":
        """Add logging observer.
        
        Args:
            log_level: Logging level.
            
        Returns:
            Builder instance for chaining.
        """
        import logging
        level = getattr(logging, log_level.upper(), logging.INFO)
        self._observers.append(LoggingObserver(log_level=level))
        return self
    
    def with_progress(self, report_interval: int = 10) -> "ConversionBuilder":
        """Add progress observer.
        
        Args:
            report_interval: Progress report interval (percentage).
            
        Returns:
            Builder instance for chaining.
        """
        self._observers.append(ProgressObserver(report_interval))
        return self
    
    def with_metrics(self) -> "ConversionBuilder":
        """Add metrics observer.
        
        Returns:
            Builder instance for chaining.
        """
        self._observers.append(MetricsObserver())
        return self
    
    def validate_before_convert(self, fail_on_error: bool = True) -> "ConversionBuilder":
        """Enable validation before conversion.
        
        Args:
            fail_on_error: Whether to fail on validation errors.
            
        Returns:
            Builder instance for chaining.
        """
        self._validate_before_convert = True
        self._fail_on_validation_error = fail_on_error
        return self
    
    def inspect_before_convert(self) -> "ConversionBuilder":
        """Enable inspection before conversion.
        
        Returns:
            Builder instance for chaining.
        """
        self._inspect_before_convert = True
        return self
    
    def auto_detect_format(self, enabled: bool = True) -> "ConversionBuilder":
        """Enable or disable auto format detection.
        
        Args:
            enabled: Whether to enable auto detection.
            
        Returns:
            Builder instance for chaining.
        """
        self._auto_detect_format = enabled
        return self
    
    def using_config(self, config_name: str) -> "ConversionBuilder":
        """Use a named configuration.
        
        Args:
            config_name: Configuration name.
            
        Returns:
            Builder instance for chaining.
        """
        config = get_config()
        # This would need to be implemented in the config manager
        # For now, just log it
        self.logger.info(f"Using configuration: {config_name}")
        return self
    
    def _validate(self) -> None:
        """Validate builder configuration."""
        if self._built:
            raise ValueError("Builder has already been used")
        
        if not self._source_path:
            raise ValueError("Source path is required")
        
        if not self._output_path:
            raise ValueError("Output path is required")
        
        if not self._target_format:
            raise ValueError("Target format is required")
        
        if self._auto_detect_format and not self._source_format:
            # Will be detected during build
            pass
        elif not self._source_format:
            raise ValueError("Source format is required when auto-detection is disabled")
    
    def build(self) -> ConvertDatasetCommand:
        """Build the conversion command.
        
        Returns:
            Configured conversion command.
            
        Raises:
            ValueError: If builder configuration is invalid.
        """
        self._validate()
        
        # Use default config if not provided
        if not self._config:
            self._config = ConversionConfig()
        
        # Auto-detect source format if needed
        if self._auto_detect_format and not self._source_format:
            from robodata.formats.factory import get_global_factory
            factory = get_global_factory()
            self._source_format = factory.detect_format(self._source_path)
            self.logger.info(f"Auto-detected source format: {self._source_format}")
        
        # Create command
        command = ConvertDatasetCommand(
            source_path=self._source_path,
            output_path=self._output_path,
            source_format=self._source_format,
            target_format=self._target_format,
            config=self._config,
        )
        
        # Configure strategy
        if self._strategy:
            # This would need to be passed to the command
            # For now, store in config
            self._config.strategy = self._strategy.value
        
        if self._parallel_config:
            self._config.parallel_config = self._parallel_config.__dict__
        
        # Add observers
        for observer in self._observers:
            command.add_observer(observer)
        
        self._built = True
        return command


class PipelineBuilder:
    """Builder for creating conversion pipelines."""
    
    def __init__(self):
        """Initialize pipeline builder."""
        self.logger = logging.getLogger(__name__)
        self._commands: List[Command] = []
        self._global_observers: List[ConversionObserver] = []
        self._invoker = CommandInvoker()
    
    def add_command(self, command: Command) -> "PipelineBuilder":
        """Add a command to the pipeline.
        
        Args:
            command: Command to add.
            
        Returns:
            Builder instance for chaining.
        """
        self._commands.append(command)
        return self
    
    def convert(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        target_format: str,
    ) -> "PipelineBuilder":
        """Add a conversion command to the pipeline.
        
        Args:
            source: Source dataset path.
            output: Output dataset path.
            target_format: Target format.
            
        Returns:
            Builder instance for chaining.
        """
        command = ConversionBuilder().source(source).output(output).target_format(target_format).build()
        return self.add_command(command)
    
    def inspect(self, dataset_path: Union[str, Path], deep_scan: bool = False) -> "PipelineBuilder":
        """Add an inspection command to the pipeline.
        
        Args:
            dataset_path: Dataset path to inspect.
            deep_scan: Whether to perform deep scan.
            
        Returns:
            Builder instance for chaining.
        """
        command = InspectDatasetCommand(
            dataset_path=Path(dataset_path),
            deep_scan=deep_scan,
        )
        return self.add_command(command)
    
    def validate(self, dataset_path: Union[str, Path], strict: bool = False) -> "PipelineBuilder":
        """Add a validation command to the pipeline.
        
        Args:
            dataset_path: Dataset path to validate.
            strict: Whether to use strict validation.
            
        Returns:
            Builder instance for chaining.
        """
        command = ValidateDatasetCommand(
            dataset_path=Path(dataset_path),
            strict=strict,
        )
        return self.add_command(command)
    
    def add_global_observer(self, observer: ConversionObserver) -> "PipelineBuilder":
        """Add an observer to all commands in the pipeline.
        
        Args:
            observer: Observer to add.
            
        Returns:
            Builder instance for chaining.
        """
        self._global_observers.append(observer)
        return self
    
    def with_logging(self, log_level: str = "INFO") -> "PipelineBuilder":
        """Add logging observer to all commands.
        
        Args:
            log_level: Logging level.
            
        Returns:
            Builder instance for chaining.
        """
        import logging
        level = getattr(logging, log_level.upper(), logging.INFO)
        return self.add_global_observer(LoggingObserver(log_level=level))
    
    def with_progress(self, report_interval: int = 10) -> "PipelineBuilder":
        """Add progress observer to all commands.
        
        Args:
            report_interval: Progress report interval.
            
        Returns:
            Builder instance for chaining.
        """
        return self.add_global_observer(ProgressObserver(report_interval))
    
    def with_metrics(self) -> "PipelineBuilder":
        """Add metrics observer to all commands.
        
        Returns:
            Builder instance for chaining.
        """
        return self.add_global_observer(MetricsObserver())
    
    def build(self) -> "ConversionPipeline":
        """Build the conversion pipeline.
        
        Returns:
            Configured conversion pipeline.
        """
        # Add global observers to all commands
        for command in self._commands:
            for observer in self._global_observers:
                command.add_observer(observer)
        
        return ConversionPipeline(
            commands=self._commands.copy(),
            invoker=self._invoker,
        )


class ConversionPipeline:
    """Pipeline for executing multiple conversion commands."""
    
    def __init__(self, commands: List[Command], invoker: CommandInvoker):
        """Initialize pipeline.
        
        Args:
            commands: List of commands to execute.
            invoker: Command invoker.
        """
        self.commands = commands
        self.invoker = invoker
        self.logger = logging.getLogger(__name__)
        self.results: List[Any] = []
    
    def execute(self) -> List[Any]:
        """Execute all commands in the pipeline.
        
        Returns:
            List of command results.
        """
        self.logger.info(f"Executing pipeline with {len(self.commands)} commands")
        
        self.results.clear()
        for i, command in enumerate(self.commands):
            self.logger.info(f"Executing command {i+1}/{len(self.commands)}: {command.__class__.__name__}")
            
            try:
                result = self.invoker.execute_command(command)
                self.results.append(result)
                
                if not result.success:
                    self.logger.error(f"Command {i+1} failed: {result.message}")
                    if len(result.errors) > 0:
                        self.logger.error(f"Errors: {result.errors}")
                    break  # Stop pipeline on failure
                    
            except Exception as e:
                self.logger.error(f"Command {i+1} raised exception: {e}")
                break
        
        self.logger.info(f"Pipeline completed with {len(self.results)} results")
        return self.results
    
    def execute_with_rollback(self) -> List[Any]:
        """Execute commands with rollback on failure.
        
        Returns:
            List of command results.
        """
        try:
            return self.execute()
        except Exception as e:
            self.logger.error(f"Pipeline failed, attempting rollback: {e}")
            self.rollback()
            raise
    
    def rollback(self) -> None:
        """Rollback all executed commands."""
        self.logger.info("Rolling back pipeline")
        
        while self.invoker.command_history:
            try:
                result = self.invoker.undo_last_command()
                if not result.success:
                    self.logger.warning(f"Rollback failed: {result.message}")
            except Exception as e:
                self.logger.error(f"Rollback error: {e}")
        
        self.logger.info("Rollback completed")
    
    def get_results(self) -> List[Any]:
        """Get pipeline execution results.
        
        Returns:
            List of command results.
        """
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear pipeline results."""
        self.results.clear()


# Convenience functions for common operations
def convert_dataset(
    source: Union[str, Path],
    output: Union[str, Path],
    target_format: str,
    **kwargs,
) -> ConvertDatasetCommand:
    """Create a conversion command with common defaults.
    
    Args:
        source: Source dataset path.
        output: Output dataset path.
        target_format: Target format.
        **kwargs: Additional builder options.
        
    Returns:
        Configured conversion command.
    """
    builder = ConversionBuilder().source(source).output(output).target_format(target_format)
    
    # Apply common options
    if "parallel" in kwargs and kwargs["parallel"]:
        builder.parallel()
    elif "sequential" in kwargs and kwargs["sequential"]:
        builder.sequential()
    elif "streaming" in kwargs and kwargs["streaming"]:
        builder.streaming()
    
    if "with_logging" in kwargs and kwargs["with_logging"]:
        builder.with_logging(kwargs.get("log_level", "INFO"))
    
    if "with_progress" in kwargs and kwargs["with_progress"]:
        builder.with_progress(kwargs.get("report_interval", 10))
    
    if "with_metrics" in kwargs and kwargs["with_metrics"]:
        builder.with_metrics()
    
    return builder.build()


def create_pipeline() -> PipelineBuilder:
    """Create a new pipeline builder.
    
    Returns:
        Pipeline builder instance.
    """
    return PipelineBuilder()
