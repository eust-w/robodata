"""Command pattern for conversion operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from robodata.config.models import ConversionConfig
from robodata.convert.observers import ConversionEvent, ConversionEventType, ConversionObserver
from robodata.convert.strategies import ConversionContext, ConversionStrategyBase, ConversionStrategySelector
from robodata.formats.factory import get_global_factory
from robodata.formats.base import FormatReader, FormatWriter

logger = logging.getLogger(__name__)


@runtime_checkable
class Command(Protocol):
    """Protocol for conversion commands."""
    
    def execute(self) -> Dict[str, Any]:
        """Execute the command.
        
        Returns:
            Command execution result.
        """
        ...
    
    def undo(self) -> Dict[str, Any]:
        """Undo the command.
        
        Returns:
            Undo operation result.
        """
        ...


@dataclass
class CommandResult:
    """Result of command execution."""
    
    command_id: str
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize command ID if not provided."""
        if not self.command_id:
            self.command_id = str(uuid4())


class ConversionCommand(ABC):
    """Abstract base class for conversion commands."""
    
    def __init__(self, command_id: Optional[str] = None):
        """Initialize command.
        
        Args:
            command_id: Optional command identifier.
        """
        self.command_id = command_id or str(uuid4())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.observers: List[ConversionObserver] = []
        self.executed = False
        self.result: Optional[CommandResult] = None
    
    def add_observer(self, observer: ConversionObserver) -> None:
        """Add an observer to this command.
        
        Args:
            observer: Observer to add.
        """
        self.observers.append(observer)
    
    def remove_observer(self, observer: ConversionObserver) -> None:
        """Remove an observer from this command.
        
        Args:
            observer: Observer to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def _notify_observers(self, event: ConversionEvent) -> None:
        """Notify all observers of an event.
        
        Args:
            event: Event to notify about.
        """
        for observer in self.observers:
            try:
                observer.on_event(event)
            except Exception as e:
                self.logger.error(f"Observer failed to handle event: {e}")
    
    @abstractmethod
    def execute(self) -> CommandResult:
        """Execute the command."""
        pass
    
    @abstractmethod
    def undo(self) -> CommandResult:
        """Undo the command."""
        pass


class ConvertDatasetCommand(ConversionCommand):
    """Command to convert a dataset between formats."""
    
    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        source_format: str,
        target_format: str,
        config: ConversionConfig,
        command_id: Optional[str] = None,
    ):
        """Initialize conversion command.
        
        Args:
            source_path: Source dataset path.
            output_path: Output dataset path.
            source_format: Source format name.
            target_format: Target format name.
            config: Conversion configuration.
            command_id: Optional command identifier.
        """
        super().__init__(command_id)
        self.source_path = source_path
        self.output_path = output_path
        self.source_format = source_format
        self.target_format = target_format
        self.config = config
        
        # Will be populated during execution
        self.reader: Optional[FormatReader] = None
        self.writer: Optional[FormatWriter] = None
        self.strategy: Optional[ConversionStrategyBase] = None
        self.context: Optional[ConversionContext] = None
    
    def execute(self) -> CommandResult:
        """Execute the conversion command."""
        import time
        start_time = time.time()
        
        try:
            # Notify start
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.STARTED,
                source_path=self.source_path,
                target_path=self.output_path,
                message=f"Converting {self.source_format} to {self.target_format}"
            ))
            
            # Validate formats
            factory = get_global_factory()
            factory.validate_format_support(self.source_format, self.target_format)
            
            # Get reader and writer
            self.reader = factory.get_reader(self.source_format)
            self.writer = factory.get_writer(self.target_format)
            
            # Inspect dataset
            dataset_info = self.reader.inspect(self.source_path)
            
            # Create conversion context
            self.context = ConversionContext(
                source_path=self.source_path,
                output_path=self.output_path,
                source_format=self.source_format,
                target_format=self.target_format,
                dataset_info=dataset_info,
                config=self.config.__dict__,
            )
            
            # Select conversion strategy
            strategy_selector = ConversionStrategySelector()
            self.strategy = strategy_selector.select_strategy(self.context)
            
            # Prepare output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Execute conversion
            result_data = self.strategy.execute(
                self.context,
                self.reader,
                self.writer,
                progress_callback=self._create_progress_callback(),
            )
            
            # Create success result
            execution_time = time.time() - start_time
            self.result = CommandResult(
                command_id=self.command_id,
                success=True,
                message=f"Successfully converted {result_data.get('episodes_converted', 0)} episodes",
                data=result_data,
                execution_time=execution_time,
            )
            
            # Notify completion
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.COMPLETED,
                source_path=self.source_path,
                target_path=self.output_path,
                message=f"Conversion completed successfully"
            ))
            
            self.executed = True
            return self.result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Conversion failed: {e}"
            
            # Create failure result
            self.result = CommandResult(
                command_id=self.command_id,
                success=False,
                message=error_message,
                execution_time=execution_time,
                errors=[str(e)],
            )
            
            # Notify error
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.ERROR,
                source_path=self.source_path,
                target_path=self.output_path,
                message=error_message,
                error=e
            ))
            
            self.logger.error(f"Conversion command failed: {e}")
            return self.result
    
    def undo(self) -> CommandResult:
        """Undo the conversion by removing output files."""
        if not self.executed:
            return CommandResult(
                command_id=self.command_id,
                success=False,
                message="Command was not executed, nothing to undo",
                errors=["Command not executed"]
            )
        
        try:
            # Remove output directory if it exists
            if self.output_path.exists():
                import shutil
                shutil.rmtree(self.output_path)
            
            return CommandResult(
                command_id=self.command_id,
                success=True,
                message=f"Successfully removed output: {self.output_path}",
            )
            
        except Exception as e:
            return CommandResult(
                command_id=self.command_id,
                success=False,
                message=f"Failed to undo conversion: {e}",
                errors=[str(e)]
            )
    
    def _create_progress_callback(self):
        """Create a progress callback that notifies observers."""
        def progress_callback(stage: str, current: int, total: int, message: str = ""):
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.PROGRESS_UPDATE,
                source_path=self.source_path,
                target_path=self.output_path,
                current_progress=current,
                total_progress=total,
                message=f"{stage}: {current}/{total} {message}"
            ))
        
        return progress_callback


class InspectDatasetCommand(ConversionCommand):
    """Command to inspect a dataset."""
    
    def __init__(
        self,
        dataset_path: Path,
        format_hint: Optional[str] = None,
        deep_scan: bool = False,
        command_id: Optional[str] = None,
    ):
        """Initialize inspection command.
        
        Args:
            dataset_path: Path to dataset.
            format_hint: Optional format hint.
            deep_scan: Whether to perform deep scan.
            command_id: Optional command identifier.
        """
        super().__init__(command_id)
        self.dataset_path = dataset_path
        self.format_hint = format_hint
        self.deep_scan = deep_scan
        
        self.reader: Optional[FormatReader] = None
        self.detected_format: Optional[str] = None
    
    def execute(self) -> CommandResult:
        """Execute the inspection command."""
        import time
        start_time = time.time()
        
        try:
            # Notify start
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.STARTED,
                source_path=self.dataset_path,
                message=f"Inspecting dataset"
            ))
            
            # Detect format
            factory = get_global_factory()
            if self.format_hint:
                self.detected_format = self.format_hint
            else:
                self.detected_format = factory.detect_format(self.dataset_path)
            
            # Get reader
            self.reader = factory.get_reader(self.detected_format)
            
            # Inspect dataset
            dataset_info = self.reader.inspect(self.dataset_path)
            
            # Additional deep scan if requested
            if self.deep_scan:
                episode_count = self.reader.get_episode_count(self.dataset_path)
                if episode_count >= 0:
                    dataset_info.num_episodes = episode_count
            
            # Create result
            execution_time = time.time() - start_time
            result_data = {
                "dataset_info": dataset_info,
                "detected_format": self.detected_format,
                "deep_scan": self.deep_scan,
            }
            
            self.result = CommandResult(
                command_id=self.command_id,
                success=True,
                message=f"Successfully inspected dataset: {self.detected_format}",
                data=result_data,
                execution_time=execution_time,
            )
            
            # Notify completion
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.COMPLETED,
                source_path=self.dataset_path,
                message=f"Inspection completed: {dataset_info.num_episodes} episodes"
            ))
            
            self.executed = True
            return self.result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Inspection failed: {e}"
            
            self.result = CommandResult(
                command_id=self.command_id,
                success=False,
                message=error_message,
                execution_time=execution_time,
                errors=[str(e)],
            )
            
            # Notify error
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.ERROR,
                source_path=self.dataset_path,
                message=error_message,
                error=e
            ))
            
            return self.result
    
    def undo(self) -> CommandResult:
        """Undo inspection (no-op)."""
        return CommandResult(
            command_id=self.command_id,
            success=True,
            message="Inspection command has no side effects to undo",
        )


class ValidateDatasetCommand(ConversionCommand):
    """Command to validate a dataset."""
    
    def __init__(
        self,
        dataset_path: Path,
        format_hint: Optional[str] = None,
        strict: bool = False,
        command_id: Optional[str] = None,
    ):
        """Initialize validation command.
        
        Args:
            dataset_path: Path to dataset.
            format_hint: Optional format hint.
            strict: Whether to use strict validation.
            command_id: Optional command identifier.
        """
        super().__init__(command_id)
        self.dataset_path = dataset_path
        self.format_hint = format_hint
        self.strict = strict
        
        self.reader: Optional[FormatReader] = None
        self.detected_format: Optional[str] = None
    
    def execute(self) -> CommandResult:
        """Execute the validation command."""
        import time
        start_time = time.time()
        
        try:
            # Notify start
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.STARTED,
                source_path=self.dataset_path,
                message=f"Validating dataset"
            ))
            
            # Detect format
            factory = get_global_factory()
            if self.format_hint:
                self.detected_format = self.format_hint
            else:
                self.detected_format = factory.detect_format(self.dataset_path)
            
            # Get reader
            self.reader = factory.get_reader(self.detected_format)
            
            # Validate by attempting to read episodes
            validation_errors = []
            validation_warnings = []
            episode_count = 0
            
            for episode_idx, episode in enumerate(self.reader.read_episodes(self.dataset_path)):
                try:
                    # Basic validation
                    if not episode.episode_id:
                        validation_errors.append(f"Episode {episode_idx}: Missing episode ID")
                    
                    # Validate frames
                    frame_count = 0
                    for frame in episode.frames():
                        frame_count += 1
                        if frame.frame_index < 0:
                            validation_errors.append(f"Episode {episode_idx}, Frame {frame.frame_index}: Invalid frame index")
                    
                    if frame_count == 0:
                        validation_warnings.append(f"Episode {episode_idx}: No frames found")
                    
                    episode_count += 1
                    
                    # Limit validation for large datasets
                    if not self.strict and episode_count >= 100:
                        break
                        
                except Exception as e:
                    validation_errors.append(f"Episode {episode_idx}: {e}")
                    if self.strict:
                        break
            
            # Create result
            execution_time = time.time() - start_time
            result_data = {
                "detected_format": self.detected_format,
                "episodes_validated": episode_count,
                "validation_errors": validation_errors,
                "validation_warnings": validation_warnings,
                "strict_mode": self.strict,
            }
            
            success = len(validation_errors) == 0
            message = f"Validation {'passed' if success else 'failed'}: {episode_count} episodes checked"
            
            self.result = CommandResult(
                command_id=self.command_id,
                success=success,
                message=message,
                data=result_data,
                execution_time=execution_time,
                errors=validation_errors,
                warnings=validation_warnings,
            )
            
            # Notify completion
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.COMPLETED,
                source_path=self.dataset_path,
                message=message
            ))
            
            self.executed = True
            return self.result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Validation failed: {e}"
            
            self.result = CommandResult(
                command_id=self.command_id,
                success=False,
                message=error_message,
                execution_time=execution_time,
                errors=[str(e)],
            )
            
            # Notify error
            self._notify_observers(ConversionEvent(
                event_type=ConversionEventType.ERROR,
                source_path=self.dataset_path,
                message=error_message,
                error=e
            ))
            
            return self.result
    
    def undo(self) -> CommandResult:
        """Undo validation (no-op)."""
        return CommandResult(
            command_id=self.command_id,
            success=True,
            message="Validation command has no side effects to undo",
        )


class CommandInvoker:
    """Invoker for executing and managing commands."""
    
    def __init__(self):
        """Initialize invoker."""
        self.command_history: List[Command] = []
        self.logger = logging.getLogger(__name__)
    
    def execute_command(self, command: Command) -> CommandResult:
        """Execute a command and add to history.
        
        Args:
            command: Command to execute.
            
        Returns:
            Command execution result.
        """
        self.logger.info(f"Executing command: {command.__class__.__name__}")
        
        try:
            result = command.execute()
            self.command_history.append(command)
            
            self.logger.info(
                f"Command executed: {result.success} in {result.execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise
    
    def undo_last_command(self) -> Optional[CommandResult]:
        """Undo the last executed command.
        
        Returns:
            Undo operation result, or None if no commands to undo.
        """
        if not self.command_history:
            self.logger.warning("No commands to undo")
            return None
        
        last_command = self.command_history.pop()
        self.logger.info(f"Undoing command: {last_command.__class__.__name__}")
        
        try:
            result = last_command.undo()
            self.logger.info(f"Command undone: {result.success}")
            return result
            
        except Exception as e:
            self.logger.error(f"Command undo failed: {e}")
            raise
    
    def get_command_history(self) -> List[str]:
        """Get command history.
        
        Returns:
            List of command descriptions.
        """
        return [
            f"{cmd.__class__.__name__} ({cmd.command_id})"
            for cmd in self.command_history
        ]
    
    def clear_history(self) -> None:
        """Clear command history."""
        self.command_history.clear()
        self.logger.info("Command history cleared")
