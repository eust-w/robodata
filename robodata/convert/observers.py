"""Observer pattern for conversion event handling."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class ConversionEventType(Enum):
    """Types of conversion events."""
    STARTED = "started"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    EPISODE_FAILED = "episode_failed"
    PROGRESS_UPDATE = "progress_update"
    WARNING = "warning"
    ERROR = "error"
    COMPLETED = "completed"
    FINALIZED = "finalized"


@dataclass
class ConversionEvent:
    """Event data for conversion operations."""
    
    event_type: ConversionEventType
    timestamp: datetime = field(default_factory=datetime.now)
    source_path: Optional[Path] = None
    target_path: Optional[Path] = None
    episode_index: Optional[int] = None
    episode_id: Optional[str] = None
    frame_count: Optional[int] = None
    current_progress: Optional[int] = None
    total_progress: Optional[int] = None
    message: Optional[str] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of the event."""
        parts = [f"[{self.event_type.value.upper()}]"]
        
        if self.episode_index is not None:
            parts.append(f"Episode {self.episode_index}")
        
        if self.message:
            parts.append(self.message)
        
        if self.error:
            parts.append(f"Error: {self.error}")
        
        return " ".join(parts)


@runtime_checkable
class ConversionObserver(Protocol):
    """Protocol for conversion observers."""
    
    def on_event(self, event: ConversionEvent) -> None:
        """Handle a conversion event.
        
        Args:
            event: The conversion event.
        """
        ...


class ConversionObserverBase(ABC):
    """Abstract base class for conversion observers."""
    
    def __init__(self, name: str):
        """Initialize observer.
        
        Args:
            name: Observer name for logging.
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def on_event(self, event: ConversionEvent) -> None:
        """Handle a conversion event."""
        pass
    
    def _log_event(self, event: ConversionEvent, level: int = logging.INFO) -> None:
        """Log an event.
        
        Args:
            event: Event to log.
            level: Logging level.
        """
        self.logger.log(level, str(event))


class ProgressObserver(ConversionObserverBase):
    """Observer that tracks and reports conversion progress."""
    
    def __init__(self, report_interval: int = 10):
        """Initialize progress observer.
        
        Args:
            report_interval: Interval for progress reports (percentage).
        """
        super().__init__("progress")
        self.report_interval = report_interval
        self.last_report_percentage = 0
        self.start_time: Optional[datetime] = None
        self.episodes_started = 0
        self.episodes_completed = 0
        self.episodes_failed = 0
        self.total_frames_processed = 0
    
    def on_event(self, event: ConversionEvent) -> None:
        """Handle conversion events."""
        if event.event_type == ConversionEventType.STARTED:
            self.start_time = event.timestamp
            self._log_event(event)
        
        elif event.event_type == ConversionEventType.EPISODE_STARTED:
            self.episodes_started += 1
        
        elif event.event_type == ConversionEventType.EPISODE_COMPLETED:
            self.episodes_completed += 1
            if event.frame_count:
                self.total_frames_processed += event.frame_count
            self._maybe_report_progress(event)
        
        elif event.event_type == ConversionEventType.EPISODE_FAILED:
            self.episodes_failed += 1
            self._log_event(event, logging.WARNING)
        
        elif event.event_type == ConversionEventType.COMPLETED:
            self._log_completion_summary(event)
        
        elif event.event_type == ConversionEventType.ERROR:
            self._log_event(event, logging.ERROR)
    
    def _maybe_report_progress(self, event: ConversionEvent) -> None:
        """Report progress if interval has passed."""
        if not event.current_progress or not event.total_progress:
            return
        
        percentage = (event.current_progress / event.total_progress) * 100
        
        if percentage - self.last_report_percentage >= self.report_interval:
            self._report_progress(percentage, event)
            self.last_report_percentage = percentage
    
    def _report_progress(self, percentage: float, event: ConversionEvent) -> None:
        """Report current progress."""
        elapsed = datetime.now() - (self.start_time or datetime.now())
        rate = self.episodes_completed / max(1, elapsed.total_seconds())
        
        self.logger.info(
            f"Progress: {percentage:.1f}% "
            f"({event.current_progress}/{event.total_progress}) "
            f"Episodes: {self.episodes_completed} completed, {self.episodes_failed} failed "
            f"Rate: {rate:.2f} episodes/sec"
        )
    
    def _log_completion_summary(self, event: ConversionEvent) -> None:
        """Log completion summary."""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            rate = self.episodes_completed / max(1, elapsed.total_seconds())
            
            self.logger.info(
                f"Conversion completed in {elapsed}: "
                f"{self.episodes_completed} episodes, "
                f"{self.total_frames_processed} frames, "
                f"{self.episodes_failed} failed, "
                f"{rate:.2f} episodes/sec"
            )


class LoggingObserver(ConversionObserverBase):
    """Observer that logs all conversion events."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize logging observer.
        
        Args:
            log_level: Logging level for events.
        """
        super().__init__("logging")
        self.log_level = log_level
    
    def on_event(self, event: ConversionEvent) -> None:
        """Log all events."""
        if event.event_type in [ConversionEventType.ERROR]:
            self._log_event(event, logging.ERROR)
        elif event.event_type in [ConversionEventType.WARNING, ConversionEventType.EPISODE_FAILED]:
            self._log_event(event, logging.WARNING)
        else:
            self._log_event(event, self.log_level)


class MetricsObserver(ConversionObserverBase):
    """Observer that collects conversion metrics."""
    
    def __init__(self):
        """Initialize metrics observer."""
        super().__init__("metrics")
        self.metrics: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_episodes": 0,
            "completed_episodes": 0,
            "failed_episodes": 0,
            "total_frames": 0,
            "errors": [],
            "warnings": [],
            "episode_durations": [],
        }
    
    def on_event(self, event: ConversionEvent) -> None:
        """Collect metrics from events."""
        if event.event_type == ConversionEventType.STARTED:
            self.metrics["start_time"] = event.timestamp
        
        elif event.event_type == ConversionEventType.EPISODE_STARTED:
            self.metrics["total_episodes"] += 1
        
        elif event.event_type == ConversionEventType.EPISODE_COMPLETED:
            self.metrics["completed_episodes"] += 1
            if event.frame_count:
                self.metrics["total_frames"] += event.frame_count
        
        elif event.event_type == ConversionEventType.EPISODE_FAILED:
            self.metrics["failed_episodes"] += 1
        
        elif event.event_type == ConversionEventType.ERROR:
            self.metrics["errors"].append({
                "timestamp": event.timestamp,
                "episode_index": event.episode_index,
                "message": event.message,
                "error": str(event.error) if event.error else None,
            })
        
        elif event.event_type == ConversionEventType.WARNING:
            self.metrics["warnings"].append({
                "timestamp": event.timestamp,
                "episode_index": event.episode_index,
                "message": event.message,
            })
        
        elif event.event_type == ConversionEventType.COMPLETED:
            self.metrics["end_time"] = event.timestamp
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.
        
        Returns:
            Dictionary of conversion metrics.
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics["start_time"] and metrics["end_time"]:
            duration = (metrics["end_time"] - metrics["start_time"]).total_seconds()
            metrics["duration_seconds"] = duration
            
            if metrics["completed_episodes"] > 0:
                metrics["episodes_per_second"] = metrics["completed_episodes"] / duration
            else:
                metrics["episodes_per_second"] = 0.0
            
            if metrics["total_frames"] > 0:
                metrics["frames_per_second"] = metrics["total_frames"] / duration
            else:
                metrics["frames_per_second"] = 0.0
        
        # Calculate success rate
        total_episodes = metrics["total_episodes"]
        if total_episodes > 0:
            metrics["success_rate"] = metrics["completed_episodes"] / total_episodes
        else:
            metrics["success_rate"] = 0.0
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_episodes": 0,
            "completed_episodes": 0,
            "failed_episodes": 0,
            "total_frames": 0,
            "errors": [],
            "warnings": [],
            "episode_durations": [],
        }


class FileObserver(ConversionObserverBase):
    """Observer that writes events to a log file."""
    
    def __init__(self, log_file: Path):
        """Initialize file observer.
        
        Args:
            log_file: Path to log file.
        """
        super().__init__("file")
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def on_event(self, event: ConversionEvent) -> None:
        """Write event to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{event.timestamp.isoformat()} {str(event)}\n")
        except Exception as e:
            self.logger.error(f"Failed to write to log file {self.log_file}: {e}")


class ConversionEventDispatcher:
    """Dispatches conversion events to registered observers."""
    
    def __init__(self):
        """Initialize dispatcher."""
        self.observers: List[ConversionObserver] = []
        self.logger = logging.getLogger(__name__)
    
    def add_observer(self, observer: ConversionObserver) -> None:
        """Add an observer to the dispatcher.
        
        Args:
            observer: Observer to add.
        """
        if observer not in self.observers:
            self.observers.append(observer)
            self.logger.debug(f"Added observer: {observer}")
    
    def remove_observer(self, observer: ConversionObserver) -> None:
        """Remove an observer from the dispatcher.
        
        Args:
            observer: Observer to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)
            self.logger.debug(f"Removed observer: {observer}")
    
    def dispatch_event(self, event: ConversionEvent) -> None:
        """Dispatch an event to all observers.
        
        Args:
            event: Event to dispatch.
        """
        for observer in self.observers:
            try:
                observer.on_event(event)
            except Exception as e:
                self.logger.error(f"Observer {observer} failed to handle event: {e}")
    
    def clear_observers(self) -> None:
        """Remove all observers."""
        self.observers.clear()
        self.logger.debug("Cleared all observers")
    
    def get_observer_count(self) -> int:
        """Get number of registered observers.
        
        Returns:
            Number of observers.
        """
        return len(self.observers)


# Global event dispatcher
_global_dispatcher: Optional[ConversionEventDispatcher] = None


def get_global_dispatcher() -> ConversionEventDispatcher:
    """Get the global event dispatcher.
    
    Returns:
        Global ConversionEventDispatcher instance.
    """
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = ConversionEventDispatcher()
    return _global_dispatcher


def add_global_observer(observer: ConversionObserver) -> None:
    """Add an observer to the global dispatcher.
    
    Args:
        observer: Observer to add.
    """
    get_global_dispatcher().add_observer(observer)


def remove_global_observer(observer: ConversionObserver) -> None:
    """Remove an observer from the global dispatcher.
    
    Args:
        observer: Observer to remove.
    """
    get_global_dispatcher().remove_observer(observer)


def dispatch_global_event(event: ConversionEvent) -> None:
    """Dispatch an event to all global observers.
    
    Args:
        event: Event to dispatch.
    """
    get_global_dispatcher().dispatch_event(event)
