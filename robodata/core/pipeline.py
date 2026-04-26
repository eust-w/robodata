"""Core abstractions for Robodata data processing pipeline.

This module defines the foundational interfaces and abstractions that enable
format-agnostic data processing while maintaining performance and extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from robodata.core.exceptions import RobodataError

T = TypeVar("T")


class ProcessingMode(Enum):
    """Processing mode for data operations."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for data processing operations."""
    mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    num_workers: int = 1
    chunk_size: int = 1000
    memory_limit_mb: int = 2048
    fail_fast: bool = True
    progress_callback: callable | None = None


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    items_processed: int = 0
    items_failed: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_items(self) -> int:
        return self.items_processed + self.items_failed


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources."""
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over data items."""
        ...
    
    def __len__(self) -> int:
        """Return total number of items."""
        ...


@runtime_checkable
class DataSink(Protocol[T]):
    """Protocol for data sinks."""
    
    def write(self, item: T, index: int) -> None:
        """Write a single data item."""
        ...
    
    def finalize(self) -> None:
        """Finalize the sink after all items are written."""
        ...


class Processor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process_item(self, item: Any) -> Any:
        """Process a single data item."""
        ...
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the processor."""
        ...
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the processor."""
        ...


class ProcessingPipeline:
    """High-performance data processing pipeline with adaptive execution."""
    
    def __init__(
        self,
        processor: Processor,
        config: ProcessingConfig | None = None
    ):
        self.processor = processor
        self.config = config or ProcessingConfig()
        self._setup_complete = False
    
    @contextmanager
    def _managed_processor(self):
        """Context manager for processor lifecycle."""
        try:
            if not self._setup_complete:
                self.processor.setup()
                self._setup_complete = True
            yield self.processor
        finally:
            if self._setup_complete:
                self.processor.cleanup()
                self._setup_complete = False
    
    def execute(
        self,
        source: DataSource,
        sink: DataSink | None = None
    ) -> ProcessingResult:
        """Execute the processing pipeline."""
        result = ProcessingResult(success=False)
        
        with self._managed_processor():
            if self.config.mode == ProcessingMode.SEQUENTIAL:
                result = self._execute_sequential(source, sink)
            elif self.config.mode == ProcessingMode.PARALLEL:
                result = self._execute_parallel(source, sink)
            elif self.config.mode == ProcessingMode.DISTRIBUTED:
                result = self._execute_distributed(source, sink)
            else:
                raise RobodataError(f"Unsupported processing mode: {self.config.mode}")
        
        return result
    
    def _execute_sequential(
        self,
        source: DataSource,
        sink: DataSink | None
    ) -> ProcessingResult:
        """Execute processing sequentially."""
        result = ProcessingResult(success=True)
        
        for index, item in enumerate(source):
            try:
                processed_item = self.processor.process_item(item)
                
                if sink is not None:
                    sink.write(processed_item, index)
                
                result.items_processed += 1
                
                if self.config.progress_callback:
                    self.config.progress_callback(index + 1, len(source))
                    
            except Exception as e:
                result.items_failed += 1
                error_msg = f"Item {index}: {e}"
                result.errors.append(error_msg)
                
                if self.config.fail_fast:
                    result.success = False
                    break
        
        if sink is not None:
            try:
                sink.finalize()
            except Exception as e:
                result.errors.append(f"Finalization failed: {e}")
                result.success = False
        
        return result.success and result.items_failed == 0
    
    def _execute_parallel(
        self,
        source: DataSource,
        sink: DataSink | None
    ) -> ProcessingResult:
        """Execute processing in parallel."""
        # Implementation for parallel processing
        # This would use multiprocessing or concurrent.futures
        return self._execute_sequential(source, sink)  # Placeholder
    
    def _execute_distributed(
        self,
        source: DataSource,
        sink: DataSink | None
    ) -> ProcessingResult:
        """Execute processing in distributed mode."""
        # Implementation for distributed processing (e.g., Ray)
        return self._execute_sequential(source, sink)  # Placeholder


class AdaptiveProcessor(Processor):
    """Processor that adapts its behavior based on data characteristics."""
    
    def __init__(self, base_processor: Processor):
        self.base_processor = base_processor
        self._adaptation_applied = False
    
    def process_item(self, item: Any) -> Any:
        """Process item with adaptive behavior."""
        if not self._adaptation_applied:
            self._adapt_to_item(item)
            self._adaptation_applied = True
        
        return self.base_processor.process_item(item)
    
    def setup(self) -> None:
        """Setup the base processor."""
        self.base_processor.setup()
    
    def cleanup(self) -> None:
        """Cleanup the base processor."""
        self.base_processor.cleanup()
    
    def _adapt_to_item(self, item: Any) -> None:
        """Adapt processing strategy based on item characteristics."""
        # Implementation for adaptive behavior
        pass


class BatchProcessor(Processor):
    """Processor that handles items in batches for better performance."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self._current_batch = []
    
    def process_item(self, item: Any) -> Any:
        """Process item in batch context."""
        self._current_batch.append(item)
        
        if len(self._current_batch) >= self.batch_size:
            result = self._process_batch(self._current_batch)
            self._current_batch.clear()
            return result
        
        return None  # Batch not ready yet
    
    def finalize_batch(self) -> Any:
        """Process remaining items in current batch."""
        if self._current_batch:
            result = self._process_batch(self._current_batch)
            self._current_batch.clear()
            return result
        return None
    
    @abstractmethod
    def _process_batch(self, batch: list[Any]) -> Any:
        """Process a batch of items."""
        ...


class CachingProcessor(Processor):
    """Processor with intelligent caching capabilities."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self._cache: dict[str, Any] = {}
        self._cache_order: list[str] = []
    
    def process_item(self, item: Any) -> Any:
        """Process item with caching."""
        cache_key = self._get_cache_key(item)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._process_uncached(item)
        self._add_to_cache(cache_key, result)
        return result
    
    @abstractmethod
    def _process_uncached(self, item: Any) -> Any:
        """Process item without caching."""
        ...
    
    @abstractmethod
    def _get_cache_key(self, item: Any) -> str:
        """Generate cache key for item."""
        ...
    
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add item to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Evict oldest item
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        self._cache_order.append(key)


# Factory functions for creating common processors
def create_adaptive_pipeline(
    processor: Processor,
    config: ProcessingConfig | None = None
) -> ProcessingPipeline:
    """Create an adaptive processing pipeline."""
    adaptive_processor = AdaptiveProcessor(processor)
    return ProcessingPipeline(adaptive_processor, config)


def create_caching_pipeline(
    processor: Processor,
    cache_size: int = 1000,
    config: ProcessingConfig | None = None
) -> ProcessingPipeline:
    """Create a caching processing pipeline."""
    caching_processor = CachingProcessor(cache_size)
    # Note: This would need composition to combine caching with base processing
    return ProcessingPipeline(caching_processor, config)
