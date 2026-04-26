"""Streaming computation framework for high-throughput data processing."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from robodata.core.async_utils import AsyncConfig, AsyncStreamProcessor
from robodata.core.exceptions import RobodataError

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
K = TypeVar("K")


class StreamingError(RobodataError):
    """Streaming-specific error."""
    pass


@dataclass
class StreamMetrics:
    """Metrics for stream processing."""
    
    items_processed: int = 0
    items_per_second: float = 0.0
    total_bytes: int = 0
    bytes_per_second: float = 0.0
    processing_time: float = 0.0
    error_count: int = 0
    buffer_utilization: float = 0.0
    
    def update_rate(self, elapsed_time: float) -> None:
        """Update processing rates.
        
        Args:
            elapsed_time: Elapsed time in seconds.
        """
        if elapsed_time > 0:
            self.items_per_second = self.items_processed / elapsed_time
            self.bytes_per_second = self.total_bytes / elapsed_time


class StreamOperator(ABC, Generic[T, R]):
    """Abstract base class for stream operators."""
    
    def __init__(self, name: str):
        """Initialize operator.
        
        Args:
            name: Operator name.
        """
        self.name = name
        self.metrics = StreamMetrics()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
    
    @abstractmethod
    async def process(self, item: T) -> R:
        """Process a single item.
        
        Args:
            item: Input item.
            
        Returns:
            Processed item.
        """
        pass
    
    async def process_batch(self, items: List[T]) -> List[R]:
        """Process a batch of items.
        
        Args:
            items: Input batch.
            
        Returns:
            Processed batch.
        """
        results = []
        for item in items:
            try:
                result = await self.process(item)
                results.append(result)
                self.metrics.items_processed += 1
            except Exception as e:
                self.metrics.error_count += 1
                self._logger.error(f"Error processing item: {e}")
                # Continue processing other items
        
        return results
    
    def get_metrics(self) -> StreamMetrics:
        """Get processing metrics.
        
        Returns:
            Stream metrics.
        """
        return self.metrics


class MapOperator(StreamOperator[T, R]):
    """Map operator for transforming items."""
    
    def __init__(self, name: str, mapper: Callable[[T], R]):
        """Initialize map operator.
        
        Args:
            name: Operator name.
            mapper: Mapping function.
        """
        super().__init__(name)
        self.mapper = mapper
    
    async def process(self, item: T) -> R:
        """Apply mapping function to item.
        
        Args:
            item: Input item.
            
        Returns:
            Mapped item.
        """
        if asyncio.iscoroutinefunction(self.mapper):
            return await self.mapper(item)
        else:
            return await asyncio.to_thread(self.mapper, item)


class FilterOperator(StreamOperator[T, T]):
    """Filter operator for selecting items."""
    
    def __init__(self, name: str, predicate: Callable[[T], bool]):
        """Initialize filter operator.
        
        Args:
            name: Operator name.
            predicate: Filter predicate function.
        """
        super().__init__(name)
        self.predicate = predicate
    
    async def process(self, item: T) -> Optional[T]:
        """Apply filter predicate to item.
        
        Args:
            item: Input item.
            
        Returns:
            Item if predicate passes, None otherwise.
        """
        if asyncio.iscoroutinefunction(self.predicate):
            should_keep = await self.predicate(item)
        else:
            should_keep = await asyncio.to_thread(self.predicate, item)
        
        return item if should_keep else None
    
    async def process_batch(self, items: List[T]) -> List[T]:
        """Filter batch of items.
        
        Args:
            items: Input batch.
            
        Returns:
            Filtered batch.
        """
        results = []
        for item in items:
            try:
                result = await self.process(item)
                if result is not None:
                    results.append(result)
                    self.metrics.items_processed += 1
            except Exception as e:
                self.metrics.error_count += 1
                self._logger.error(f"Error filtering item: {e}")
        
        return results


class ReduceOperator(StreamOperator[T, R]):
    """Reduce operator for aggregating items."""
    
    def __init__(
        self,
        name: str,
        reducer: Callable[[R, T], R],
        initializer: Optional[R] = None,
    ):
        """Initialize reduce operator.
        
        Args:
            name: Operator name.
            reducer: Reducing function.
            initializer: Initial accumulator value.
        """
        super().__init__(name)
        self.reducer = reducer
        self.initializer = initializer
        self._accumulator: Optional[R] = initializer
    
    async def process(self, item: T) -> Optional[R]:
        """Reduce item into accumulator.
        
        Args:
            item: Input item.
            
        Returns:
            Current accumulator value or None.
        """
        if self._accumulator is None:
            self._accumulator = item
        else:
            if asyncio.iscoroutinefunction(self.reducer):
                self._accumulator = await self.reducer(self._accumulator, item)
            else:
                self._accumulator = await asyncio.to_thread(self.reducer, self._accumulator, item)
        
        self.metrics.items_processed += 1
        return self._accumulator
    
    def get_result(self) -> R:
        """Get final reduction result.
        
        Returns:
            Accumulator value.
        
        Raises:
            StreamingError: If no items were processed.
        """
        if self._accumulator is None:
            raise StreamingError("No items were processed for reduction")
        return self._accumulator


class WindowOperator(StreamOperator[T, List[T]]):
    """Window operator for time-based or count-based windows."""
    
    def __init__(
        self,
        name: str,
        window_size: int,
        slide_size: Optional[int] = None,
    ):
        """Initialize window operator.
        
        Args:
            name: Operator name.
            window_size: Window size in items.
            slide_size: Slide size in items (defaults to window_size).
        """
        super().__init__(name)
        self.window_size = window_size
        self.slide_size = slide_size or window_size
        self._buffer: List[T] = []
        self._position = 0
    
    async def process(self, item: T) -> Optional[List[T]]:
        """Add item to buffer and emit windows.
        
        Args:
            item: Input item.
            
        Returns:
            Window of items or None.
        """
        self._buffer.append(item)
        self.metrics.items_processed += 1
        
        # Check if we have enough items for a window
        if len(self._buffer) >= self.window_size:
            # Emit window
            window = self._buffer[:self.window_size]
            
            # Slide buffer
            if self.slide_size < len(self._buffer):
                self._buffer = self._buffer[self.slide_size:]
            else:
                self._buffer.clear()
            
            return window
        
        return None


class GroupByOperator(StreamOperator[T, Dict[K, List[T]]]):
    """Group by operator for partitioning items."""
    
    def __init__(self, name: str, key_extractor: Callable[[T], K]):
        """Initialize group by operator.
        
        Args:
            name: Operator name.
            key_extractor: Function to extract grouping key.
        """
        super().__init__(name)
        self.key_extractor = key_extractor
        self._groups: Dict[K, List[T]] = {}
    
    async def process(self, item: T) -> Optional[Dict[K, List[T]]]:
        """Group item by key.
        
        Args:
            item: Input item.
            
        Returns:
            Current groups or None.
        """
        if asyncio.iscoroutinefunction(self.key_extractor):
            key = await self.key_extractor(item)
        else:
            key = await asyncio.to_thread(self.key_extractor, item)
        
        if key not in self._groups:
            self._groups[key] = []
        
        self._groups[key].append(item)
        self.metrics.items_processed += 1
        
        return self._groups
    
    def get_groups(self) -> Dict[K, List[T]]:
        """Get current groups.
        
        Returns:
            Grouped items.
        """
        return self._groups.copy()


class StreamPipeline:
    """High-performance stream processing pipeline."""
    
    def __init__(self, name: str, config: AsyncConfig):
        """Initialize pipeline.
        
        Args:
            name: Pipeline name.
            config: Async configuration.
        """
        self.name = name
        self.config = config
        self.operators: List[StreamOperator] = []
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        self._metrics = StreamMetrics()
    
    def add_operator(self, operator: StreamOperator) -> "StreamPipeline":
        """Add operator to pipeline.
        
        Args:
            operator: Operator to add.
            
        Returns:
            Pipeline instance for chaining.
        """
        self.operators.append(operator)
        self._logger.debug(f"Added operator: {operator.name}")
        return self
    
    def map(self, mapper: Callable[[T], R], name: str = "map") -> "StreamPipeline":
        """Add map operator.
        
        Args:
            mapper: Mapping function.
            name: Operator name.
            
        Returns:
            Pipeline instance for chaining.
        """
        operator = MapOperator(name, mapper)
        return self.add_operator(operator)
    
    def filter(self, predicate: Callable[[T], bool], name: str = "filter") -> "StreamPipeline":
        """Add filter operator.
        
        Args:
            predicate: Filter predicate.
            name: Operator name.
            
        Returns:
            Pipeline instance for chaining.
        """
        operator = FilterOperator(name, predicate)
        return self.add_operator(operator)
    
    def reduce(
        self,
        reducer: Callable[[R, T], R],
        initializer: Optional[R] = None,
        name: str = "reduce",
    ) -> "StreamPipeline":
        """Add reduce operator.
        
        Args:
            reducer: Reducing function.
            initializer: Initial value.
            name: Operator name.
            
        Returns:
            Pipeline instance for chaining.
        """
        operator = ReduceOperator(name, reducer, initializer)
        return self.add_operator(operator)
    
    def window(
        self,
        window_size: int,
        slide_size: Optional[int] = None,
        name: str = "window",
    ) -> "StreamPipeline":
        """Add window operator.
        
        Args:
            window_size: Window size.
            slide_size: Slide size.
            name: Operator name.
            
        Returns:
            Pipeline instance for chaining.
        """
        operator = WindowOperator(name, window_size, slide_size)
        return self.add_operator(operator)
    
    def group_by(
        self,
        key_extractor: Callable[[T], K],
        name: str = "group_by",
    ) -> "StreamPipeline":
        """Add group by operator.
        
        Args:
            key_extractor: Key extraction function.
            name: Operator name.
            
        Returns:
            Pipeline instance for chaining.
        """
        operator = GroupByOperator(name, key_extractor)
        return self.add_operator(operator)
    
    async def process(
        self,
        source: AsyncIterator[T],
        sink: Optional[Callable[[Any], Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Process stream through pipeline.
        
        Args:
            source: Source async iterator.
            sink: Optional sink function.
            
        Yields:
            Processed items.
        """
        import time
        start_time = time.time()
        
        if not self.operators:
            # No operators, pass through
            async for item in source:
                if sink:
                    await self._call_sink(sink, item)
                yield item
            return
        
        # Create stream processor
        processor = AsyncStreamProcessor(self.config)
        
        # Process through operators sequentially
        current_stream = source
        
        for i, operator in enumerate(self.operators):
            self._logger.debug(f"Processing through operator: {operator.name}")
            
            # Create processing function for this operator
            async def process_item(item: Any, op: StreamOperator = operator) -> Any:
                return await op.process(item)
            
            # Process stream
            current_stream = processor.process_stream(
                current_stream,
                process_item,
                sink,
            )
        
        # Yield final results
        async for result in current_stream:
            if sink:
                await self._call_sink(sink, result)
            yield result
        
        # Update metrics
        elapsed_time = time.time() - start_time
        self._metrics.processing_time = elapsed_time
        
        # Aggregate operator metrics
        total_items = sum(op.metrics.items_processed for op in self.operators)
        total_errors = sum(op.metrics.error_count for op in self.operators)
        
        self._metrics.items_processed = total_items
        self._metrics.error_count = total_errors
        self._metrics.update_rate(elapsed_time)
        
        self._logger.info(
            f"Pipeline '{self.name}' completed: "
            f"{total_items} items, {total_errors} errors, "
            f"{elapsed_time:.2f}s, {self._metrics.items_per_second:.2f} items/s"
        )
    
    async def _call_sink(self, sink: Callable[[Any], Any], item: Any) -> None:
        """Call sink function safely.
        
        Args:
            sink: Sink function.
            item: Item to send to sink.
        """
        try:
            if asyncio.iscoroutinefunction(sink):
                await sink(item)
            else:
                sink(item)
        except Exception as e:
            self._logger.error(f"Sink error: {e}")
    
    def get_metrics(self) -> StreamMetrics:
        """Get pipeline metrics.
        
        Returns:
            Pipeline metrics.
        """
        return self._metrics
    
    def get_operator_metrics(self) -> Dict[str, StreamMetrics]:
        """Get metrics for all operators.
        
        Returns:
            Operator metrics dictionary.
        """
        return {op.name: op.get_metrics() for op in self.operators}


class StreamBuilder:
    """Builder for creating stream pipelines."""
    
    def __init__(self, config: AsyncConfig):
        """Initialize builder.
        
        Args:
            config: Async configuration.
        """
        self.config = config
        self._pipelines: List[StreamPipeline] = []
    
    def create_pipeline(self, name: str) -> StreamPipeline:
        """Create a new stream pipeline.
        
        Args:
            name: Pipeline name.
            
        Returns:
            Stream pipeline instance.
        """
        pipeline = StreamPipeline(name, self.config)
        self._pipelines.append(pipeline)
        return pipeline
    
    def get_pipelines(self) -> List[StreamPipeline]:
        """Get all created pipelines.
        
        Returns:
            List of pipelines.
        """
        return self._pipelines.copy()


# Utility functions
async def create_stream_from_iterable(
    items: List[T],
    delay: float = 0.0,
) -> AsyncIterator[T]:
    """Create async stream from iterable.
    
    Args:
        items: Iterable of items.
        delay: Delay between items.
        
    Yields:
        Items from iterable.
    """
    for item in items:
        yield item
        if delay > 0:
            await asyncio.sleep(delay)


async def create_stream_from_file(
    file_path: str,
    chunk_size: int = 8192,
    encoding: str = "utf-8",
) -> AsyncIterator[str]:
    """Create async stream from file.
    
    Args:
        file_path: Path to file.
        chunk_size: Chunk size.
        encoding: File encoding.
        
    Yields:
        File lines or chunks.
    """
    from robodata.core.async_utils import AsyncFileProcessor
    
    processor = AsyncFileProcessor(AsyncConfig())
    path = __import__("pathlib").Path(file_path)
    
    if path.suffix in [".txt", ".log", ".jsonl"]:
        # Read line by line
        async for chunk in processor.read_file_chunks(path, chunk_size):
            lines = chunk.decode(encoding).splitlines()
            for line in lines:
                yield line
    else:
        # Read as binary chunks
        async for chunk in processor.read_file_chunks(path, chunk_size):
            yield chunk.decode(encoding, errors="ignore")


async def merge_streams(
    *streams: AsyncIterator[T],
    strategy: str = "round_robin",
) -> AsyncIterator[T]:
    """Merge multiple streams.
    
    Args:
        *streams: Streams to merge.
        strategy: Merge strategy ("round_robin", "priority").
        
    Yields:
        Items from merged streams.
    """
    if strategy == "round_robin":
        queues = [asyncio.Queue() for _ in streams]
        
        # Create consumer tasks
        async def consumer(stream: AsyncIterator[T], queue: asyncio.Queue):
            async for item in stream:
                await queue.put(item)
            await queue.put(None)  # Sentinel
        
        consumers = [
            asyncio.create_task(consumer(stream, queue))
            for stream, queue in zip(streams, queues)
        ]
        
        # Round-robin merge
        active_queues = len(queues)
        while active_queues > 0:
            for i, queue in enumerate(queues):
                try:
                    item = queue.get_nowait()
                    if item is None:
                        active_queues -= 1
                    else:
                        yield item
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0)  # Yield control
        
        # Wait for consumers
        await asyncio.gather(*consumers)
    
    elif strategy == "priority":
        # Priority: yield from first stream until exhausted, then next
        for stream in streams:
            async for item in stream:
                yield item


async def buffer_stream(
    stream: AsyncIterator[T],
    buffer_size: int = 100,
    strategy: str = "sliding",
) -> AsyncIterator[List[T]]:
    """Buffer stream into batches.
    
    Args:
        stream: Input stream.
        buffer_size: Buffer size.
        strategy: Buffer strategy ("sliding", "tumbling").
        
    Yields:
        Buffered batches.
    """
    buffer = []
    
    async for item in stream:
        buffer.append(item)
        
        if len(buffer) >= buffer_size:
            yield buffer.copy()
            
            if strategy == "tumbling":
                buffer.clear()
            elif strategy == "sliding":
                buffer.pop(0)
    
    # Yield remaining items
    if buffer:
        yield buffer
