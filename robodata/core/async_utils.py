"""Asynchronous utilities for high-performance processing."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from asyncio import Queue, Task, gather
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar, Union

from robodata.core.exceptions import RobodataError

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class AsyncRobodataError(RobodataError):
    """Async-specific error."""
    pass


@dataclass
class AsyncConfig:
    """Configuration for async operations."""
    
    max_concurrent_tasks: int = 10
    queue_size: int = 1000
    timeout: Optional[float] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    enable_progress: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class AsyncWorkerPool:
    """High-performance async worker pool with queue management."""
    
    def __init__(self, config: AsyncConfig):
        """Initialize worker pool.
        
        Args:
            config: Async configuration.
        """
        self.config = config
        self._queue: Optional[Queue] = None
        self._workers: List[Task] = []
        self._running = False
        self._processed_count = 0
        self._error_count = 0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def __aenter__(self) -> "AsyncWorkerPool":
        """Enter async context manager."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        
        self._queue = Queue(maxsize=self.config.queue_size)
        self._running = True
        
        # Start worker tasks
        for worker_id in range(self.config.max_concurrent_tasks):
            worker_task = asyncio.create_task(
                self._worker(f"worker-{worker_id}")
            )
            self._workers.append(worker_task)
        
        self._logger.info(
            f"Started async worker pool with {self.config.max_concurrent_tasks} workers"
        )
    
    async def stop(self) -> None:
        """Stop the worker pool."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        self._queue = None
        
        self._logger.info(
            f"Stopped worker pool: {self._processed_count} processed, {self._error_count} errors"
        )
    
    async def submit(self, item: T) -> None:
        """Submit an item to the worker pool.
        
        Args:
            item: Item to process.
            
        Raises:
            AsyncRobodataError: If queue is full or pool is not running.
        """
        if not self._running or not self._queue:
            raise AsyncRobodataError("Worker pool is not running")
        
        try:
            await self._queue.put(item)
        except asyncio.QueueFull:
            raise AsyncRobodataError("Worker queue is full")
    
    async def submit_batch(self, items: List[T]) -> None:
        """Submit a batch of items.
        
        Args:
            items: Items to submit.
        """
        for item in items:
            await self.submit(item)
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine.
        
        Args:
            worker_name: Worker identifier.
        """
        self._logger.debug(f"Starting worker: {worker_name}")
        
        while self._running:
            try:
                # Get item from queue
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                # Process item
                await self._process_item(item, worker_name)
                
                # Mark task as done
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue
            except Exception as e:
                self._error_count += 1
                self._logger.error(f"Worker {worker_name} error: {e}")
                
                if self.config.retry_attempts > 0:
                    # Retry logic would be implemented here
                    pass
    
    async def _process_item(self, item: T, worker_name: str) -> None:
        """Process a single item.
        
        Args:
            item: Item to process.
            worker_name: Worker identifier.
        """
        # This would be overridden by subclasses
        self._processed_count += 1
        await asyncio.sleep(0)  # Yield control
    
    @property
    def processed_count(self) -> int:
        """Get number of processed items."""
        return self._processed_count
    
    @property
    def error_count(self) -> int:
        """Get number of errors."""
        return self._error_count
    
    @property
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._running


class AsyncStreamProcessor:
    """High-performance stream processor with backpressure control."""
    
    def __init__(self, config: AsyncConfig):
        """Initialize stream processor.
        
        Args:
            config: Async configuration.
        """
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def process_stream(
        self,
        source: AsyncIterator[T],
        processor: Callable[[T], R],
        sink: Optional[Callable[[R], Any]] = None,
    ) -> AsyncGenerator[R, None]:
        """Process a stream of items.
        
        Args:
            source: Source async iterator.
            processor: Processing function.
            sink: Optional sink function.
            
        Yields:
            Processed items.
        """
        batch: List[T] = []
        processed_count = 0
        
        async for item in source:
            batch.append(item)
            
            # Process batch when full
            if len(batch) >= self.config.batch_size:
                async for result in self._process_batch(batch, processor, sink):
                    processed_count += 1
                    yield result
                
                batch.clear()
        
        # Process remaining items
        if batch:
            async for result in self._process_batch(batch, processor, sink):
                processed_count += 1
                yield result
        
        self._logger.info(f"Stream processing completed: {processed_count} items")
    
    async def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[T], R],
        sink: Optional[Callable[[R], Any]],
    ) -> AsyncGenerator[R, None]:
        """Process a batch of items concurrently.
        
        Args:
            batch: Batch of items.
            processor: Processing function.
            sink: Optional sink function.
            
        Yields:
            Processed items.
        """
        # Create tasks for concurrent processing
        tasks = []
        for item in batch:
            if asyncio.iscoroutinefunction(processor):
                task = asyncio.create_task(processor(item))
            else:
                # Run sync function in thread pool
                task = asyncio.create_task(
                    asyncio.to_thread(processor, item)
                )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self._logger.error(f"Processing error: {result}")
                continue
            
            yield result
            
            # Send to sink if provided
            if sink:
                try:
                    if asyncio.iscoroutinefunction(sink):
                        await sink(result)
                    else:
                        sink(result)
                except Exception as e:
                    self._logger.error(f"Sink error: {e}")


class AsyncCache:
    """High-performance async cache with TTL and eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[float] = None,
        eviction_policy: str = "lru",
    ):
        """Initialize async cache.
        
        Args:
            max_size: Maximum cache size.
            ttl: Time to live in seconds.
            eviction_policy: Eviction policy ("lru", "lfu", "fifo").
        """
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None.
        """
        async with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if self.ttl and self._is_expired(key):
                await self._remove(key)
                return None
            
            # Update access tracking
            self._access_times[key] = asyncio.get_event_loop().time()
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            return self._cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            
            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict()
            
            # Set value
            self._cache[key] = value
            self._creation_times[key] = current_time
            self._access_times[key] = current_time
            self._access_counts[key] = 0
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if key was deleted.
        """
        async with self._lock:
            return await self._remove(key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
            self._access_counts.clear()
    
    async def size(self) -> int:
        """Get cache size.
        
        Returns:
            Number of cached items.
        """
        async with self._lock:
            return len(self._cache)
    
    async def keys(self) -> List[str]:
        """Get all cache keys.
        
        Returns:
            List of cache keys.
        """
        async with self._lock:
            return list(self._cache.keys())
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired.
        
        Args:
            key: Cache key.
            
        Returns:
            True if expired.
        """
        if not self.ttl or key not in self._creation_times:
            return False
        
        current_time = asyncio.get_event_loop().time()
        return (current_time - self._creation_times[key]) > self.ttl
    
    async def _remove(self, key: str) -> bool:
        """Remove cache entry.
        
        Args:
            key: Cache key.
            
        Returns:
            True if removed.
        """
        if key in self._cache:
            del self._cache[key]
            self._access_times.pop(key, None)
            self._creation_times.pop(key, None)
            self._access_counts.pop(key, None)
            return True
        return False
    
    async def _evict(self) -> None:
        """Evict cache entry based on policy."""
        if not self._cache:
            return
        
        if self.eviction_policy == "lru":
            # Evict least recently used
            lru_key = min(self._access_times, key=self._access_times.get)
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            lfu_key = min(self._access_counts, key=self._access_counts.get)
        elif self.eviction_policy == "fifo":
            # Evict oldest
            fifo_key = min(self._creation_times, key=self._creation_times.get)
        else:
            # Default to LRU
            lru_key = min(self._access_times, key=self._access_times.get)
        
        evict_key = lru_key if self.eviction_policy == "lru" else (
            lfu_key if self.eviction_policy == "lfu" else fifo_key
        )
        
        await self._remove(evict_key)
        self._logger.debug(f"Evicted cache entry: {evict_key}")


class AsyncFileProcessor:
    """High-performance async file processor with streaming support."""
    
    def __init__(self, config: AsyncConfig):
        """Initialize file processor.
        
        Args:
            config: Async configuration.
        """
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def read_file_chunks(
        self,
        file_path: Path,
        chunk_size: int = 8192,
    ) -> AsyncGenerator[bytes, None]:
        """Read file in chunks asynchronously.
        
        Args:
            file_path: Path to file.
            chunk_size: Chunk size in bytes.
            
        Yields:
            File chunks.
        """
        loop = asyncio.get_event_loop()
        
        def read_chunk():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        # Run in thread pool to avoid blocking
        async for chunk in await loop.run_in_executor(None, lambda: read_chunk()):
            yield chunk
    
    async def write_file_chunks(
        self,
        file_path: Path,
        chunks: AsyncGenerator[bytes, None],
    ) -> None:
        """Write file chunks asynchronously.
        
        Args:
            file_path: Path to output file.
            chunks: Chunk generator.
        """
        loop = asyncio.get_event_loop()
        
        def write_chunks():
            with open(file_path, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
        
        await loop.run_in_executor(None, write_chunks)
    
    async def process_file_stream(
        self,
        input_path: Path,
        output_path: Path,
        processor: Callable[[bytes], bytes],
        chunk_size: int = 8192,
    ) -> None:
        """Process file stream with async chunks.
        
        Args:
            input_path: Input file path.
            output_path: Output file path.
            processor: Chunk processing function.
            chunk_size: Chunk size in bytes.
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create processing pipeline
        async def process_chunks():
            async for chunk in self.read_file_chunks(input_path, chunk_size):
                if asyncio.iscoroutinefunction(processor):
                    processed_chunk = await processor(chunk)
                else:
                    processed_chunk = await asyncio.to_thread(processor, chunk)
                yield processed_chunk
        
        await self.write_file_chunks(output_path, process_chunks())


@asynccontextmanager
async def async_context(config: AsyncConfig) -> AsyncGenerator[AsyncConfig, None]:
    """Async context for configuration management.
    
    Args:
        config: Async configuration.
        
    Yields:
        Configuration object.
    """
    try:
        yield config
    finally:
        # Cleanup if needed
        pass


# Utility functions
async def gather_with_concurrency(
    tasks: List[Task],
    max_concurrency: int = 10,
) -> List[Any]:
    """Gather tasks with concurrency limit.
    
    Args:
        tasks: List of tasks to execute.
        max_concurrency: Maximum concurrent tasks.
        
    Returns:
        List of results.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_task(task: Task) -> Any:
        async with semaphore:
            return await task
    
    bounded_tasks = [bounded_task(task) for task in tasks]
    return await gather(*bounded_tasks, return_exceptions=True)


async def retry_async(
    func: Callable[..., Any],
    *args,
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs,
) -> Any:
    """Retry async function with exponential backoff.
    
    Args:
        func: Function to retry.
        *args: Function arguments.
        attempts: Number of retry attempts.
        delay: Initial delay between retries.
        backoff: Backoff multiplier.
        **kwargs: Function keyword arguments.
        
    Returns:
        Function result.
        
    Raises:
        Exception: If all attempts fail.
    """
    last_exception = None
    
    for attempt in range(attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < attempts - 1:
                await asyncio.sleep(delay * (backoff ** attempt))
            else:
                raise last_exception


async def timeout_async(
    func: Callable[..., Any],
    timeout: float,
    *args,
    **kwargs,
) -> Any:
    """Execute function with timeout.
    
    Args:
        func: Function to execute.
        timeout: Timeout in seconds.
        *args: Function arguments.
        **kwargs: Function keyword arguments.
        
    Returns:
        Function result.
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded.
    """
    if asyncio.iscoroutinefunction(func):
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
    else:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout
        )
