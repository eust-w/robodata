"""Parallel processing utilities for Robodata performance optimization."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, List, TypeVar

from robodata.core.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    MAX_WORKERS_CPU,
    MAX_WORKERS_IO,
)

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    num_workers: int | None = None
    use_processes: bool = True
    chunk_size: int = DEFAULT_CHUNK_SIZE
    timeout: float | None = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.num_workers is None:
            # Auto-detect optimal worker count
            if self.use_processes:
                self.num_workers = min(mp.cpu_count(), MAX_WORKERS_CPU)
            else:
                self.num_workers = min(mp.cpu_count() * 2, MAX_WORKERS_IO)  # More threads for I/O bound tasks


class WorkerPool:
    """Enhanced worker pool with better error handling and resource management."""
    
    def __init__(self, config: ParallelConfig):
        """Initialize worker pool.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self._executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        self._shutdown = False
        
    def __enter__(self):
        """Context manager entry."""
        if self.config.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.shutdown()
        return False
    
    def shutdown(self) -> None:
        """Shutdown the executor properly."""
        if self._executor is not None and not self._shutdown:
            self._executor.shutdown(wait=True)
            self._shutdown = True
    
    def map(self, func: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Map function over items with parallel processing.
        
        Args:
            func: Function to apply to each item
            items: Items to process
            
        Returns:
            List of results in the same order as input
        """
        if self._executor is None:
            raise RuntimeError("WorkerPool not initialized. Use as context manager.")
        
        items_list = list(items)
        if not items_list:
            return []
        
        futures = []
        for item in items_list:
            future = self._executor.submit(func, item)
            futures.append(future)
        
        results = []
        errors = []
        
        for future, item in zip(futures, items_list):
            try:
                if self.config.timeout is not None:
                    result = future.result(timeout=self.config.timeout)
                else:
                    result = future.result()
                results.append(result)
            except Exception as e:
                error_msg = f"Error processing item: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                if self.config.max_retries > 0:
                    # Retry logic could be implemented here
                    pass
        
        if errors and self.config.max_retries == 0:
            logger.warning(f"Completed with {len(errors)} errors")
        
        return results
    
    def map_unordered(self, func: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Map function over items, returning results as they complete.
        
        Args:
            func: Function to apply to each item
            items: Items to process
            
        Returns:
            List of results in completion order (not input order)
        """
        if self._executor is None:
            raise RuntimeError("WorkerPool not initialized. Use as context manager.")
        
        items_list = list(items)
        if not items_list:
            return []
        
        futures = []
        for item in items_list:
            future = self._executor.submit(func, item)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                if self.config.timeout is not None:
                    result = future.result(timeout=self.config.timeout)
                else:
                    result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
        
        return results


class BatchProcessor:
    """Process items in batches for better memory efficiency."""
    
    def __init__(self, batch_size: int = 32):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
        """
        self.batch_size = batch_size
    
    def process_batches(
        self, 
        items: Iterable[T], 
        batch_func: Callable[[List[T]], R],
        progress_callback: Callable[[int, int], None] | None = None
    ) -> List[R]:
        """Process items in batches.
        
        Args:
            items: Items to process
            batch_func: Function to process a batch of items
            progress_callback: Optional progress callback
            
        Returns:
            List of batch results
        """
        items_list = list(items)
        total_items = len(items_list)
        results = []
        
        for i in range(0, total_items, self.batch_size):
            batch = items_list[i:i + self.batch_size]
            batch_result = batch_func(batch)
            results.append(batch_result)
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, total_items), total_items)
        
        return results


class ProgressTracker:
    """Thread-safe progress tracking for parallel operations."""
    
    def __init__(self, total: int):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
        """
        self.total = total
        self.completed = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """Update progress.
        
        Args:
            increment: Number of items completed
        """
        with self._lock:
            self.completed += increment
    
    def get_progress(self) -> tuple[float, float]:
        """Get current progress.
        
        Returns:
            Tuple of (completed_ratio, elapsed_time)
        """
        with self._lock:
            ratio = self.completed / self.total if self.total > 0 else 0.0
            elapsed = time.time() - self._start_time
            return ratio, elapsed
    
    def estimate_remaining(self) -> float:
        """Estimate remaining time in seconds.
        
        Returns:
            Estimated remaining time
        """
        ratio, elapsed = self.get_progress()
        if ratio > 0:
            return elapsed * (1 - ratio) / ratio
        return 0.0


def optimize_worker_count(
    task_type: str = "cpu",
    memory_limit_mb: int = 1024,
    io_bound: bool = False
) -> int:
    """Optimize worker count based on system resources and task type.
    
    Args:
        task_type: Type of task ("cpu", "io", "mixed")
        memory_limit_mb: Memory limit per worker in MB
        io_bound: Whether task is I/O bound
        
    Returns:
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()
    
    # Get available memory
    try:
        import psutil
        available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
    except ImportError:
        available_memory_mb = 8192  # Assume 8GB if psutil not available
    
    # Calculate memory-based limit
    memory_based_limit = max(1, available_memory_mb // memory_limit_mb)
    
    if task_type == "cpu" or not io_bound:
        # CPU-bound tasks: use CPU count, limited by memory
        return min(cpu_count, memory_based_limit, MAX_WORKERS_CPU)
    else:
        # I/O-bound tasks: can use more workers
        return min(cpu_count * 2, memory_based_limit, MAX_WORKERS_IO)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    num_workers: int | None = None,
    use_processes: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_callback: Callable[[int, int], None] | None = None
) -> List[R]:
    """Convenient parallel map function.
    
    Args:
        func: Function to apply to each item
        items: Items to process
        num_workers: Number of worker processes/threads
        use_processes: Whether to use processes (True) or threads (False)
        chunk_size: Size of work chunks
        progress_callback: Optional progress callback
        
    Returns:
        List of results
    """
    config = ParallelConfig(
        num_workers=num_workers,
        use_processes=use_processes,
        chunk_size=chunk_size
    )
    
    with WorkerPool(config) as pool:
        if progress_callback:
            # Track progress manually
            items_list = list(items)
            total = len(items_list)
            tracker = ProgressTracker(total)
            
            def track_progress(item):
                result = func(item)
                tracker.update()
                progress_callback(tracker.completed, total)
                return result
            
            return pool.map(track_progress, items_list)
        else:
            return pool.map(func, items)


# Environment-aware worker optimization
def get_optimal_config() -> ParallelConfig:
    """Get optimal parallel configuration based on environment.
    
    Returns:
        Optimized ParallelConfig
    """
    # Check if running in container/CI
    in_container = os.getenv("CI") == "true" or os.path.exists("/.dockerenv")
    
    if in_container:
        # Conservative settings for CI/containers
        return ParallelConfig(
            num_workers=min(mp.cpu_count(), 4),
            use_processes=True,
            chunk_size=2
        )
    else:
        # Full performance for local development
        return ParallelConfig(
            num_workers=optimize_worker_count(),
            use_processes=True,
            chunk_size=DEFAULT_CHUNK_SIZE
        )
