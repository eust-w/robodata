"""Advanced caching utilities for Robodata performance optimization."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from robodata.core.constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_SIZE_BYTES,
    DEFAULT_IMAGE_CACHE_SIZE,
    DEFAULT_IMAGE_CACHE_SIZE_BYTES,
    DEFAULT_MEMORY_LIMIT_MB,
)
from robodata.core.typing_utils import ensure_numpy

T = TypeVar("T")
K = TypeVar("K")


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    max_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def utilization(self) -> float:
        """Calculate cache utilization."""
        return self.size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0


class LRUCache(Generic[K, T]):
    """Thread-safe LRU cache with size limits and statistics."""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, max_size_bytes: int = DEFAULT_CACHE_SIZE_BYTES):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            max_size_bytes: Maximum memory usage in bytes
        """
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self._cache: OrderedDict[K, tuple[T, int]] = OrderedDict()  # key -> (value, size_bytes)
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size_bytes=max_size_bytes)
        self._size_bytes = 0
    
    def get(self, key: K) -> T | None:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                value, size = self._cache.pop(key)
                self._cache[key] = (value, size)  # Move to end (most recently used)
                self._stats.hits += 1
                return value
            else:
                self._stats.misses += 1
                return None
    
    def put(self, key: K, value: T, size_bytes: int = 0) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Estimated memory usage of value
        """
        with self._lock:
            # Remove existing key if present
            if key in self._cache:
                _, old_size = self._cache.pop(key)
                self._size_bytes -= old_size
            
            # Evict items if necessary
            while (len(self._cache) >= self.max_size or 
                   self._size_bytes + size_bytes > self.max_size_bytes) and self._cache:
                evicted_key, (evicted_value, evicted_size) = self._cache.popitem(last=False)
                self._size_bytes -= evicted_size
                self._stats.evictions += 1
            
            # Add new item
            self._cache[key] = (value, size_bytes)
            self._size_bytes += size_bytes
            self._stats.size_bytes = self._size_bytes
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._size_bytes = 0
            self._stats.size_bytes = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._size_bytes,
                max_size_bytes=self.max_size_bytes,
            )


class MemoryManager:
    """Manages memory usage and garbage collection for large datasets."""
    
    def __init__(self, max_memory_mb: int = DEFAULT_MEMORY_LIMIT_MB):  # Default 4GB
        """Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback for memory pressure.
        
        Args:
            callback: Function to call when memory is low
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def check_memory_pressure(self) -> None:
        """Check memory usage and trigger cleanup if needed."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            
            if current_memory > self.max_memory_bytes:
                self._trigger_cleanup()
        except ImportError:
            # Fallback: trigger cleanup periodically
            pass
    
    def _trigger_cleanup(self) -> None:
        """Trigger all registered cleanup callbacks."""
        with self._lock:
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Force garbage collection
        gc.collect()
    
    def force_cleanup(self) -> None:
        """Force immediate cleanup."""
        self._trigger_cleanup()


class ImageCache:
    """Specialized cache for image data with compression support."""
    
    def __init__(self, max_size: int = DEFAULT_IMAGE_CACHE_SIZE, max_size_bytes: int = DEFAULT_IMAGE_CACHE_SIZE_BYTES):
        """Initialize image cache.
        
        Args:
            max_size: Maximum number of images to cache
            max_size_bytes: Maximum memory usage in bytes
        """
        self._cache = LRUCache(max_size, max_size_bytes)
        self._compression_enabled = True
    
    def get_image(self, key: str) -> Any | None:
        """Get image from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Image array or None if not found
        """
        cached_data = self._cache.get(key)
        if cached_data is None:
            return None
        
        if self._compression_enabled and isinstance(cached_data, bytes):
            # Decompress image
            return self._decompress_image(cached_data)
        else:
            return cached_data
    
    def put_image(self, key: str, image: Any) -> None:
        """Put image in cache.
        
        Args:
            key: Cache key
            image: Image array
        """
        if self._compression_enabled:
            # Compress image to save memory
            compressed_data = self._compress_image(image)
            size_bytes = len(compressed_data)
            self._cache.put(key, compressed_data, size_bytes)
        else:
            # Estimate size based on array dimensions
            np = ensure_numpy()
            if hasattr(image, 'nbytes'):
                size_bytes = image.nbytes
            else:
                size_bytes = image.size * image.dtype.itemsize if hasattr(image, 'dtype') else 1024
            self._cache.put(key, image, size_bytes)
    
    def _compress_image(self, image: Any) -> bytes:
        """Compress image using JPEG compression."""
        try:
            from PIL import Image
            import io
            
            # Convert to PIL Image
            if hasattr(image, 'astype'):
                image_array = image.astype(np.uint8)
            else:
                image_array = image
            
            pil_image = Image.fromarray(image_array)
            
            # Compress to JPEG
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
            return buffer.getvalue()
        except Exception:
            # Fallback: return as-is
            return image
    
    def _decompress_image(self, compressed_data: bytes) -> Any:
        """Decompress image from JPEG."""
        try:
            from PIL import Image
            import io
            
            pil_image = Image.open(io.BytesIO(compressed_data))
            np = ensure_numpy()
            return np.array(pil_image)
        except Exception:
            # Fallback: return as-is
            return compressed_data
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def clear(self) -> None:
        """Clear all cached images."""
        self._cache.clear()


# Global cache instances
_image_cache = ImageCache()
_memory_manager = MemoryManager()


def get_image_cache() -> ImageCache:
    """Get the global image cache instance."""
    return _image_cache


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return _memory_manager


def cleanup_caches() -> None:
    """Clean up all global caches."""
    _image_cache.clear()
    _memory_manager.force_cleanup()


# Context manager for automatic cleanup
class CacheContext:
    """Context manager for automatic cache cleanup."""
    
    def __init__(self, cleanup_on_exit: bool = True):
        """Initialize cache context.
        
        Args:
            cleanup_on_exit: Whether to clean up caches on exit
        """
        self.cleanup_on_exit = cleanup_on_exit
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            cleanup_caches()
        return False
