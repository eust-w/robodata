"""Intelligent caching system with adaptive strategies and machine learning."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from robodata.core.async_utils import AsyncCache
from robodata.core.exceptions import RobodataError

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")


class SmartCacheError(RobodataError):
    """Smart cache specific error."""
    pass


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class CacheLevel(Enum):
    """Cache levels."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if entry is expired.
        
        Args:
            current_time: Current timestamp.
            
        Returns:
            True if expired.
        """
        if self.ttl is None:
            return False
        return (current_time - self.created_at) > self.ttl
    
    def update_access(self, current_time: float) -> None:
        """Update access information.
        
        Args:
            current_time: Current timestamp.
        """
        self.last_accessed = current_time
        self.access_count += 1
    
    def get_score(self, strategy: CacheStrategy, current_time: float) -> float:
        """Get eviction score based on strategy.
        
        Args:
            strategy: Eviction strategy.
            current_time: Current timestamp.
            
        Returns:
            Eviction score (lower = more likely to evict).
        """
        if strategy == CacheStrategy.LRU:
            # Least recently used (older is better to evict)
            return current_time - self.last_accessed
        
        elif strategy == CacheStrategy.LFU:
            # Least frequently used (lower count is better to evict)
            return -self.access_count
        
        elif strategy == CacheStrategy.FIFO:
            # First in, first out (older is better to evict)
            return current_time - self.created_at
        
        elif strategy == CacheStrategy.RANDOM:
            # Random (uniform)
            return hash(self.key) % 1000
        
        elif strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: combination of recency, frequency, and size
            age = current_time - self.last_accessed
            frequency = self.access_count
            size_factor = self.size_bytes / 1024  # Size in KB
            
            # Lower score = more likely to evict
            return (age * 0.4) - (frequency * 0.4) + (size_factor * 0.2)
        
        return 0.0


class CacheStorage(ABC):
    """Abstract cache storage backend."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry.
        
        Args:
            key: Cache key.
            
        Returns:
            Cache entry or None.
        """
        pass
    
    @abstractmethod
    async def set(self, entry: CacheEntry) -> None:
        """Set cache entry.
        
        Args:
            entry: Cache entry.
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry.
        
        Args:
            key: Cache key.
            
        Returns:
            True if deleted.
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys.
        
        Returns:
            List of keys.
        """
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size.
        
        Returns:
            Number of entries.
        """
        pass


class MemoryStorage(CacheStorage):
    """In-memory cache storage."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize memory storage.
        
        Args:
            max_size: Maximum number of entries.
        """
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                entry.update_access(time.time())
            return entry
    
    async def set(self, entry: CacheEntry) -> None:
        """Set cache entry."""
        async with self._lock:
            # Evict if necessary
            if len(self._cache) >= self.max_size and entry.key not in self._cache:
                await self._evict_lru()
            
            self._cache[entry.key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        async with self._lock:
            return list(self._cache.keys())
    
    async def size(self) -> int:
        """Get cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]
        self._logger.debug(f"Evicted LRU entry: {lru_key}")


class DiskStorage(CacheStorage):
    """Disk-based cache storage."""
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1024,
        compression: bool = True,
    ):
        """Initialize disk storage.
        
        Args:
            cache_dir: Cache directory.
            max_size_mb: Maximum size in MB.
            compression: Whether to compress entries.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.compression = compression
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for key.
        
        Args:
            key: Cache key.
            
        Returns:
            File path.
        """
        # Use hash of key as filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            async with self._lock:
                with open(file_path, "rb") as f:
                    data = f.read()
                
                if self.compression:
                    import gzip
                    data = gzip.decompress(data)
                
                entry = pickle.loads(data)
                entry.update_access(time.time())
                return entry
                
        except Exception as e:
            self._logger.error(f"Error reading cache entry {key}: {e}")
            # Remove corrupted file
            try:
                file_path.unlink()
            except:
                pass
            return None
    
    async def set(self, entry: CacheEntry) -> None:
        """Set cache entry."""
        file_path = self._get_file_path(entry.key)
        
        # Check size limit
        await self._enforce_size_limit()
        
        try:
            data = pickle.dumps(entry)
            
            if self.compression:
                import gzip
                data = gzip.compress(data)
            
            async with self._lock:
                with open(file_path, "wb") as f:
                    f.write(data)
            
            self._logger.debug(f"Wrote cache entry to disk: {entry.key}")
            
        except Exception as e:
            self._logger.error(f"Error writing cache entry {entry.key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception as e:
            self._logger.error(f"Error deleting cache entry {key}: {e}")
        
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            self._logger.info("Cleared disk cache")
        except Exception as e:
            self._logger.error(f"Error clearing disk cache: {e}")
    
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        keys = []
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                # This would require reading each file to get the key
                # For simplicity, return empty list
                pass
        except Exception as e:
            self._logger.error(f"Error listing cache keys: {e}")
        
        return keys
    
    async def size(self) -> int:
        """Get cache size."""
        try:
            return len(list(self.cache_dir.glob("*.cache")))
        except Exception as e:
            self._logger.error(f"Error getting cache size: {e}")
            return 0
    
    async def _enforce_size_limit(self) -> None:
        """Enforce size limit by evicting old entries."""
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.cache")
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Evict oldest files until under limit
        files = sorted(
            self.cache_dir.glob("*.cache"),
            key=lambda f: f.stat().st_mtime
        )
        
        for file_path in files:
            try:
                file_path.unlink()
                total_size -= file_path.stat().st_size
                
                if total_size <= max_size_bytes:
                    break
                    
            except Exception as e:
                self._logger.error(f"Error evicting file {file_path}: {e}")
        
        self._logger.debug(f"Evicted old files to enforce size limit")


class SmartCache:
    """Intelligent cache with adaptive strategies and multi-level storage."""
    
    def __init__(
        self,
        name: str,
        memory_size: int = 1000,
        disk_size_mb: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl: Optional[float] = None,
        enable_stats: bool = True,
    ):
        """Initialize smart cache.
        
        Args:
            name: Cache name.
            memory_size: Memory cache size.
            disk_size_mb: Disk cache size in MB.
            strategy: Eviction strategy.
            ttl: Default TTL in seconds.
            enable_stats: Enable statistics collection.
        """
        self.name = name
        self.strategy = strategy
        self.default_ttl = ttl
        self.enable_stats = enable_stats
        
        # Create storage layers
        self.memory_storage = MemoryStorage(memory_size)
        self.disk_storage = DiskStorage(Path(f"/tmp/robocache_{name}"), disk_size_mb)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "sets": 0,
            "evictions": 0,
            "total_requests": 0,
        }
        
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None.
        """
        if self.enable_stats:
            self.stats["total_requests"] += 1
        
        current_time = time.time()
        
        # Try memory first
        entry = await self.memory_storage.get(key)
        if entry and not entry.is_expired(current_time):
            if self.enable_stats:
                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1
            return entry.value
        
        # Try disk
        entry = await self.disk_storage.get(key)
        if entry and not entry.is_expired(current_time):
            # Promote to memory
            await self.memory_storage.set(entry)
            
            if self.enable_stats:
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
            return entry.value
        
        if self.enable_stats:
            self.stats["misses"] += 1
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        priority: float = 1.0,
        tags: List[str] = None,
    ) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.
            priority: Priority for eviction.
            tags: Cache tags.
        """
        current_time = time.time()
        
        # Calculate entry size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=current_time,
            last_accessed=current_time,
            size_bytes=size_bytes,
            ttl=ttl or self.default_ttl,
            priority=priority,
            tags=tags or [],
        )
        
        # Store in memory first
        await self.memory_storage.set(entry)
        
        # Also store on disk for persistence
        await self.disk_storage.set(entry)
        
        if self.enable_stats:
            self.stats["sets"] += 1
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if deleted.
        """
        memory_deleted = await self.memory_storage.delete(key)
        disk_deleted = await self.disk_storage.delete(key)
        
        return memory_deleted or disk_deleted
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        await self.memory_storage.clear()
        await self.disk_storage.clear()
        
        if self.enable_stats:
            self.stats.update({
                "hits": 0,
                "misses": 0,
                "memory_hits": 0,
                "disk_hits": 0,
                "sets": 0,
                "evictions": 0,
                "total_requests": 0,
            })
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics.
        """
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]
        
        memory_hit_rate = 0.0
        if self.stats["hits"] > 0:
            memory_hit_rate = self.stats["memory_hits"] / self.stats["hits"]
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_hit_rate": memory_hit_rate,
            "memory_size": await self.memory_storage.size(),
            "disk_size": await self.disk_storage.size(),
            "strategy": self.strategy.value,
        }
    
    async def optimize_strategy(self) -> None:
        """Optimize eviction strategy based on usage patterns."""
        if not self.enable_stats:
            return
        
        stats = await self.get_stats()
        
        # Simple optimization logic
        if stats["memory_hit_rate"] > 0.8:
            # High memory hit rate, keep current strategy
            return
        
        if stats["disk_hits"] > stats["memory_hits"]:
            # More disk hits, increase memory size or change strategy
            self._logger.info("High disk hit rate detected, consider increasing memory cache size")
        
        # Could implement more sophisticated ML-based optimization here
        self._logger.info(f"Current cache performance: {stats['hit_rate']:.2%} hit rate")


class CacheManager:
    """Manager for multiple smart caches."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches: Dict[str, SmartCache] = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def create_cache(
        self,
        name: str,
        memory_size: int = 1000,
        disk_size_mb: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl: Optional[float] = None,
    ) -> SmartCache:
        """Create a new cache.
        
        Args:
            name: Cache name.
            memory_size: Memory cache size.
            disk_size_mb: Disk cache size in MB.
            strategy: Eviction strategy.
            ttl: Default TTL.
            
        Returns:
            Created cache.
        """
        if name in self.caches:
            raise SmartCacheError(f"Cache '{name}' already exists")
        
        cache = SmartCache(
            name=name,
            memory_size=memory_size,
            disk_size_mb=disk_size_mb,
            strategy=strategy,
            ttl=ttl,
        )
        
        self.caches[name] = cache
        self._logger.info(f"Created cache: {name}")
        
        return cache
    
    def get_cache(self, name: str) -> Optional[SmartCache]:
        """Get existing cache.
        
        Args:
            name: Cache name.
            
        Returns:
            Cache or None.
        """
        return self.caches.get(name)
    
    async def delete_cache(self, name: str) -> bool:
        """Delete a cache.
        
        Args:
            name: Cache name.
            
        Returns:
            True if deleted.
        """
        if name in self.caches:
            await self.caches[name].clear()
            del self.caches[name]
            self._logger.info(f"Deleted cache: {name}")
            return True
        return False
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches.
        
        Returns:
            Cache statistics dictionary.
        """
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = await cache.get_stats()
        return stats
    
    async def optimize_all_caches(self) -> None:
        """Optimize all caches."""
        for cache in self.caches.values():
            await cache.optimize_strategy()


# Global cache manager
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager.
    
    Returns:
        Global cache manager.
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


# Decorators
def cache_result(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator to cache function results.
    
    Args:
        cache_name: Cache name.
        ttl: Time to live.
        key_func: Function to generate cache key.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_cache_manager()
            cache = manager.get_cache(cache_name)
            
            if cache is None:
                # Create default cache
                cache = manager.create_cache(cache_name)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Utility functions
async def benchmark_cache(
    cache: SmartCache,
    num_operations: int = 10000,
    value_size: int = 1024,
) -> Dict[str, Any]:
    """Benchmark cache performance.
    
    Args:
        cache: Cache to benchmark.
        num_operations: Number of operations.
        value_size: Size of cached values.
        
    Returns:
        Benchmark results.
    """
    import time
    import os
    
    # Generate test data
    test_value = os.urandom(value_size)
    test_keys = [f"test_key_{i}" for i in range(num_operations)]
    
    # Benchmark writes
    start_time = time.time()
    for key in test_keys:
        await cache.set(key, test_value)
    write_time = time.time() - start_time
    
    # Benchmark reads
    start_time = time.time()
    for key in test_keys:
        await cache.get(key)
    read_time = time.time() - start_time
    
    # Get final stats
    stats = await cache.get_stats()
    
    return {
        "num_operations": num_operations,
        "value_size": value_size,
        "write_time": write_time,
        "read_time": read_time,
        "writes_per_second": num_operations / write_time,
        "reads_per_second": num_operations / read_time,
        "final_stats": stats,
    }
