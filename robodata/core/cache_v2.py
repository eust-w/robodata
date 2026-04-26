"""Optimized caching system for Robodata.

Provides efficient, memory-safe caching with automatic eviction,
performance monitoring, and multiple cache strategies.
"""

from __future__ import annotations

import hashlib
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar, Union

T = TypeVar("T")


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    max_size: int = 1000
    strategy: CacheStrategy = CacheStrategy.LRU
    ttl_seconds: int | None = None  # Only used with TTL strategy
    enable_metrics: bool = True
    enable_disk_cache: bool = False
    disk_cache_dir: Path | None = None
    max_disk_size_mb: int = 1024


@dataclass
class CacheMetrics:
    """Performance metrics for cache operations."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.hit_rate = 0.0


class CacheEntry(Generic[T]):
    """Single cache entry with metadata."""
    
    def __init__(self, value: T, key: str):
        self.value = value
        self.key = key
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired based on TTL."""
        return time.time() - self.created_at > ttl_seconds


class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get value by key."""
        ...
    
    @abstractmethod
    def put(self, key: str, value: T) -> None:
        """Put value with key."""
        ...
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove entry by key."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        ...
    
    @abstractmethod
    def size(self) -> int:
        """Get current size."""
        ...


class MemoryCache(CacheBackend[T]):
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, CacheEntry[T]] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_order = {}  # For LFU
        self._lock = threading.RLock()
        self.metrics = CacheMetrics()
    
    def get(self, key: str) -> T | None:
        """Get value by key."""
        with self._lock:
            self.metrics.total_requests += 1
            
            if key not in self._cache:
                self.metrics.misses += 1
                self.metrics.update_hit_rate()
                return None
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if (self.config.strategy == CacheStrategy.TTL and 
                self.config.ttl_seconds and 
                entry.is_expired(self.config.ttl_seconds)):
                self._remove_entry(key)
                self.metrics.misses += 1
                self.metrics.update_hit_rate()
                return None
            
            # Update access information
            entry.touch()
            self._update_access_order(key)
            self._update_frequency_order(key)
            
            self.metrics.hits += 1
            self.metrics.update_hit_rate()
            return entry.value
    
    def put(self, key: str, value: T) -> None:
        """Put value with key."""
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.config.max_size and key not in self._cache:
                self._evict()
            
            # Add or update entry
            entry = CacheEntry(value, key)
            self._cache[key] = entry
            self._update_access_order(key)
            self._update_frequency_order(key)
    
    def remove(self, key: str) -> bool:
        """Remove entry by key."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_order.clear()
            self.metrics.reset()
    
    def size(self) -> int:
        """Get current size."""
        with self._lock:
            return len(self._cache)
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update tracking structures."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.pop(key, None)
            self._frequency_order.pop(key, None)
            return True
        return False
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if self.config.strategy == CacheStrategy.LRU:
            self._access_order.move_to_end(key)
    
    def _update_frequency_order(self, key: str) -> None:
        """Update frequency order for LFU."""
        if self.config.strategy == CacheStrategy.LFU:
            self._frequency_order[key] = self._cache[key].access_count
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self._cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = next(iter(self._access_order))
            self._remove_entry(oldest_key)
            
        elif self.config.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            if self._frequency_order:
                lfu_key = min(self._frequency_order, key=self._frequency_order.get)
                self._remove_entry(lfu_key)
                
        elif self.config.strategy == CacheStrategy.FIFO:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].created_at)
            self._remove_entry(oldest_key)
            
        elif self.config.strategy == CacheStrategy.TTL:
            # Evict expired entries
            if self.config.ttl_seconds:
                expired_keys = [
                    key for key, entry in self._cache.items()
                    if entry.is_expired(self.config.ttl_seconds)
                ]
                for key in expired_keys:
                    self._remove_entry(key)
                
                # If no expired entries, evict oldest
                if len(self._cache) >= self.config.max_size:
                    oldest_key = min(self._cache.keys(), 
                                   key=lambda k: self._cache[k].created_at)
                    self._remove_entry(oldest_key)
        
        self.metrics.evictions += 1


class DiskCache(CacheBackend[T]):
    """Disk-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = config.disk_cache_dir or Path.home() / ".robodata" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.metrics = CacheMetrics()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> T | None:
        """Get value by key."""
        with self._lock:
            self.metrics.total_requests += 1
            
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                self.metrics.misses += 1
                self.metrics.update_hit_rate()
                return None
            
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check TTL
                if (self.config.strategy == CacheStrategy.TTL and 
                    self.config.ttl_seconds and 
                    entry.is_expired(self.config.ttl_seconds)):
                    cache_path.unlink()
                    self.metrics.misses += 1
                    self.metrics.update_hit_rate()
                    return None
                
                entry.touch()
                # Update access time on disk
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                self.metrics.hits += 1
                self.metrics.update_hit_rate()
                return entry.value
                
            except Exception:
                self.metrics.misses += 1
                self.metrics.update_hit_rate()
                return None
    
    def put(self, key: str, value: T) -> None:
        """Put value with key."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            entry = CacheEntry(value, key)
            
            try:
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception:
                pass  # Silently fail disk writes
    
    def remove(self, key: str) -> bool:
        """Remove entry by key."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            self.metrics.reset()
    
    def size(self) -> int:
        """Get current size."""
        with self._lock:
            return len(list(self.cache_dir.glob("*.cache")))


class HybridCache(CacheBackend[T]):
    """Hybrid cache that uses memory and disk."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config)
        self.disk_cache = DiskCache(config) if config.enable_disk_cache else None
        self.metrics = CacheMetrics()
    
    def get(self, key: str) -> T | None:
        """Get value by key, checking memory first then disk."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.metrics.hits += 1
            self.metrics.total_requests += 1
            return value
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                self.metrics.hits += 1
                self.metrics.total_requests += 1
                return value
        
        self.metrics.misses += 1
        self.metrics.total_requests += 1
        return None
    
    def put(self, key: str, value: T) -> None:
        """Put value in both memory and disk caches."""
        self.memory_cache.put(key, value)
        if self.disk_cache:
            self.disk_cache.put(key, value)
    
    def remove(self, key: str) -> bool:
        """Remove from both caches."""
        memory_removed = self.memory_cache.remove(key)
        disk_removed = self.disk_cache.remove(key) if self.disk_cache else False
        return memory_removed or disk_removed
    
    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
        self.metrics.reset()
    
    def size(self) -> int:
        """Get total size (memory cache size)."""
        return self.memory_cache.size()


class CacheFactory:
    """Factory for creating cache instances."""
    
    @staticmethod
    def create_cache(config: CacheConfig) -> CacheBackend:
        """Create cache instance based on configuration."""
        if config.enable_disk_cache:
            return HybridCache(config)
        else:
            return MemoryCache(config)


# Global cache manager
class CacheManager:
    """Global cache manager for Robodata."""
    
    _instance: CacheManager | None = None
    _caches: dict[str, CacheBackend] = {}
    
    def __new__(cls) -> CacheManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_cache(self, name: str, config: CacheConfig) -> CacheBackend:
        """Register a named cache."""
        cache = CacheFactory.create_cache(config)
        self._caches[name] = cache
        return cache
    
    def get_cache(self, name: str) -> CacheBackend | None:
        """Get a registered cache."""
        return self._caches.get(name)
    
    def clear_all(self) -> None:
        """Clear all registered caches."""
        for cache in self._caches.values():
            cache.clear()
    
    def get_metrics(self) -> dict[str, CacheMetrics]:
        """Get metrics for all caches."""
        return {
            name: getattr(cache, 'metrics', CacheMetrics())
            for name, cache in self._caches.items()
        }


# Convenience functions
def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return CacheManager()


def create_cache(
    name: str,
    max_size: int = 1000,
    strategy: CacheStrategy = CacheStrategy.LRU,
    enable_disk_cache: bool = False,
    **kwargs
) -> CacheBackend:
    """Create and register a cache."""
    config = CacheConfig(
        max_size=max_size,
        strategy=strategy,
        enable_disk_cache=enable_disk_cache,
        **kwargs
    )
    manager = get_cache_manager()
    return manager.register_cache(name, config)


# Cache decorator
def cached(
    cache_name: str = "default",
    ttl_seconds: int | None = None,
    key_func: callable | None = None
):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_cache_manager()
            cache = manager.get_cache(cache_name)
            
            if cache is None:
                # Create default cache if it doesn't exist
                config = CacheConfig()
                if ttl_seconds:
                    config.strategy = CacheStrategy.TTL
                    config.ttl_seconds = ttl_seconds
                cache = manager.register_cache(cache_name, config)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        return wrapper
    return decorator
