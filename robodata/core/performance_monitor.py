"""Performance monitoring and optimization system for Robodata.

Provides real-time performance metrics, resource monitoring, and
automatic optimization suggestions.
"""

from __future__ import annotations

import psutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from robodata.core.logging_system import get_logger


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    PROCESSING_RATE = "processing_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    metric: PerformanceMetric
    current: float
    average: float
    min: float
    max: float
    samples: int
    trend: str  # "increasing", "decreasing", "stable"


class ResourceMonitor:
    """Monitor system resources in real-time."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._metrics: Dict[PerformanceMetric, deque] = {
            metric: deque(maxlen=1000) for metric in PerformanceMetric
        }
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> None:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric(PerformanceMetric.CPU_USAGE, MetricPoint(timestamp, cpu_percent))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self._add_metric(PerformanceMetric.MEMORY_USAGE, MetricPoint(timestamp, memory_percent))
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_read_mb = disk_io.read_bytes / (1024 * 1024)
            disk_write_mb = disk_io.write_bytes / (1024 * 1024)
            self._add_metric(PerformanceMetric.DISK_IO, MetricPoint(timestamp, disk_read_mb + disk_write_mb))
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            network_sent_mb = network_io.bytes_sent / (1024 * 1024)
            network_recv_mb = network_io.bytes_recv / (1024 * 1024)
            self._add_metric(PerformanceMetric.NETWORK_IO, MetricPoint(timestamp, network_sent_mb + network_recv_mb))
    
    def _add_metric(self, metric: PerformanceMetric, point: MetricPoint) -> None:
        """Add a metric point."""
        with self._lock:
            self._metrics[metric].append(point)
    
    def get_stats(self, metric: PerformanceMetric, window_size: int = 100) -> Optional[PerformanceStats]:
        """Get statistics for a metric."""
        with self._lock:
            points = list(self._metrics[metric])[-window_size:]
        
        if not points:
            return None
        
        values = [p.value for p in points]
        current = values[-1]
        average = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        # Calculate trend
        trend = self._calculate_trend(values)
        
        return PerformanceStats(
            metric=metric,
            current=current,
            average=average,
            min=min_val,
            max=max_val,
            samples=len(values),
            trend=trend
        )
    
    def _calculate_trend(self, values: List[float], threshold: float = 0.1) -> str:
        """Calculate trend direction."""
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression to determine trend
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if abs(slope) < threshold:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def get_all_stats(self) -> Dict[PerformanceMetric, PerformanceStats]:
        """Get statistics for all metrics."""
        stats = {}
        for metric in PerformanceMetric:
            metric_stats = self.get_stats(metric)
            if metric_stats:
                stats[metric] = metric_stats
        return stats


class PerformanceProfiler:
    """Profile performance of specific operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._profiles: Dict[str, List[float]] = {}
        self._current_operations: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        self._current_operations[operation_name] = time.time()
    
    def end_operation(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if operation_name not in self._current_operations:
            self.logger.warning(f"Operation {operation_name} was not started")
            return 0.0
        
        start_time = self._current_operations.pop(operation_name)
        duration = time.time() - start_time
        
        if operation_name not in self._profiles:
            self._profiles[operation_name] = []
        
        self._profiles[operation_name].append(duration)
        return duration
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        if operation_name not in self._profiles or not self._profiles[operation_name]:
            return None
        
        durations = self._profiles[operation_name]
        return {
            "count": len(durations),
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            op: stats for op, stats in 
            ((op, self.get_operation_stats(op)) for op in self._profiles)
            if stats is not None
        }
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self._profiles.clear()
        self._current_operations.clear()


class PerformanceOptimizer:
    """Provide performance optimization suggestions."""
    
    def __init__(self, resource_monitor: ResourceMonitor, profiler: PerformanceProfiler):
        self.resource_monitor = resource_monitor
        self.profiler = profiler
        self.logger = get_logger(__name__)
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on current performance."""
        suggestions = []
        
        # Check memory usage
        memory_stats = self.resource_monitor.get_stats(PerformanceMetric.MEMORY_USAGE)
        if memory_stats and memory_stats.current > 80:
            suggestions.append({
                "type": "memory",
                "severity": "high",
                "message": "High memory usage detected",
                "suggestions": [
                    "Reduce batch size",
                    "Enable disk caching",
                    "Close other applications",
                    "Use streaming mode"
                ]
            })
        
        # Check CPU usage
        cpu_stats = self.resource_monitor.get_stats(PerformanceMetric.CPU_USAGE)
        if cpu_stats and cpu_stats.current > 90:
            suggestions.append({
                "type": "cpu",
                "severity": "medium",
                "message": "High CPU usage detected",
                "suggestions": [
                    "Reduce number of workers",
                    "Use parallel processing efficiently",
                    "Check for CPU-intensive operations"
                ]
            })
        
        # Check operation performance
        operation_stats = self.profiler.get_all_stats()
        for op_name, stats in operation_stats.items():
            if stats["average"] > 10.0:  # Slow operations
                suggestions.append({
                    "type": "operation",
                    "severity": "medium",
                    "message": f"Slow operation detected: {op_name}",
                    "suggestions": [
                        "Optimize algorithm",
                        "Use caching",
                        "Consider parallel processing",
                        "Profile for bottlenecks"
                    ],
                    "operation": op_name,
                    "stats": stats
                })
        
        return suggestions
    
    def auto_optimize(self) -> Dict[str, Any]:
        """Attempt automatic optimizations."""
        optimizations = {}
        
        # Auto-adjust worker count based on CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count:
            memory_stats = self.resource_monitor.get_stats(PerformanceMetric.MEMORY_USAGE)
            if memory_stats and memory_stats.current < 50:
                # Low memory usage, can use more workers
                optimal_workers = min(cpu_count, 8)
                optimizations["num_workers"] = optimal_workers
            else:
                # High memory usage, reduce workers
                optimal_workers = max(1, cpu_count // 2)
                optimizations["num_workers"] = optimal_workers
        
        # Auto-adjust batch size based on memory
        memory_stats = self.resource_monitor.get_stats(PerformanceMetric.MEMORY_USAGE)
        if memory_stats:
            if memory_stats.current > 70:
                optimizations["batch_size"] = 50  # Small batch size
            elif memory_stats.current > 40:
                optimizations["batch_size"] = 200  # Medium batch size
            else:
                optimizations["batch_size"] = 1000  # Large batch size
        
        return optimizations


class PerformanceManager:
    """Central performance management system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.resource_monitor = ResourceMonitor()
        self.profiler = PerformanceProfiler()
        self.optimizer = PerformanceOptimizer(self.resource_monitor, self.profiler)
        self._monitoring = False
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.resource_monitor.start_monitoring()
        self._monitoring = True
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.resource_monitor.stop_monitoring()
        self._monitoring = False
        self.logger.info("Performance monitoring stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        resource_stats = self.resource_monitor.get_all_stats()
        operation_stats = self.profiler.get_all_stats()
        suggestions = self.optimizer.get_optimization_suggestions()
        optimizations = self.optimizer.auto_optimize()
        
        return {
            "timestamp": time.time(),
            "resources": {
                metric.value: {
                    "current": stats.current,
                    "average": stats.average,
                    "min": stats.min,
                    "max": stats.max,
                    "trend": stats.trend,
                }
                for metric, stats in resource_stats.items()
            },
            "operations": operation_stats,
            "suggestions": suggestions,
            "auto_optimizations": optimizations,
        }
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return OperationProfiler(self, operation_name)


class OperationProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, manager: PerformanceManager, operation_name: str):
        self.manager = manager
        self.operation_name = operation_name
    
    def __enter__(self):
        self.manager.profiler.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.manager.profiler.end_operation(self.operation_name)
        
        # Log slow operations
        if duration > 5.0:
            self.manager.logger.warning(
                f"Slow operation: {self.operation_name} took {duration:.2f}s"
            )


# Global performance manager
_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> PerformanceManager:
    """Get the global performance manager."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager()
    return _performance_manager


def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    get_performance_manager().start_monitoring()


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    get_performance_manager().stop_monitoring()


def profile_operation(operation_name: str):
    """Context manager for profiling operations."""
    return get_performance_manager().profile_operation(operation_name)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    return get_performance_manager().get_performance_report()


# Performance decorator
def performance_monitor(operation_name: Optional[str] = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            manager = get_performance_manager()
            with manager.profile_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
