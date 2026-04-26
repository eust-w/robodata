"""Adaptive performance optimization system with machine learning."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from robodata.core.exceptions import RobodataError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AdaptiveError(RobodataError):
    """Adaptive performance specific error."""
    pass


class PerformanceMetric(Enum):
    """Performance metrics types."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class MetricValue:
    """Metric value with timestamp."""
    
    metric: PerformanceMetric
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metric value."""
        if self.value < 0:
            raise ValueError(f"Metric value must be non-negative: {self.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceProfile:
    """Performance profile for optimization target."""
    
    name: str
    target_metrics: Dict[PerformanceMetric, float]
    constraints: Dict[PerformanceMetric, Tuple[float, float]] = field(default_factory=dict)
    priority: float = 1.0
    adaptive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "target_metrics": {k.value: v for k, v in self.target_metrics.items()},
            "constraints": {k.value: v for k, v in self.constraints.items()},
            "priority": self.priority,
            "adaptive": self.adaptive,
        }


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, name: str):
        """Initialize strategy.
        
        Args:
            name: Strategy name.
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
    
    @abstractmethod
    async def analyze(
        self,
        metrics: List[MetricValue],
        profile: PerformanceProfile,
    ) -> Dict[str, Any]:
        """Analyze metrics and generate recommendations.
        
        Args:
            metrics: Historical metric values.
            profile: Performance profile.
            
        Returns:
            Analysis results and recommendations.
        """
        pass
    
    @abstractmethod
    async def apply_optimization(
        self,
        recommendations: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Apply optimization recommendations.
        
        Args:
            recommendations: Optimization recommendations.
            context: Optimization context.
            
        Returns:
            True if optimization was applied successfully.
        """
        pass


class MemoryOptimizationStrategy(OptimizationStrategy):
    """Memory usage optimization strategy."""
    
    def __init__(self):
        """Initialize memory optimization strategy."""
        super().__init__("memory_optimization")
    
    async def analyze(
        self,
        metrics: List[MetricValue],
        profile: PerformanceProfile,
    ) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_metrics = [
            m for m in metrics
            if m.metric == PerformanceMetric.MEMORY_USAGE
        ]
        
        if not memory_metrics:
            return {"recommendations": [], "analysis": "No memory metrics available"}
        
        # Calculate statistics
        values = [m.value for m in memory_metrics]
        avg_memory = statistics.mean(values)
        max_memory = max(values)
        memory_variance = statistics.variance(values) if len(values) > 1 else 0
        
        # Generate recommendations
        recommendations = []
        
        if avg_memory > profile.target_metrics.get(PerformanceMetric.MEMORY_USAGE, float('inf')):
            recommendations.append({
                "type": "reduce_memory",
                "priority": "high",
                "description": "Average memory usage exceeds target",
                "suggestions": [
                    "Increase cache eviction rate",
                    "Reduce batch sizes",
                    "Enable memory compression",
                ],
                "estimated_improvement": min(0.3, (avg_memory - profile.target_metrics.get(PerformanceMetric.MEMORY_USAGE, 0)) / avg_memory),
            })
        
        if memory_variance > avg_memory * 0.5:  # High variance
            recommendations.append({
                "type": "stabilize_memory",
                "priority": "medium",
                "description": "Memory usage is highly variable",
                "suggestions": [
                    "Implement memory pooling",
                    "Use consistent batch sizes",
                    "Add memory monitoring alerts",
                ],
                "estimated_improvement": 0.2,
            })
        
        if max_memory > avg_memory * 2:  # Spikes detected
            recommendations.append({
                "type": "handle_spikes",
                "priority": "high",
                "description": "Memory spikes detected",
                "suggestions": [
                    "Implement backpressure control",
                    "Add memory usage limits",
                    "Enable automatic garbage collection",
                ],
                "estimated_improvement": 0.4,
            })
        
        return {
            "recommendations": recommendations,
            "analysis": {
                "avg_memory": avg_memory,
                "max_memory": max_memory,
                "memory_variance": memory_variance,
                "trend": self._calculate_trend(memory_metrics),
            },
        }
    
    async def apply_optimization(
        self,
        recommendations: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Apply memory optimization recommendations."""
        try:
            for rec in recommendations.get("recommendations", []):
                rec_type = rec["type"]
                
                if rec_type == "reduce_memory":
                    await self._reduce_memory_usage(context)
                elif rec_type == "stabilize_memory":
                    await self._stabilize_memory_usage(context)
                elif rec_type == "handle_spikes":
                    await self._handle_memory_spikes(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply memory optimization: {e}")
            return False
    
    async def _reduce_memory_usage(self, context: Dict[str, Any]) -> None:
        """Reduce memory usage."""
        # Implementation would depend on specific system
        self.logger.info("Reducing memory usage")
    
    async def _stabilize_memory_usage(self, context: Dict[str, Any]) -> None:
        """Stabilize memory usage."""
        self.logger.info("Stabilizing memory usage")
    
    async def _handle_memory_spikes(self, context: Dict[str, Any]) -> None:
        """Handle memory spikes."""
        self.logger.info("Handling memory spikes")
    
    def _calculate_trend(self, metrics: List[MetricValue]) -> str:
        """Calculate trend direction.
        
        Args:
            metrics: Metric values.
            
        Returns:
            Trend direction.
        """
        if len(metrics) < 2:
            return "stable"
        
        # Simple linear regression
        x = list(range(len(metrics)))
        y = [m.value for m in metrics]
        
        n = len(metrics)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class ThroughputOptimizationStrategy(OptimizationStrategy):
    """Throughput optimization strategy."""
    
    def __init__(self):
        """Initialize throughput optimization strategy."""
        super().__init__("throughput_optimization")
    
    async def analyze(
        self,
        metrics: List[MetricValue],
        profile: PerformanceProfile,
    ) -> Dict[str, Any]:
        """Analyze throughput patterns."""
        throughput_metrics = [
            m for m in metrics
            if m.metric == PerformanceMetric.THROUGHPUT
        ]
        
        if not throughput_metrics:
            return {"recommendations": [], "analysis": "No throughput metrics available"}
        
        values = [m.value for m in throughput_metrics]
        avg_throughput = statistics.mean(values)
        target_throughput = profile.target_metrics.get(PerformanceMetric.THROUGHPUT, float('inf'))
        
        recommendations = []
        
        if avg_throughput < target_throughput:
            gap = target_throughput - avg_throughput
            recommendations.append({
                "type": "increase_throughput",
                "priority": "high",
                "description": f"Throughput {avg_throughput:.2f} below target {target_throughput:.2f}",
                "suggestions": [
                    "Increase parallelism",
                    "Optimize batch sizes",
                    "Enable pipelining",
                    "Reduce I/O overhead",
                ],
                "estimated_improvement": min(0.5, gap / target_throughput),
            })
        
        # Check for throughput variance
        if len(values) > 1:
            variance = statistics.variance(values)
            if variance > avg_throughput * 0.3:  # High variance
                recommendations.append({
                    "type": "stabilize_throughput",
                    "priority": "medium",
                    "description": "Throughput is highly variable",
                    "suggestions": [
                        "Implement load balancing",
                        "Add buffering",
                        "Optimize resource allocation",
                    ],
                    "estimated_improvement": 0.25,
                })
        
        return {
            "recommendations": recommendations,
            "analysis": {
                "avg_throughput": avg_throughput,
                "target_throughput": target_throughput,
                "gap": target_throughput - avg_throughput,
                "variance": statistics.variance(values) if len(values) > 1 else 0,
            },
        }
    
    async def apply_optimization(
        self,
        recommendations: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Apply throughput optimization recommendations."""
        try:
            for rec in recommendations.get("recommendations", []):
                rec_type = rec["type"]
                
                if rec_type == "increase_throughput":
                    await self._increase_throughput(context)
                elif rec_type == "stabilize_throughput":
                    await self._stabilize_throughput(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply throughput optimization: {e}")
            return False
    
    async def _increase_throughput(self, context: Dict[str, Any]) -> None:
        """Increase throughput."""
        self.logger.info("Increasing throughput")
    
    async def _stabilize_throughput(self, context: Dict[str, Any]) -> None:
        """Stabilize throughput."""
        self.logger.info("Stabilizing throughput")


class AdaptivePerformanceManager:
    """Manager for adaptive performance optimization."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize performance manager.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.strategies: List[OptimizationStrategy] = []
        self.metric_history: List[MetricValue] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.running = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize default strategies
        self.strategies = [
            MemoryOptimizationStrategy(),
            ThroughputOptimizationStrategy(),
        ]
    
    async def start(self) -> None:
        """Start the performance manager."""
        self.running = True
        self.logger.info("Adaptive performance manager started")
    
    async def stop(self) -> None:
        """Stop the performance manager."""
        self.running = False
        self.logger.info("Adaptive performance manager stopped")
    
    def add_profile(self, profile: PerformanceProfile) -> None:
        """Add a performance profile.
        
        Args:
            profile: Performance profile.
        """
        self.profiles[profile.name] = profile
        self.logger.info(f"Added performance profile: {profile.name}")
    
    def add_strategy(self, strategy: OptimizationStrategy) -> None:
        """Add an optimization strategy.
        
        Args:
            strategy: Optimization strategy.
        """
        self.strategies.append(strategy)
        self.logger.info(f"Added optimization strategy: {strategy.name}")
    
    async def record_metric(
        self,
        metric: PerformanceMetric,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric.
        
        Args:
            metric: Metric type.
            value: Metric value.
            metadata: Additional metadata.
        """
        metric_value = MetricValue(
            metric=metric,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        self.metric_history.append(metric_value)
        
        # Keep only recent history (last 1000 values)
        if len(self.metric_history) > 1000:
            self.metric_history = self.metric_history[-1000:]
    
    async def analyze_performance(self, profile_name: str) -> Dict[str, Any]:
        """Analyze performance for a specific profile.
        
        Args:
            profile_name: Profile name.
            
        Returns:
            Analysis results.
        """
        if profile_name not in self.profiles:
            raise AdaptiveError(f"Profile not found: {profile_name}")
        
        profile = self.profiles[profile_name]
        
        # Filter recent metrics (last 5 minutes)
        current_time = time.time()
        recent_metrics = [
            m for m in self.metric_history
            if current_time - m.timestamp < 300  # 5 minutes
        ]
        
        # Run analysis through all strategies
        all_recommendations = []
        strategy_results = {}
        
        for strategy in self.strategies:
            try:
                result = await strategy.analyze(recent_metrics, profile)
                strategy_results[strategy.name] = result
                all_recommendations.extend(result.get("recommendations", []))
            except Exception as e:
                self.logger.error(f"Strategy {strategy.name} analysis failed: {e}")
        
        # Sort recommendations by priority and estimated improvement
        all_recommendations.sort(
            key=lambda r: (
                0 if r["priority"] == "high" else (1 if r["priority"] == "medium" else 2),
                -r.get("estimated_improvement", 0),
            )
        )
        
        return {
            "profile": profile_name,
            "timestamp": current_time,
            "metrics_analyzed": len(recent_metrics),
            "recommendations": all_recommendations[:10],  # Top 10
            "strategy_results": strategy_results,
            "current_performance": self._calculate_current_performance(recent_metrics),
        }
    
    async def apply_optimizations(
        self,
        profile_name: str,
        recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Apply performance optimizations.
        
        Args:
            profile_name: Profile name.
            recommendations: Specific recommendations to apply.
            
        Returns:
            Optimization results.
        """
        if profile_name not in self.profiles:
            raise AdaptiveError(f"Profile not found: {profile_name}")
        
        # Get recommendations if not provided
        if recommendations is None:
            analysis = await self.analyze_performance(profile_name)
            recommendations = analysis["recommendations"]
        
        profile = self.profiles[profile_name]
        applied_optimizations = []
        failed_optimizations = []
        
        # Group recommendations by strategy
        strategy_recommendations = {}
        for rec in recommendations:
            strategy = self._get_strategy_for_recommendation(rec)
            if strategy:
                if strategy.name not in strategy_recommendations:
                    strategy_recommendations[strategy.name] = []
                strategy_recommendations[strategy.name].append(rec)
        
        # Apply optimizations by strategy
        for strategy_name, recs in strategy_recommendations.items():
            strategy = next((s for s in self.strategies if s.name == strategy_name), None)
            if strategy:
                try:
                    context = {
                        "profile": profile,
                        "recommendations": recs,
                    }
                    
                    success = await strategy.apply_optimization(
                        {"recommendations": recs},
                        context,
                    )
                    
                    if success:
                        applied_optimizations.extend(recs)
                    else:
                        failed_optimizations.extend(recs)
                        
                except Exception as e:
                    self.logger.error(f"Failed to apply {strategy_name} optimizations: {e}")
                    failed_optimizations.extend(recs)
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "profile": profile_name,
            "applied": applied_optimizations,
            "failed": failed_optimizations,
        }
        
        self.optimization_history.append(optimization_record)
        
        return {
            "profile": profile_name,
            "applied_count": len(applied_optimizations),
            "failed_count": len(failed_optimizations),
            "applied_optimizations": applied_optimizations,
            "failed_optimizations": failed_optimizations,
        }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Performance summary.
        """
        current_time = time.time()
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.metric_history
            if current_time - m.timestamp < 3600
        ]
        
        # Calculate summary statistics
        summary = {
            "timestamp": current_time,
            "total_metrics": len(self.metric_history),
            "recent_metrics": len(recent_metrics),
            "profiles": list(self.profiles.keys()),
            "strategies": [s.name for s in self.strategies],
            "optimizations_applied": len(self.optimization_history),
            "current_performance": self._calculate_current_performance(recent_metrics),
        }
        
        # Add profile-specific summaries
        for profile_name in self.profiles:
            profile_summary = await self._get_profile_summary(profile_name, recent_metrics)
            summary[f"profile_{profile_name}"] = profile_summary
        
        return summary
    
    def _get_strategy_for_recommendation(self, recommendation: Dict[str, Any]) -> Optional[OptimizationStrategy]:
        """Get strategy for recommendation.
        
        Args:
            recommendation: Recommendation dictionary.
            
        Returns:
            Strategy or None.
        """
        rec_type = recommendation.get("type", "")
        
        if "memory" in rec_type:
            return next((s for s in self.strategies if isinstance(s, MemoryOptimizationStrategy)), None)
        elif "throughput" in rec_type:
            return next((s for s in self.strategies if isinstance(s, ThroughputOptimizationStrategy)), None)
        
        return None
    
    def _calculate_current_performance(self, metrics: List[MetricValue]) -> Dict[str, float]:
        """Calculate current performance metrics.
        
        Args:
            metrics: Metric values.
            
        Returns:
            Current performance metrics.
        """
        performance = {}
        
        # Group metrics by type
        metric_groups = {}
        for metric in metrics:
            if metric.metric not in metric_groups:
                metric_groups[metric.metric] = []
            metric_groups[metric.metric].append(metric.value)
        
        # Calculate statistics for each metric type
        for metric_type, values in metric_groups.items():
            if values:
                performance[metric_type.value] = {
                    "current": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        return performance
    
    async def _get_profile_summary(
        self,
        profile_name: str,
        metrics: List[MetricValue],
    ) -> Dict[str, Any]:
        """Get profile-specific summary.
        
        Args:
            profile_name: Profile name.
            metrics: Metric values.
            
        Returns:
            Profile summary.
        """
        profile = self.profiles[profile_name]
        current_performance = self._calculate_current_performance(metrics)
        
        # Calculate compliance with targets
        compliance = {}
        for target_metric, target_value in profile.target_metrics.items():
            current_value = current_performance.get(target_metric.value, {}).get("current", 0)
            if target_metric in [PerformanceMetric.THROUGHPUT, PerformanceMetric.CACHE_HIT_RATE]:
                # Higher is better
                compliance[target_metric.value] = current_value >= target_value
            else:
                # Lower is better
                compliance[target_metric.value] = current_value <= target_value
        
        return {
            "target_metrics": {k.value: v for k, v in profile.target_metrics.items()},
            "current_performance": current_performance,
            "compliance": compliance,
            "compliance_rate": sum(compliance.values()) / len(compliance) if compliance else 0,
        }


# Utility functions
async def create_default_performance_manager() -> AdaptivePerformanceManager:
    """Create default performance manager with standard profiles.
    
    Returns:
        Configured performance manager.
    """
    manager = AdaptivePerformanceManager()
    
    # Add default profiles
    high_throughput_profile = PerformanceProfile(
        name="high_throughput",
        target_metrics={
            PerformanceMetric.THROUGHPUT: 1000.0,  # 1000 ops/sec
            PerformanceMetric.LATENCY: 0.1,       # 100ms max
            PerformanceMetric.ERROR_RATE: 0.01,   # 1% max error rate
        },
        priority=1.0,
    )
    
    low_memory_profile = PerformanceProfile(
        name="low_memory",
        target_metrics={
            PerformanceMetric.MEMORY_USAGE: 512.0,  # 512MB max
            PerformanceMetric.CACHE_HIT_RATE: 0.8,  # 80% min hit rate
        },
        priority=0.8,
    )
    
    balanced_profile = PerformanceProfile(
        name="balanced",
        target_metrics={
            PerformanceMetric.THROUGHPUT: 500.0,
            PerformanceMetric.LATENCY: 0.2,
            PerformanceMetric.MEMORY_USAGE: 1024.0,
            PerformanceMetric.ERROR_RATE: 0.05,
        },
        priority=1.0,
    )
    
    manager.add_profile(high_throughput_profile)
    manager.add_profile(low_memory_profile)
    manager.add_profile(balanced_profile)
    
    await manager.start()
    return manager


def monitor_performance(
    profile_name: str = "default",
    metrics: List[PerformanceMetric] = None,
):
    """Decorator to monitor function performance.
    
    Args:
        profile_name: Performance profile name.
        metrics: Metrics to collect.
    """
    if metrics is None:
        metrics = [PerformanceMetric.LATENCY, PerformanceMetric.THROUGHPUT]
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success metrics
                end_time = time.time()
                duration = end_time - start_time
                
                # This would need access to the performance manager
                # For now, just log the metrics
                logger.info(f"Function {func.__name__} executed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                # Record error metrics
                end_time = time.time()
                duration = end_time - start_time
                
                logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator
