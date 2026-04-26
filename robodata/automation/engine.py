"""Intelligent automation system for Robodata.

Provides automated decision making, optimization, and
self-healing capabilities for data processing workflows.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from robodata.core.logging_system import get_logger
from robodata.core.performance_monitor import get_performance_manager
from robodata.core.config_manager_v2 import get_config_manager, temporary_config


class AutomationLevel(Enum):
    """Levels of automation."""
    MANUAL = "manual"
    ASSISTED = "assisted"
    SEMI_AUTOMATIC = "semi_automatic"
    FULL_AUTOMATIC = "full_automatic"


class DecisionType(Enum):
    """Types of automated decisions."""
    RESOURCE_ALLOCATION = "resource_allocation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_RECOVERY = "error_recovery"
    QUALITY_IMPROVEMENT = "quality_improvement"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class AutomationDecision:
    """Decision made by automation system."""
    decision_type: DecisionType
    confidence: float  # 0.0 to 1.0
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_impact: str = ""
    risk_level: float = 0.0  # 0.0 to 1.0


@dataclass
class AutomationResult:
    """Result of automated action."""
    decision: AutomationDecision
    success: bool
    actual_impact: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class AutomationRule(ABC):
    """Abstract base class for automation rules."""
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
        """Evaluate rule and return decision if triggered."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Rule priority (higher = more important)."""
        pass


class PerformanceOptimizationRule(AutomationRule):
    """Rule for automatic performance optimization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_manager = get_performance_manager()
    
    @property
    def name(self) -> str:
        return "performance_optimization"
    
    @property
    def priority(self) -> int:
        return 80  # High priority
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
        """Evaluate performance and suggest optimizations."""
        report = self.performance_manager.get_performance_report()
        
        # Check CPU usage
        cpu_usage = report.get("resources", {}).get("cpu_usage", {}).get("current", 0)
        if cpu_usage < 50:
            return AutomationDecision(
                decision_type=DecisionType.PERFORMANCE_OPTIMIZATION,
                confidence=0.8,
                action="increase_parallelism",
                parameters={"num_workers": min(8, context.get("current_workers", 1) * 2)},
                rationale="CPU usage is low, can increase parallelism",
                expected_impact="20-40% performance improvement",
                risk_level=0.1
            )
        
        # Check memory usage
        memory_usage = report.get("resources", {}).get("memory_usage", {}).get("current", 0)
        if memory_usage > 80:
            return AutomationDecision(
                decision_type=DecisionType.PERFORMANCE_OPTIMIZATION,
                confidence=0.9,
                action="reduce_memory_usage",
                parameters={
                    "batch_size": max(10, context.get("batch_size", 100) // 2),
                    "enable_disk_cache": True
                },
                rationale="Memory usage is high, need to reduce memory footprint",
                expected_impact="30-50% memory reduction",
                risk_level=0.2
            )
        
        return None


class ErrorRecoveryRule(AutomationRule):
    """Rule for automatic error recovery."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.error_history: List[Dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        return "error_recovery"
    
    @property
    def priority(self) -> int:
        return 90  # Very high priority
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
        """Evaluate errors and suggest recovery actions."""
        recent_errors = context.get("recent_errors", [])
        
        if not recent_errors:
            return None
        
        # Analyze error patterns
        error_types = {}
        for error in recent_errors:
            error_type = error.get("type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Most common error type
        most_common_error = max(error_types.items(), key=lambda x: x[1])
        error_type, count = most_common_error
        
        if count >= 3:  # Threshold for automation
            if error_type == "memory_error":
                return AutomationDecision(
                    decision_type=DecisionType.ERROR_RECOVERY,
                    confidence=0.9,
                    action="reduce_memory_pressure",
                    parameters={
                        "num_workers": 1,
                        "batch_size": 10,
                        "enable_streaming": True
                    },
                    rationale=f"Multiple memory errors detected ({count} occurrences)",
                    expected_impact="Eliminate memory errors",
                    risk_level=0.1
                )
            
            elif error_type == "io_error":
                return AutomationDecision(
                    decision_type=DecisionType.ERROR_RECOVERY,
                    confidence=0.8,
                    action="improve_io_handling",
                    parameters={
                        "retry_attempts": 3,
                        "retry_delay": 1.0,
                        "enable_io_buffering": True
                    },
                    rationale=f"Multiple IO errors detected ({count} occurrences)",
                    expected_impact="Reduce IO errors by 80%",
                    risk_level=0.2
                )
        
        return None


class QualityImprovementRule(AutomationRule):
    """Rule for automatic quality improvement."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "quality_improvement"
    
    @property
    def priority(self) -> int:
        return 60  # Medium priority
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
        """Evaluate data quality and suggest improvements."""
        quality_metrics = context.get("quality_metrics", {})
        
        if quality_metrics.get("overall_score", 1.0) < 0.7:
            return AutomationDecision(
                decision_type=DecisionType.QUALITY_IMPROVEMENT,
                confidence=0.7,
                action="enable_data_cleaning",
                parameters={
                    "remove_outliers": True,
                    "fill_missing_values": True,
                    "normalize_data": True
                },
                rationale="Data quality score is below threshold",
                expected_impact="20-30% quality improvement",
                risk_level=0.3
            )
        
        # Check for specific quality issues
        missing_data_rate = quality_metrics.get("missing_data_rate", 0.0)
        if missing_data_rate > 0.1:
            return AutomationDecision(
                decision_type=DecisionType.QUALITY_IMPROVEMENT,
                confidence=0.8,
                action="handle_missing_data",
                parameters={
                    "strategy": "interpolation",
                    "fill_method": "forward_fill"
                },
                rationale=f"High missing data rate: {missing_data_rate:.1%}",
                expected_impact="Reduce missing data by 90%",
                risk_level=0.2
            )
        
        return None


class ResourceAllocationRule(AutomationRule):
    """Rule for intelligent resource allocation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "resource_allocation"
    
    @property
    def priority(self) -> int:
        return 70  # High priority
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
        """Evaluate resource usage and suggest allocation changes."""
        workload_size = context.get("workload_size", 0)
        current_resources = context.get("current_resources", {})
        
        if workload_size > 10000:  # Large workload
            available_cores = current_resources.get("cpu_cores", 4)
            if current_resources.get("num_workers", 1) < available_cores:
                return AutomationDecision(
                    decision_type=DecisionType.RESOURCE_ALLOCATION,
                    confidence=0.8,
                    action="scale_up_workers",
                    parameters={"num_workers": available_cores},
                    rationale="Large workload detected, scaling up workers",
                    expected_impact="2-4x processing speed",
                    risk_level=0.2
                )
        
        elif workload_size < 100:  # Small workload
            if current_resources.get("num_workers", 1) > 2:
                return AutomationDecision(
                    decision_type=DecisionType.RESOURCE_ALLOCATION,
                    confidence=0.7,
                    action="scale_down_workers",
                    parameters={"num_workers": 2},
                    rationale="Small workload, reducing resource usage",
                    expected_impact="50% resource savings",
                    risk_level=0.1
                )
        
        return None


class AutomationEngine:
    """Main automation engine that evaluates rules and executes decisions."""
    
    def __init__(self, automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATIC):
        self.automation_level = automation_level
        self.logger = get_logger(__name__)
        self.rules: List[AutomationRule] = []
        self.decision_history: List[AutomationDecision] = []
        self.execution_history: List[AutomationResult] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default automation rules."""
        self.rules = [
            PerformanceOptimizationRule(),
            ErrorRecoveryRule(),
            QualityImprovementRule(),
            ResourceAllocationRule(),
        ]
        
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_rule(self, rule: AutomationRule):
        """Add a custom automation rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Added automation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an automation rule."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        self.logger.info(f"Removed automation rule: {rule_name}")
    
    def evaluate_decisions(self, context: Dict[str, Any]) -> List[AutomationDecision]:
        """Evaluate all rules and return decisions."""
        decisions = []
        
        for rule in self.rules:
            try:
                decision = rule.evaluate(context)
                if decision:
                    decisions.append(decision)
                    self.decision_history.append(decision)
                    self.logger.debug(f"Rule {rule.name} triggered: {decision.action}")
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        # Filter decisions based on automation level
        if self.automation_level == AutomationLevel.MANUAL:
            return []  # No automatic decisions
        elif self.automation_level == AutomationLevel.ASSISTED:
            # Only low-risk decisions
            decisions = [d for d in decisions if d.risk_level < 0.3]
        elif self.automation_level == AutomationLevel.SEMI_AUTOMATIC:
            # Medium risk and below
            decisions = [d for d in decisions if d.risk_level < 0.5]
        
        # Sort by confidence and priority
        decisions.sort(key=lambda d: (d.confidence, d.priority), reverse=True)
        
        return decisions
    
    async def execute_decision(self, decision: AutomationDecision) -> AutomationResult:
        """Execute an automation decision."""
        start_time = time.time()
        result = AutomationResult(decision=decision, success=False)
        
        try:
            self.logger.info(f"Executing decision: {decision.action}")
            
            if decision.action == "increase_parallelism":
                result = await self._execute_increase_parallelism(decision)
            elif decision.action == "reduce_memory_usage":
                result = await self._execute_reduce_memory_usage(decision)
            elif decision.action == "reduce_memory_pressure":
                result = await self._execute_reduce_memory_pressure(decision)
            elif decision.action == "improve_io_handling":
                result = await self._execute_improve_io_handling(decision)
            elif decision.action == "enable_data_cleaning":
                result = await self._execute_enable_data_cleaning(decision)
            elif decision.action == "handle_missing_data":
                result = await self._execute_handle_missing_data(decision)
            elif decision.action == "scale_up_workers":
                result = await self._execute_scale_up_workers(decision)
            elif decision.action == "scale_down_workers":
                result = await self._execute_scale_down_workers(decision)
            else:
                result.errors.append(f"Unknown action: {decision.action}")
        
        except Exception as e:
            result.errors.append(f"Execution failed: {e}")
            self.logger.error(f"Decision execution failed: {e}")
        
        result.execution_time = time.time() - start_time
        self.execution_history.append(result)
        
        return result
    
    async def _execute_increase_parallelism(self, decision: AutomationDecision) -> AutomationResult:
        """Execute parallelism increase."""
        num_workers = decision.parameters.get("num_workers", 2)
        
        with temporary_config(conversion_num_workers=num_workers):
            self.logger.info(f"Increased workers to {num_workers}")
            return AutomationResult(
                decision=decision,
                success=True,
                actual_impact={"workers_set": num_workers}
            )
    
    async def _execute_reduce_memory_usage(self, decision: AutomationDecision) -> AutomationResult:
        """Execute memory usage reduction."""
        batch_size = decision.parameters.get("batch_size", 50)
        enable_disk_cache = decision.parameters.get("enable_disk_cache", False)
        
        with temporary_config(
            performance_chunk_size=batch_size,
            cache_enable_disk_cache=enable_disk_cache
        ):
            self.logger.info(f"Reduced batch size to {batch_size}, disk cache: {enable_disk_cache}")
            return AutomationResult(
                decision=decision,
                success=True,
                actual_impact={"batch_size": batch_size, "disk_cache": enable_disk_cache}
            )
    
    async def _execute_reduce_memory_pressure(self, decision: AutomationDecision) -> AutomationResult:
        """Execute memory pressure reduction."""
        params = decision.parameters
        
        with temporary_config(
            conversion_num_workers=params.get("num_workers", 1),
            performance_chunk_size=params.get("batch_size", 10),
            streaming_enable_streaming=params.get("enable_streaming", True)
        ):
            self.logger.info("Applied memory pressure reduction settings")
            return AutomationResult(
                decision=decision,
                success=True,
                actual_impact=params
            )
    
    async def _execute_improve_io_handling(self, decision: AutomationDecision) -> AutomationResult:
        """Execute IO handling improvement."""
        params = decision.parameters
        
        config_manager = get_config_manager()
        config_manager.set("io.retry_attempts", params.get("retry_attempts", 3))
        config_manager.set("io.retry_delay", params.get("retry_delay", 1.0))
        config_manager.set("io.enable_buffering", params.get("enable_io_buffering", True))
        
        self.logger.info("Improved IO handling settings")
        return AutomationResult(
            decision=decision,
            success=True,
            actual_impact=params
        )
    
    async def _execute_enable_data_cleaning(self, decision: AutomationDecision) -> AutomationResult:
        """Execute data cleaning enablement."""
        params = decision.parameters
        
        config_manager = get_config_manager()
        config_manager.set("preprocessing.enable_cleaning", True)
        config_manager.set("preprocessing.remove_outliers", params.get("remove_outliers", True))
        config_manager.set("preprocessing.fill_missing", params.get("fill_missing_values", True))
        config_manager.set("preprocessing.normalize", params.get("normalize_data", True))
        
        self.logger.info("Enabled data cleaning")
        return AutomationResult(
            decision=decision,
            success=True,
            actual_impact=params
        )
    
    async def _execute_handle_missing_data(self, decision: AutomationDecision) -> AutomationResult:
        """Execute missing data handling."""
        params = decision.parameters
        
        config_manager = get_config_manager()
        config_manager.set("preprocessing.missing_strategy", params.get("strategy", "interpolation"))
        config_manager.set("preprocessing.fill_method", params.get("fill_method", "forward_fill"))
        
        self.logger.info("Configured missing data handling")
        return AutomationResult(
            decision=decision,
            success=True,
            actual_impact=params
        )
    
    async def _execute_scale_up_workers(self, decision: AutomationDecision) -> AutomationResult:
        """Execute worker scaling up."""
        num_workers = decision.parameters.get("num_workers", 4)
        
        with temporary_config(conversion_num_workers=num_workers):
            self.logger.info(f"Scaled up workers to {num_workers}")
            return AutomationResult(
                decision=decision,
                success=True,
                actual_impact={"workers_scaled": num_workers}
            )
    
    async def _execute_scale_down_workers(self, decision: AutomationDecision) -> AutomationResult:
        """Execute worker scaling down."""
        num_workers = decision.parameters.get("num_workers", 2)
        
        with temporary_config(conversion_num_workers=num_workers):
            self.logger.info(f"Scaled down workers to {num_workers}")
            return AutomationResult(
                decision=decision,
                success=True,
                actual_impact={"workers_scaled": num_workers}
            )
    
    async def run_automation_cycle(self, context: Dict[str, Any]) -> List[AutomationResult]:
        """Run a complete automation cycle."""
        self.logger.info("Starting automation cycle")
        
        # Evaluate decisions
        decisions = self.evaluate_decisions(context)
        
        if not decisions:
            self.logger.info("No automation decisions made")
            return []
        
        # Execute decisions
        results = []
        for decision in decisions:
            result = await self.execute_decision(decision)
            results.append(result)
            
            if result.success:
                self.logger.info(f"Successfully executed: {decision.action}")
            else:
                self.logger.error(f"Failed to execute: {decision.action}")
        
        self.logger.info(f"Automation cycle complete: {len(results)} actions taken")
        return results
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics."""
        total_decisions = len(self.decision_history)
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.success)
        
        decision_types = {}
        for decision in self.decision_history:
            dt = decision.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "total_executions": total_executions,
            "success_rate": successful_executions / max(1, total_executions),
            "decision_types": decision_types,
            "automation_level": self.automation_level.value,
            "active_rules": len(self.rules)
        }


class SelfHealingSystem:
    """Self-healing system for automatic problem resolution."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.automation_engine = AutomationEngine()
        self.health_checks: List[Callable] = []
        self.healing_actions: List[AutomationResult] = []
    
    def add_health_check(self, check_func: Callable[[], Dict[str, Any]]):
        """Add a health check function."""
        self.health_checks.append(check_func)
        self.logger.info(f"Added health check: {check_func.__name__}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        health_status = {
            "overall_health": "healthy",
            "checks": {},
            "issues": []
        }
        
        for check in self.health_checks:
            try:
                result = check()
                health_status["checks"][check.__name__] = result
                
                if not result.get("healthy", True):
                    health_status["overall_health"] = "unhealthy"
                    health_status["issues"].append({
                        "check": check.__name__,
                        "issue": result.get("issue", "Unknown"),
                        "severity": result.get("severity", "medium")
                    })
            except Exception as e:
                health_status["checks"][check.__name__] = {"error": str(e)}
                health_status["overall_health"] = "error"
                health_status["issues"].append({
                    "check": check.__name__,
                    "issue": f"Health check failed: {e}",
                    "severity": "high"
                })
        
        return health_status
    
    async def self_heal(self) -> List[AutomationResult]:
        """Run self-healing process."""
        self.logger.info("Starting self-healing process")
        
        # Run health checks
        health_status = await self.run_health_checks()
        
        if health_status["overall_health"] == "healthy":
            self.logger.info("System is healthy, no healing needed")
            return []
        
        # Create context for automation
        context = {
            "health_status": health_status,
            "issues": health_status["issues"],
            "severity_counts": {}
        }
        
        # Count severity levels
        for issue in health_status["issues"]:
            severity = issue.get("severity", "medium")
            context["severity_counts"][severity] = context["severity_counts"].get(severity, 0) + 1
        
        # Run automation for healing
        healing_results = await self.automation_engine.run_automation_cycle(context)
        self.healing_actions.extend(healing_results)
        
        self.logger.info(f"Self-healing complete: {len(healing_results)} actions taken")
        return healing_results


# Convenience functions
async def run_automation(
    context: Dict[str, Any],
    automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATIC
) -> List[AutomationResult]:
    """Run automation with given context."""
    engine = AutomationEngine(automation_level)
    return await engine.run_automation_cycle(context)


def create_custom_rule(
    name: str,
    priority: int,
    evaluate_func: Callable[[Dict[str, Any]], Optional[AutomationDecision]]
) -> AutomationRule:
    """Create a custom automation rule."""
    
    class CustomRule(AutomationRule):
        def __init__(self):
            self._name = name
            self._priority = priority
            self._evaluate_func = evaluate_func
        
        @property
        def name(self) -> str:
            return self._name
        
        @property
        def priority(self) -> int:
            return self._priority
        
        def evaluate(self, context: Dict[str, Any]) -> Optional[AutomationDecision]:
            return self._evaluate_func(context)
    
    return CustomRule()


# Default health checks
def memory_health_check() -> Dict[str, Any]:
    """Check memory health."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        healthy = memory.percent < 80
        
        return {
            "healthy": healthy,
            "memory_usage": memory.percent,
            "issue": "High memory usage" if not healthy else None,
            "severity": "high" if memory.percent > 90 else "medium"
        }
    except ImportError:
        return {"healthy": True, "issue": "psutil not available"}


def disk_space_health_check() -> Dict[str, Any]:
    """Check disk space health."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        healthy = disk.percent < 90
        
        return {
            "healthy": healthy,
            "disk_usage": disk.percent,
            "issue": "Low disk space" if not healthy else None,
            "severity": "high" if disk.percent > 95 else "medium"
        }
    except ImportError:
        return {"healthy": True, "issue": "psutil not available"}
