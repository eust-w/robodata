"""Dataset inspection module for Robodata.

This module provides tools for analyzing robotics datasets without full loading,
including schema inference, statistics collection, and format detection.
"""

from robodata.inspect.inspector import InspectionOptions, Inspector
from robodata.inspect.schema_analyzer import SchemaAnalyzer
from robodata.inspect.stats_collector import StatsCollector

__all__ = [
    "Inspector",
    "InspectionOptions",
    "SchemaAnalyzer",
    "StatsCollector",
]
