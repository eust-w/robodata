"""Robodata - The normalization layer for robotics data.

Robodata converts between robotics episode formats (RLDS, LeRobot v2/v3, Zarr, Rosbag)
with minimal friction. It's the "on-ramp" to the VLA data platform — whatever format
you captured, Robodata makes it usable by downstream tools.

Key Characteristics:
- I/O-bound, no GPU required
- Streaming/lazy architecture for memory efficiency
- LeRobot v3 as the platform's lingua franca

Example:
    >>> import robodata
    >>> info = robodata.inspect("./my_dataset")
    >>> print(info.cameras)
    >>> print(info.missing_required)

    >>> robodata.convert("./rlds_data", "./lerobot_data", target_format="lerobot-v3")
"""

import os

# Suppress TensorFlow verbose logging before any TF import can happen
# This must be done before tensorflow is imported anywhere
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # FATAL only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Suppress oneDNN messages
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
)  # Keep existing if set
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # Google logging: ERROR+

from robodata.validation.engine import (
    ValidationEngine,
    validate_dataset,
    ValidationLevel,
    ValidationCategory,
)
from robodata.analytics.engine import (
    AnalyticsEngine,
    analyze_dataset,
    AnalysisType,
    InsightLevel,
)
from robodata.preprocessing.pipeline import (
    PreprocessingPipeline,
    preprocess_episodes,
    PreprocessingConfig,
    AugmentationType,
)
from robodata.streaming.engine import (
    StreamingEngine,
    stream_process_episodes,
    StreamingMode,
    ChunkStrategy,
)
from robodata.automation.engine import (
    AutomationEngine,
    run_automation,
    AutomationLevel,
    DecisionType,
)
from robodata.visualization.engine import (
    ReportGenerator,
    generate_dataset_report,
    VisualizationType,
    ReportFormat,
)
from robodata.ml.toolkit import (
    MLToolkit,
    train_behavior_cloning,
    train_reward_model,
    ModelType,
    TrainingMode,
)
from robodata.convert.converter_v2 import (
    ConversionConfig,
    ConversionResult,
    Converter,
    convert,
)
from robodata.core.pipeline import (
    ProcessingConfig,
    ProcessingMode,
    ProcessingPipeline,
    Processor,
)
from robodata.core.logging_system import (
    LogConfig,
    LogLevel,
    StructuredLogger,
    configure_logging,
    get_logger,
    setup_logging,
)
from robodata.core.error_handling_v2 import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    handle_errors,
)
from robodata.core.cache_v2 import (
    CacheConfig,
    CacheStrategy,
    CacheManager,
    create_cache,
    cached,
)
from robodata.core.config_manager_v2 import (
    CacheConfig as ConfigCacheConfig,
    ConfigManager,
    ConversionConfig as ConfigConversionConfig,
    LoggingConfig,
    get_config,
    get_config_manager,
    save_config,
    set_config,
    temporary_config,
)
from robodata.core.performance_monitor import (
    PerformanceManager,
    profile_operation,
    start_performance_monitoring,
    stop_performance_monitoring,
)
from robodata.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyArray,
    LazyImage,
)
from robodata.filter.engine import FilterConfig, FilterEngine, FilterResult
from robodata.formats.registry import FormatRegistry
from robodata.inspect.inspector import InspectionOptions, Inspector
from robodata.segment import SegmentAnalyzer, SegmentationReport, SegmentConfig

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core models
    "CameraInfo",
    "DatasetInfo",
    "Dtype",
    "Episode",
    "FieldSchema",
    "Frame",
    "LazyArray",
    "LazyImage",
    # Pipeline and Processing
    "ProcessingConfig",
    "ProcessingMode",
    "ProcessingPipeline",
    "Processor",
    # Logging System
    "LogConfig",
    "LogLevel",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    "setup_logging",
    # Error Handling
    "ErrorCategory",
    "ErrorContext",
    "ErrorHandler",
    "ErrorSeverity",
    "handle_errors",
    # Caching System
    "CacheConfig",
    "CacheStrategy",
    "CacheManager",
    "create_cache",
    "cached",
    # Validation and Analytics
    "ValidationEngine",
    "validate_dataset",
    "ValidationLevel",
    "ValidationCategory",
    "AnalyticsEngine",
    "analyze_dataset",
    "AnalysisType",
    "InsightLevel",
    # Preprocessing
    "PreprocessingPipeline",
    "preprocess_episodes",
    "PreprocessingConfig",
    "AugmentationType",
    # Streaming
    "StreamingEngine",
    "stream_process_episodes",
    "StreamingMode",
    "ChunkStrategy",
    # Automation
    "AutomationEngine",
    "run_automation",
    "AutomationLevel",
    "DecisionType",
    # Visualization and Reporting
    "ReportGenerator",
    "generate_dataset_report",
    "VisualizationType",
    "ReportFormat",
    # Machine Learning
    "MLToolkit",
    "train_behavior_cloning",
    "train_reward_model",
    "ModelType",
    "TrainingMode",
    # Performance Monitoring
    "PerformanceManager",
    "profile_operation",
    "start_performance_monitoring",
    "stop_performance_monitoring",
    # Registry
    "FormatRegistry",
    # Inspector
    "Inspector",
    "InspectionOptions",
    # Converter
    "Converter",
    "ConversionConfig",
    "ConversionResult",
    # Filter
    "FilterConfig",
    "FilterEngine",
    "FilterResult",
    # Segment
    "SegmentAnalyzer",
    "SegmentConfig",
    "SegmentationReport",
    # Module-level functions
    "inspect",
    "convert",
]


def inspect(
    path: str,
    format: str | None = None,
    **options,
) -> DatasetInfo:
    """Inspect a dataset and return structured information.

    Args:
        path: Path to dataset.
        format: Format hint (auto-detected if None).
        **options: InspectionOptions fields.

    Returns:
        DatasetInfo with schema, stats, and conversion readiness.

    Example:
        >>> info = robodata.inspect("./my_dataset")
        >>> print(info.cameras)
        >>> print(info.missing_required)

        >>> info = robodata.inspect("./dataset", deep_scan=True, sample_episodes=10)
    """
    opts = InspectionOptions(**options)
    inspector = Inspector(opts)
    return inspector.inspect(path, format)
