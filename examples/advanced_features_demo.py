#!/usr/bin/env python3
"""
Comprehensive example showcasing Robodata v2.0 advanced features.

This example demonstrates:
- Data validation and analytics
- Intelligent preprocessing
- Streaming processing
- Automation and self-healing
- Visualization and reporting
- Machine learning integration
"""

import asyncio
import time
from pathlib import Path

import robodata
from robodata.core.config_manager_v2 import temporary_config
from robodata.core.performance_monitor import start_performance_monitoring, profile_operation
from robodata.core.logging_system import get_logger, setup_logging, LogLevel
from robodata.validation.engine import validate_dataset, ValidationLevel
from robodata.analytics.engine import analyze_dataset, AnalysisType
from robodata.preprocessing.pipeline import preprocess_episodes, PreprocessingConfig, AugmentationType
from robodata.streaming.engine import stream_process_episodes, create_streaming_config, StreamingMode
from robodata.automation.engine import run_automation, AutomationLevel
from robodata.visualization.engine import generate_dataset_report, ReportFormat
from robodata.ml.toolkit import train_behavior_cloning, ModelType


def setup_example():
    """Setup the example environment."""
    # Setup logging
    setup_logging(LogLevel.INFO, enable_rich=True)
    logger = get_logger(__name__)
    
    # Start performance monitoring
    start_performance_monitoring()
    
    logger.info("🚀 Robodata v2.0 Advanced Features Example")
    return logger


def load_sample_dataset():
    """Load a sample dataset for demonstration."""
    logger = get_logger(__name__)
    
    # In a real scenario, you would load your actual dataset
    # For this example, we'll create a mock dataset path
    dataset_path = Path("./sample_dataset")
    
    if not dataset_path.exists():
        logger.warning(f"Sample dataset not found at {dataset_path}")
        logger.info("Using mock dataset for demonstration")
        # Create mock episodes for demonstration
        from robodata.core.models import Episode, Frame, LazyImage
        import numpy as np
        
        episodes = []
        for i in range(10):
            frames = []
            for j in range(100):
                frame = Frame(
                    index=j,
                    timestamp=j * 0.033,  # 30Hz
                    state=np.random.randn(10).astype(np.float32),
                    action=np.random.randn(7).astype(np.float32),
                    images={
                        "camera_1": LazyImage(224, 224, 3, "rgb"),
                        "camera_2": LazyImage(224, 224, 3, "rgb")
                    },
                    is_first=(j == 0),
                    is_last=(j == 99)
                )
                frames.append(frame)
            
            episode = Episode(
                episode_id=f"episode_{i:03d}",
                metadata={"task": "pick_and_place"},
                language_instruction="Pick up the object and place it in the target location",
                frames=lambda: iter(frames)
            )
            episodes.append(episode)
        
        # Create mock dataset info
        from robodata.core.models import DatasetInfo, CameraInfo, Dtype, FieldSchema
        dataset_info = DatasetInfo(
            path=dataset_path,
            format="mock",
            num_episodes=len(episodes),
            total_frames=len(episodes) * 100,
            schema={
                "observation.state": FieldSchema("observation.state", (10,), Dtype.FLOAT32),
                "action": FieldSchema("action", (7,), Dtype.FLOAT32),
                "images.camera_1": FieldSchema("images.camera_1", (224, 224, 3), Dtype.UINT8),
                "images.camera_2": FieldSchema("images.camera_2", (224, 224, 3), Dtype.UINT8),
            },
            cameras={
                "camera_1": CameraInfo("camera_1", 224, 224, 3, "rgb"),
                "camera_2": CameraInfo("camera_2", 224, 224, 3, "rgb"),
            },
            inferred_fps=30.0,
            inferred_robot_type="franka"
        )
        
        return dataset_info, episodes
    
    # Real dataset loading would be:
    # dataset_info = robodata.inspect(dataset_path)
    # episodes = list(load_episodes(dataset_path))
    
    return None, None


def demonstrate_validation(dataset_info, episodes):
    """Demonstrate data validation capabilities."""
    logger = get_logger(__name__)
    
    logger.info("🔍 Demonstrating Data Validation")
    
    # Validate dataset
    with profile_operation("dataset_validation"):
        validation_result = validate_dataset(dataset_info, episodes)
    
    logger.info(f"Validation passed: {validation_result.success}")
    logger.info(f"Issues found: {len(validation_result.issues)}")
    
    # Show critical issues
    critical_issues = validation_result.critical_issues
    if critical_issues:
        logger.warning(f"⚠️  {len(critical_issues)} critical issues:")
        for issue in critical_issues[:3]:  # Show first 3
            logger.warning(f"  - {issue.message}")
    
    # Show recommendations
    if validation_result.recommendations:
        logger.info("💡 Recommendations:")
        for rec in validation_result.recommendations[:3]:
            logger.info(f"  - {rec}")
    
    return validation_result


def demonstrate_analytics(dataset_info, episodes):
    """Demonstrate data analytics capabilities."""
    logger = get_logger(__name__)
    
    logger.info("📊 Demonstrating Data Analytics")
    
    # Analyze dataset
    with profile_operation("dataset_analytics"):
        analysis_result = analyze_dataset(
            dataset_info, 
            episodes,
            analysis_types=[AnalysisType.STATISTICAL, AnalysisType.QUALITY]
        )
    
    logger.info(f"Analysis complete: {len(analysis_result.insights)} insights generated")
    
    # Show high-priority insights
    high_insights = analysis_result.high_insights
    if high_insights:
        logger.info("🔍 High Priority Insights:")
        for insight in high_insights[:3]:
            logger.info(f"  - {insight.title}: {insight.description}")
    
    # Show quality score
    if "quality" in analysis_result.statistics:
        quality_score = analysis_result.statistics["quality"].get("overall_score", 0)
        logger.info(f"📈 Data Quality Score: {quality_score:.2f}")
    
    return analysis_result


def demonstrate_preprocessing(episodes):
    """Demonstrate data preprocessing capabilities."""
    logger = get_logger(__name__)
    
    logger.info("🧹 Demonstrating Data Preprocessing")
    
    # Configure preprocessing
    config = PreprocessingConfig(
        enable_cleaning=True,
        enable_normalization=True,
        enable_augmentation=True,
        normalization_config={"method": "standard"},
        augmentation_config={
            "factor": 2,
            "types": [AugmentationType.NOISE_ADDITION, AugmentationType.DROPOUT],
            "noise": {"action_std": 0.01, "state_std": 0.01},
            "dropout": {"action_rate": 0.1, "state_rate": 0.05}
        }
    )
    
    # Preprocess episodes
    with profile_operation("data_preprocessing"):
        result = preprocess_episodes(episodes, config)
    
    logger.info(f"Preprocessing complete: {len(result.processed_episodes)} episodes processed")
    logger.info(f"Original episodes: {result.statistics.get('data_cleaner', {}).get('original_episodes', 0)}")
    logger.info(f"Augmented episodes: {result.statistics.get('data_augmenter', {}).get('augmented_episodes', 0)}")
    
    return result.processed_episodes


async def demonstrate_streaming(episodes):
    """Demonstrate streaming processing capabilities."""
    logger = get_logger(__name__)
    
    logger.info("🌊 Demonstrating Streaming Processing")
    
    # Configure streaming
    config = create_streaming_config(
        mode=StreamingMode.PARALLEL,
        chunk_strategy="fixed_size",
        chunk_size=50,
        num_workers=2,
        enable_prefetch=True
    )
    
    # Stream process episodes
    with profile_operation("streaming_processing"):
        chunks = await stream_process_episodes(episodes, config)
    
    logger.info(f"Streaming complete: {len(chunks)} chunks processed")
    
    # Show chunk statistics
    if chunks:
        chunk_sizes = [len(chunk) for chunk in chunks]
        logger.info(f"Average chunk size: {np.mean(chunk_sizes):.1f}")
        logger.info(f"Total frames processed: {sum(chunk_sizes)}")
    
    return chunks


async def demonstrate_automation(episodes):
    """Demonstrate automation and self-healing capabilities."""
    logger = get_logger(__name__)
    
    logger.info("🤖 Demonstrating Automation System")
    
    # Create context for automation
    context = {
        "workload_size": len(episodes) * 100,  # Approximate frame count
        "current_resources": {"cpu_cores": 4, "memory_gb": 8},
        "recent_errors": [],
        "quality_metrics": {"overall_score": 0.8}
    }
    
    # Run automation
    with profile_operation("automation_cycle"):
        automation_results = await run_automation(
            context,
            automation_level=AutomationLevel.SEMI_AUTOMATIC
        )
    
    logger.info(f"Automation complete: {len(automation_results)} actions taken")
    
    # Show automation actions
    for result in automation_results:
        if result.success:
            logger.info(f"✅ {result.decision.action}: {result.decision.rationale}")
        else:
            logger.error(f"❌ {result.decision.action}: Failed")
    
    return automation_results


def demonstrate_visualization(dataset_info, episodes):
    """Demonstrate visualization and reporting capabilities."""
    logger = get_logger(__name__)
    
    logger.info("📈 Demonstrating Visualization and Reporting")
    
    # Generate comprehensive report
    with profile_operation("report_generation"):
        report = generate_dataset_report(
            dataset_info,
            episodes,
            output_dir=Path("./reports"),
            format=ReportFormat.HTML
        )
    
    logger.info(f"Report generated: {report.title}")
    logger.info(f"Sections: {len(report.sections)}")
    logger.info(f"Visualizations: {sum(len(s.visualizations) for s in report.sections)}")
    
    # Show report sections
    for section in report.sections:
        viz_count = len(section.visualizations)
        logger.info(f"  - {section.title} ({viz_count} visualizations)")
    
    return report


def demonstrate_ml_integration(episodes):
    """Demonstrate machine learning integration."""
    logger = get_logger(__name__)
    
    logger.info("🧠 Demonstrating ML Integration")
    
    # Train behavior cloning model
    try:
        with profile_operation("behavior_cloning_training"):
            training_result = train_behavior_cloning(
                episodes[:5],  # Use subset for demo
                model_name="demo_bc_model"
            )
        
        logger.info(f"✅ Behavior cloning model trained")
        logger.info(f"Training time: {training_result.training_time:.2f}s")
        logger.info(f"Validation MSE: {training_result.validation_metrics.get('mse', 'N/A')}")
        
        return training_result
    
    except Exception as e:
        logger.warning(f"⚠️  ML training failed (dependencies may be missing): {e}")
        return None


async def main():
    """Main demonstration function."""
    logger = setup_example()
    
    try:
        # Load sample dataset
        logger.info("📁 Loading sample dataset...")
        dataset_info, episodes = load_sample_dataset()
        
        if dataset_info is None:
            logger.error("❌ Failed to load dataset")
            return
        
        logger.info(f"✅ Dataset loaded: {dataset_info.num_episodes} episodes, {dataset_info.total_frames} frames")
        
        # Demonstrate all features
        logger.info("\n" + "="*60)
        logger.info("🎯 STARTING ADVANCED FEATURES DEMONSTRATION")
        logger.info("="*60)
        
        # 1. Data Validation
        validation_result = demonstrate_validation(dataset_info, episodes)
        
        # 2. Data Analytics
        analytics_result = demonstrate_analytics(dataset_info, episodes)
        
        # 3. Data Preprocessing
        processed_episodes = demonstrate_preprocessing(episodes)
        
        # 4. Streaming Processing
        chunks = await demonstrate_streaming(episodes)
        
        # 5. Automation System
        automation_results = await demonstrate_automation(episodes)
        
        # 6. Visualization and Reporting
        report = demonstrate_visualization(dataset_info, episodes)
        
        # 7. ML Integration
        ml_result = demonstrate_ml_integration(episodes)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMONSTRATION COMPLETE")
        logger.info("="*60)
        
        logger.info("📊 Summary:")
        logger.info(f"  - Validation: {'✅ Passed' if validation_result.success else '❌ Issues found'}")
        logger.info(f"  - Analytics: {len(analytics_result.insights)} insights generated")
        logger.info(f"  - Preprocessing: {len(processed_episodes)} episodes processed")
        logger.info(f"  - Streaming: {len(chunks)} chunks processed")
        logger.info(f"  - Automation: {len(automation_results)} actions taken")
        logger.info(f"  - Reporting: {len(report.sections)} sections generated")
        logger.info(f"  - ML: {'✅ Model trained' if ml_result else '⚠️  Skipped'}")
        
        logger.info("\n💡 This example showcases the power and flexibility of Robodata v2.0!")
        logger.info("🚀 Explore each feature in depth for your specific use cases.")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
