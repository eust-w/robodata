"""Advanced data analytics and insights system for Robodata.

Provides comprehensive analysis of robotics datasets including
statistical analysis, pattern detection, and quality metrics.
"""

from __future__ import annotations

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame


class AnalysisType(Enum):
    """Types of analysis available."""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COVERAGE = "coverage"


class InsightLevel(Enum):
    """Levels of insight significance."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataPoint:
    """Single data point for analysis."""
    timestamp: Optional[float]
    value: Union[float, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """Single insight from analysis."""
    title: str
    description: str
    level: InsightLevel
    analysis_type: AnalysisType
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    visualization_type: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of data analysis."""
    insights: List[Insight] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def insights_by_level(self) -> Dict[InsightLevel, List[Insight]]:
        """Group insights by level."""
        result = {level: [] for level in InsightLevel}
        for insight in self.insights:
            result[insight.level].append(insight)
        return result
    
    @property
    def critical_insights(self) -> List[Insight]:
        """Get critical insights."""
        return self.insights_by_level[InsightLevel.CRITICAL]
    
    @property
    def high_insights(self) -> List[Insight]:
        """Get high priority insights."""
        return self.insights_by_level[InsightLevel.HIGH]


class StatisticalAnalyzer:
    """Statistical analysis of robotics data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_actions(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze action statistics."""
        all_actions = []
        episode_lengths = []
        
        for episode in episodes:
            episode_actions = []
            for frame in episode.load_frames():
                if frame.action is not None:
                    action_array = np.array(frame.action).flatten()
                    episode_actions.append(action_array)
                    all_actions.append(action_array)
            
            episode_lengths.append(len(episode_actions))
        
        if not all_actions:
            return {"error": "No action data found"}
        
        actions_matrix = np.array(all_actions)
        
        analysis = {
            "total_actions": len(all_actions),
            "action_dimension": actions_matrix.shape[1],
            "episode_stats": {
                "mean_length": np.mean(episode_lengths),
                "std_length": np.std(episode_lengths),
                "min_length": np.min(episode_lengths),
                "max_length": np.max(episode_lengths),
            },
            "action_stats": {},
            "correlations": {},
            "outliers": {},
        }
        
        # Per-dimension statistics
        for i in range(actions_matrix.shape[1]):
            dim_actions = actions_matrix[:, i]
            analysis["action_stats"][f"dim_{i}"] = {
                "mean": np.mean(dim_actions),
                "std": np.std(dim_actions),
                "min": np.min(dim_actions),
                "max": np.max(dim_actions),
                "range": np.max(dim_actions) - np.min(dim_actions),
                "variance": np.var(dim_actions),
                "skewness": stats.skew(dim_actions),
                "kurtosis": stats.kurtosis(dim_actions),
            }
        
        # Correlation analysis
        if actions_matrix.shape[1] > 1:
            corr_matrix = np.corrcoef(actions_matrix.T)
            analysis["correlations"] = {
                "matrix": corr_matrix.tolist(),
                "high_correlations": self._find_high_correlations(corr_matrix),
            }
        
        # Outlier detection
        analysis["outliers"] = self._detect_outliers(actions_matrix)
        
        return analysis
    
    def analyze_states(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze state statistics."""
        all_states = []
        
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.state is not None:
                    state_array = np.array(frame.state).flatten()
                    all_states.append(state_array)
        
        if not all_states:
            return {"error": "No state data found"}
        
        states_matrix = np.array(all_states)
        
        analysis = {
            "total_states": len(all_states),
            "state_dimension": states_matrix.shape[1],
            "state_stats": {},
            "coverage": self._analyze_state_coverage(states_matrix),
            "transitions": self._analyze_state_transitions(states_matrix),
        }
        
        # Per-dimension statistics
        for i in range(states_matrix.shape[1]):
            dim_states = states_matrix[:, i]
            analysis["state_stats"][f"dim_{i}"] = {
                "mean": np.mean(dim_states),
                "std": np.std(dim_states),
                "min": np.min(dim_states),
                "max": np.max(dim_states),
                "range": np.max(dim_states) - np.min(dim_states),
                "variance": np.var(dim_states),
                "unique_values": len(np.unique(dim_states)),
            }
        
        return analysis
    
    def _find_high_correlations(self, corr_matrix: np.ndarray, threshold: float = 0.8) -> List[Dict]:
        """Find pairs of highly correlated dimensions."""
        high_corr = []
        n = corr_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr.append({
                        "dim1": i,
                        "dim2": j,
                        "correlation": corr_matrix[i, j],
                    })
        
        return high_corr
    
    def _detect_outliers(self, data: np.ndarray, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers in the data."""
        outliers = {}
        
        if method == "iqr":
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outliers["method"] = "iqr"
            outliers["outlier_count"] = np.sum(outlier_mask, axis=0)
            outliers["outlier_percentage"] = (outliers["outlier_count"] / len(data)) * 100
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data, axis=0))
            outlier_mask = z_scores > 3
            outliers["method"] = "zscore"
            outliers["outlier_count"] = np.sum(outlier_mask, axis=0)
            outliers["outlier_percentage"] = (outliers["outlier_count"] / len(data)) * 100
        
        return outliers
    
    def _analyze_state_coverage(self, states: np.ndarray) -> Dict[str, Any]:
        """Analyze state space coverage."""
        coverage = {
            "bounding_box": {
                "min": np.min(states, axis=0).tolist(),
                "max": np.max(states, axis=0).tolist(),
            },
            "volume": np.prod(np.max(states, axis=0) - np.min(states, axis=0)),
        }
        
        # Discretize and analyze coverage density
        n_bins = min(10, len(states) // 10)
        if n_bins > 1:
            for i in range(states.shape[1]):
                hist, _ = np.histogram(states[:, i], bins=n_bins)
                coverage[f"dim_{i}_coverage"] = {
                    "bins_visited": np.sum(hist > 0),
                    "total_bins": n_bins,
                    "coverage_ratio": np.sum(hist > 0) / n_bins,
                }
        
        return coverage
    
    def _analyze_state_transitions(self, states: np.ndarray) -> Dict[str, Any]:
        """Analyze state transitions."""
        if len(states) < 2:
            return {"error": "Insufficient data for transition analysis"}
        
        # Compute state differences
        transitions = np.diff(states, axis=0)
        
        analysis = {
            "mean_transition_magnitude": np.mean(np.linalg.norm(transitions, axis=1)),
            "max_transition_magnitude": np.max(np.linalg.norm(transitions, axis=1)),
            "transition_stats": {},
        }
        
        # Per-dimension transition statistics
        for i in range(transitions.shape[1]):
            dim_transitions = transitions[:, i]
            analysis["transition_stats"][f"dim_{i}"] = {
                "mean_change": np.mean(dim_transitions),
                "std_change": np.std(dim_transitions),
                "zero_changes": np.sum(np.abs(dim_transitions) < 1e-6),
                "zero_change_ratio": np.sum(np.abs(dim_transitions) < 1e-6) / len(dim_transitions),
            }
        
        return analysis


class TemporalAnalyzer:
    """Temporal analysis of robotics data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_temporal_patterns(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        analysis = {
            "episode_durations": [],
            "frame_intervals": [],
            "temporal_consistency": {},
            "periodicity": {},
            "trends": {},
        }
        
        all_timestamps = []
        episode_durations = []
        
        for episode in episodes:
            frames = list(episode.load_frames())
            timestamps = [f.timestamp for f in frames if f.timestamp is not None]
            
            if len(timestamps) >= 2:
                duration = timestamps[-1] - timestamps[0]
                episode_durations.append(duration)
                all_timestamps.extend(timestamps)
                
                # Frame intervals
                intervals = np.diff(timestamps)
                analysis["frame_intervals"].extend(intervals.tolist())
        
        if episode_durations:
            analysis["episode_durations"] = {
                "mean": np.mean(episode_durations),
                "std": np.std(episode_durations),
                "min": np.min(episode_durations),
                "max": np.max(episode_durations),
                "total_episodes": len(episode_durations),
            }
        
        if analysis["frame_intervals"]:
            intervals = np.array(analysis["frame_intervals"])
            analysis["temporal_consistency"] = {
                "mean_interval": np.mean(intervals),
                "std_interval": np.std(intervals),
                "cv_interval": np.std(intervals) / np.mean(intervals),  # Coefficient of variation
                "irregular_intervals": np.sum(np.abs(intervals - np.mean(intervals)) > 2 * np.std(intervals)),
            }
        
        return analysis
    
    def detect_anomalies(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies."""
        anomalies = []
        
        for episode in episodes:
            frames = list(episode.load_frames())
            timestamps = [f.timestamp for f in frames if f.timestamp is not None]
            
            if len(timestamps) < 3:
                continue
            
            # Detect gaps in timestamps
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Large gaps
            gap_threshold = mean_interval + 3 * std_interval
            for i, interval in enumerate(intervals):
                if interval > gap_threshold:
                    anomalies.append({
                        "type": "temporal_gap",
                        "episode_id": episode.episode_id,
                        "frame_index": i + 1,
                        "gap_size": interval,
                        "expected_size": mean_interval,
                        "severity": "high" if interval > gap_threshold * 2 else "medium",
                    })
            
            # Backward timestamps
            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i-1]:
                    anomalies.append({
                        "type": "backward_timestamp",
                        "episode_id": episode.episode_id,
                        "frame_index": i,
                        "timestamp": timestamps[i],
                        "previous_timestamp": timestamps[i-1],
                        "severity": "critical",
                    })
        
        return anomalies


class QualityAnalyzer:
    """Quality analysis of robotics data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_data_quality(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze overall data quality."""
        quality_analysis = {
            "completeness": self._analyze_completeness(episodes),
            "consistency": self._analyze_consistency(episodes),
            "accuracy": self._analyze_accuracy(episodes),
            "richness": self._analyze_richness(episodes),
            "overall_score": 0.0,
        }
        
        # Calculate overall quality score
        scores = [
            quality_analysis["completeness"]["score"],
            quality_analysis["consistency"]["score"],
            quality_analysis["accuracy"]["score"],
            quality_analysis["richness"]["score"],
        ]
        quality_analysis["overall_score"] = np.mean(scores)
        
        return quality_analysis
    
    def _analyze_completeness(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze data completeness."""
        total_frames = 0
        complete_frames = 0
        missing_data = defaultdict(int)
        
        for episode in episodes:
            for frame in episode.load_frames():
                total_frames += 1
                frame_complete = True
                
                if frame.action is None:
                    missing_data["action"] += 1
                    frame_complete = False
                
                if frame.state is None:
                    missing_data["state"] += 1
                    frame_complete = False
                
                if not frame.images:
                    missing_data["images"] += 1
                    frame_complete = False
                
                if frame_complete:
                    complete_frames += 1
        
        completeness_score = complete_frames / total_frames if total_frames > 0 else 0.0
        
        return {
            "score": completeness_score,
            "total_frames": total_frames,
            "complete_frames": complete_frames,
            "missing_data": dict(missing_data),
            "missing_percentages": {k: (v / total_frames) * 100 for k, v in missing_data.items()},
        }
    
    def _analyze_consistency(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze data consistency."""
        consistency_issues = []
        
        # Check action dimension consistency
        action_dims = set()
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.action is not None:
                    action_dims.add(len(np.array(frame.action).flatten()))
        
        if len(action_dims) > 1:
            consistency_issues.append({
                "type": "inconsistent_action_dimensions",
                "dimensions_found": list(action_dims),
                "severity": "high",
            })
        
        # Check state dimension consistency
        state_dims = set()
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.state is not None:
                    state_dims.add(len(np.array(frame.state).flatten()))
        
        if len(state_dims) > 1:
            consistency_issues.append({
                "type": "inconsistent_state_dimensions",
                "dimensions_found": list(state_dims),
                "severity": "high",
            })
        
        # Check camera consistency
        camera_names = set()
        for episode in episodes:
            for frame in episode.load_frames():
                camera_names.update(frame.images.keys())
        
        # Check if all episodes have the same cameras
        episode_cameras = []
        for episode in episodes:
            ep_cameras = set()
            for frame in episode.load_frames():
                ep_cameras.update(frame.images.keys())
            episode_cameras.append(ep_cameras)
        
        if len(set(tuple(sorted(cams)) for cams in episode_cameras)) > 1:
            consistency_issues.append({
                "type": "inconsistent_cameras",
                "severity": "medium",
            })
        
        consistency_score = 1.0 - len(consistency_issues) * 0.2  # Deduct 20% per issue
        consistency_score = max(0.0, consistency_score)
        
        return {
            "score": consistency_score,
            "issues": consistency_issues,
            "action_dimensions": list(action_dims),
            "state_dimensions": list(state_dims),
            "camera_names": list(camera_names),
        }
    
    def _analyze_accuracy(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze data accuracy (basic checks)."""
        accuracy_issues = []
        
        for episode in episodes:
            for frame_idx, frame in enumerate(episode.load_frames()):
                # Check for NaN values
                if frame.action is not None:
                    action_array = np.array(frame.action)
                    if np.any(np.isnan(action_array)):
                        accuracy_issues.append({
                            "type": "nan_in_action",
                            "episode_id": episode.episode_id,
                            "frame_index": frame_idx,
                            "severity": "high",
                        })
                
                if frame.state is not None:
                    state_array = np.array(frame.state)
                    if np.any(np.isnan(state_array)):
                        accuracy_issues.append({
                            "type": "nan_in_state",
                            "episode_id": episode.episode_id,
                            "frame_index": frame_idx,
                            "severity": "high",
                        })
                
                # Check for infinite values
                if frame.action is not None:
                    action_array = np.array(frame.action)
                    if np.any(np.isinf(action_array)):
                        accuracy_issues.append({
                            "type": "inf_in_action",
                            "episode_id": episode.episode_id,
                            "frame_index": frame_idx,
                            "severity": "high",
                        })
        
        accuracy_score = 1.0 - (len(accuracy_issues) / max(1, len(episodes) * 10))  # Normalize by expected frames
        accuracy_score = max(0.0, accuracy_score)
        
        return {
            "score": accuracy_score,
            "issues": accuracy_issues,
            "total_issues": len(accuracy_issues),
        }
    
    def _analyze_richness(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Analyze data richness and diversity."""
        richness_metrics = {
            "episode_diversity": 0.0,
            "action_diversity": 0.0,
            "state_diversity": 0.0,
            "temporal_diversity": 0.0,
        }
        
        # Episode diversity (different episode lengths)
        episode_lengths = [len(list(ep.load_frames())) for ep in episodes]
        if len(set(episode_lengths)) > 1:
            richness_metrics["episode_diversity"] = np.std(episode_lengths) / np.mean(episode_lengths)
        
        # Action diversity
        all_actions = []
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.action is not None:
                    all_actions.append(np.array(frame.action).flatten())
        
        if all_actions:
            actions_matrix = np.array(all_actions)
            # Use coefficient of variation as diversity measure
            for i in range(actions_matrix.shape[1]):
                cv = np.std(actions_matrix[:, i]) / (np.mean(actions_matrix[:, i]) + 1e-8)
                richness_metrics["action_diversity"] += cv
            richness_metrics["action_diversity"] /= actions_matrix.shape[1]
        
        # State diversity (similar to action)
        all_states = []
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.state is not None:
                    all_states.append(np.array(frame.state).flatten())
        
        if all_states:
            states_matrix = np.array(all_states)
            for i in range(states_matrix.shape[1]):
                cv = np.std(states_matrix[:, i]) / (np.mean(states_matrix[:, i]) + 1e-8)
                richness_metrics["state_diversity"] += cv
            richness_metrics["state_diversity"] /= states_matrix.shape[1]
        
        # Overall richness score
        richness_score = np.mean(list(richness_metrics.values()))
        
        return {
            "score": richness_score,
            "metrics": richness_metrics,
        }


class AnalyticsEngine:
    """Main analytics engine that orchestrates all analyzers."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
    
    def analyze_dataset(
        self,
        dataset_info: DatasetInfo,
        episodes: List[Episode],
        analysis_types: Optional[List[AnalysisType]] = None
    ) -> AnalysisResult:
        """Comprehensive dataset analysis."""
        if analysis_types is None:
            analysis_types = [
                AnalysisType.STATISTICAL,
                AnalysisType.TEMPORAL,
                AnalysisType.QUALITY,
            ]
        
        self.logger.info(f"Starting analysis with {len(analysis_types)} analysis types")
        
        result = AnalysisResult()
        result.metadata["analysis_types"] = [at.value for at in analysis_types]
        result.metadata["total_episodes"] = len(episodes)
        
        # Statistical analysis
        if AnalysisType.STATISTICAL in analysis_types:
            self.logger.info("Running statistical analysis")
            result.statistics["actions"] = self.statistical_analyzer.analyze_actions(episodes)
            result.statistics["states"] = self.statistical_analyzer.analyze_states(episodes)
            
            # Generate insights from statistical analysis
            result.insights.extend(self._generate_statistical_insights(result.statistics))
        
        # Temporal analysis
        if AnalysisType.TEMPORAL in analysis_types:
            self.logger.info("Running temporal analysis")
            result.statistics["temporal"] = self.temporal_analyzer.analyze_temporal_patterns(episodes)
            anomalies = self.temporal_analyzer.detect_anomalies(episodes)
            result.statistics["temporal"]["anomalies"] = anomalies
            
            # Generate insights from temporal analysis
            result.insights.extend(self._generate_temporal_insights(result.statistics["temporal"]))
        
        # Quality analysis
        if AnalysisType.QUALITY in analysis_types:
            self.logger.info("Running quality analysis")
            result.statistics["quality"] = self.quality_analyzer.analyze_data_quality(episodes)
            
            # Generate insights from quality analysis
            result.insights.extend(self._generate_quality_insights(result.statistics["quality"]))
        
        # Sort insights by level and confidence
        result.insights.sort(key=lambda x: (x.level.value, -x.confidence))
        
        self.logger.info(f"Analysis complete: {len(result.insights)} insights generated")
        return result
    
    def _generate_statistical_insights(self, stats: Dict[str, Any]) -> List[Insight]:
        """Generate insights from statistical analysis."""
        insights = []
        
        # Action insights
        if "actions" in stats and "error" not in stats["actions"]:
            action_stats = stats["actions"]
            
            # High variance in actions
            for dim, dim_stats in action_stats["action_stats"].items():
                if dim_stats["variance"] > 1.0:
                    insights.append(Insight(
                        title=f"High variance in action {dim}",
                        description=f"Action dimension {dim} has high variance ({dim_stats['variance']:.2f})",
                        level=InsightLevel.MEDIUM,
                        analysis_type=AnalysisType.STATISTICAL,
                        confidence=0.8,
                        evidence=dim_stats,
                        recommendations=[f"Consider normalizing action {dim}", "Check for outliers in this dimension"],
                    ))
            
            # High correlations
            if "correlations" in action_stats and "high_correlations" in action_stats["correlations"]:
                for corr in action_stats["correlations"]["high_correlations"]:
                    insights.append(Insight(
                        title=f"High correlation between actions",
                        description=f"Actions {corr['dim1']} and {corr['dim2']} are highly correlated ({corr['correlation']:.2f})",
                        level=InsightLevel.MEDIUM,
                        analysis_type=AnalysisType.STATISTICAL,
                        confidence=0.9,
                        evidence=corr,
                        recommendations=["Consider dimensionality reduction", "Check for redundant action dimensions"],
                    ))
            
            # Outliers
            if "outliers" in action_stats:
                outlier_pct = action_stats["outliers"].get("outlier_percentage", [])
                if isinstance(outlier_pct, list) and any(pct > 10 for pct in outlier_pct):
                    insights.append(Insight(
                        title="High number of outliers in actions",
                        description=f"Some action dimensions have >10% outliers",
                        level=InsightLevel.HIGH,
                        analysis_type=AnalysisType.STATISTICAL,
                        confidence=0.8,
                        evidence=action_stats["outliers"],
                        recommendations=["Investigate outlier causes", "Consider robust statistics"],
                    ))
        
        return insights
    
    def _generate_temporal_insights(self, temporal_stats: Dict[str, Any]) -> List[Insight]:
        """Generate insights from temporal analysis."""
        insights = []
        
        # Episode duration insights
        if "episode_durations" in temporal_stats:
            durations = temporal_stats["episode_durations"]
            cv = durations["std_length"] / durations["mean_length"]
            
            if cv > 0.5:
                insights.append(Insight(
                    title="High variability in episode durations",
                    description=f"Episode durations have high variability (CV={cv:.2f})",
                    level=InsightLevel.MEDIUM,
                    analysis_type=AnalysisType.TEMPORAL,
                    confidence=0.8,
                    evidence=durations,
                    recommendations=["Consider standardizing episode lengths", "Investigate causes of duration variability"],
                ))
        
        # Temporal consistency insights
        if "temporal_consistency" in temporal_stats:
            consistency = temporal_stats["temporal_consistency"]
            if consistency["cv_interval"] > 0.3:
                insights.append(Insight(
                    title="Irregular frame intervals",
                    description=f"Frame intervals are irregular (CV={consistency['cv_interval']:.2f})",
                    level=InsightLevel.HIGH,
                    analysis_type=AnalysisType.TEMPORAL,
                    confidence=0.9,
                    evidence=consistency,
                    recommendations=["Check frame rate consistency", "Consider frame interpolation"],
                ))
        
        # Anomaly insights
        if "anomalies" in temporal_stats and temporal_stats["anomalies"]:
            anomaly_count = len(temporal_stats["anomalies"])
            insights.append(Insight(
                title=f"Temporal anomalies detected",
                description=f"Found {anomaly_count} temporal anomalies in the data",
                level=InsightLevel.HIGH if anomaly_count > 10 else InsightLevel.MEDIUM,
                analysis_type=AnalysisType.TEMPORAL,
                confidence=0.9,
                evidence={"anomaly_count": anomaly_count, "anomalies": temporal_stats["anomalies"]},
                recommendations=["Review anomaly causes", "Consider data cleaning"],
            ))
        
        return insights
    
    def _generate_quality_insights(self, quality_stats: Dict[str, Any]) -> List[Insight]:
        """Generate insights from quality analysis."""
        insights = []
        
        overall_score = quality_stats.get("overall_score", 0.0)
        
        if overall_score < 0.7:
            insights.append(Insight(
                title="Low data quality score",
                description=f"Overall data quality score is {overall_score:.2f}",
                level=InsightLevel.CRITICAL if overall_score < 0.5 else InsightLevel.HIGH,
                analysis_type=AnalysisType.QUALITY,
                confidence=0.9,
                evidence=quality_stats,
                recommendations=["Improve data collection processes", "Implement data validation"],
            ))
        
        # Completeness insights
        completeness = quality_stats.get("completeness", {})
        if completeness.get("score", 1.0) < 0.9:
            insights.append(Insight(
                title="Data completeness issues",
                description=f"Data completeness is {completeness.get('score', 0):.2f}",
                level=InsightLevel.HIGH,
                analysis_type=AnalysisType.QUALITY,
                confidence=0.8,
                evidence=completeness,
                recommendations=["Address missing data", "Improve data collection coverage"],
            ))
        
        # Consistency insights
        consistency = quality_stats.get("consistency", {})
        if consistency.get("issues"):
            insights.append(Insight(
                title="Data consistency issues",
                description=f"Found {len(consistency['issues'])} consistency issues",
                level=InsightLevel.HIGH,
                analysis_type=AnalysisType.QUALITY,
                confidence=0.9,
                evidence=consistency,
                recommendations=["Standardize data formats", "Validate data consistency"],
            ))
        
        return insights


# Convenience functions
def analyze_dataset(
    dataset_info: DatasetInfo,
    episodes: List[Episode],
    analysis_types: Optional[List[AnalysisType]] = None
) -> AnalysisResult:
    """Analyze a dataset comprehensively."""
    engine = AnalyticsEngine()
    return engine.analyze_dataset(dataset_info, episodes, analysis_types)


def get_quality_score(episodes: List[Episode]) -> float:
    """Get a quick quality score for a dataset."""
    analyzer = QualityAnalyzer()
    quality_analysis = analyzer.analyze_data_quality(episodes)
    return quality_analysis["overall_score"]
