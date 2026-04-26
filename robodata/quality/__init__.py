"""Quality metrics for robotics episode data.

Compute episode-level quality scores based on proprioception data:
smoothness (LDLJ), dead actions, gripper chatter, static detection,
timestamp regularity, action saturation, and action diversity.

Usage::

    from robodata.quality import QualityAnalyzer, QualityReport

    analyzer = QualityAnalyzer(gripper_dim=-1, fps=30.0)
    report = analyzer.analyze_dataset("./bridge_v2")
    print(report.overall_score)
"""

from robodata.quality.analyzer import QualityAnalyzer
from robodata.quality.config import QualityConfig
from robodata.quality.models import EpisodeQuality, QualityReport

__all__ = [
    "QualityAnalyzer",
    "QualityConfig",
    "QualityReport",
    "EpisodeQuality",
]
