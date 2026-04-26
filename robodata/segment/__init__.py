"""Episode segmentation via PELT changepoint detection.

Detects phase transitions in proprioception signals to split episodes
into meaningful segments (sub-skills, regime changes).

Usage::

    from robodata.segment import SegmentAnalyzer

    analyzer = SegmentAnalyzer(penalty="bic", cost_model="rbf")
    report = analyzer.segment_dataset("./bridge_v2")
    report.to_json("segments.json")
"""

from robodata.segment.analyzer import SegmentAnalyzer
from robodata.segment.config import SegmentConfig
from robodata.segment.labeler import PhaseLabeler
from robodata.segment.models import EpisodeSegmentation, SegmentationReport

__all__ = [
    "SegmentAnalyzer",
    "SegmentConfig",
    "EpisodeSegmentation",
    "PhaseLabeler",
    "SegmentationReport",
]
