"""Configuration for PELT changepoint segmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentConfig:
    """Configuration for episode segmentation via PELT changepoint detection.

    Attributes:
        signal: Which proprioception signal to segment on.
            Supported: "observation.state", "qpos", "joint_positions",
            "joint_velocities", "action".
        penalty: PELT penalty method. "bic", "aic", or a numeric string.
        cost_model: Cost function for PELT (passed to ruptures.Pelt model=).
            Common: "rbf", "l2", "l1", "normal", "ar".
        min_segment_length: Minimum segment length in frames (ruptures min_size).
        normalize: Whether to z-score normalize per dimension before detection.
        min_frames: Skip episodes shorter than this.
    """

    signal: str = "observation.state"
    penalty: str = "bic"
    cost_model: str = "rbf"
    min_segment_length: int = 10
    normalize: bool = True
    min_frames: int = 5
    label_phases: bool = False
