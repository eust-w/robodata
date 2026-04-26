"""Semantic phase labeling for segments using proprioception signals.

Classifies each segment into a manipulation phase (idle, reaching, grasping,
transporting, placing, retracting, fine_manipulation) using adaptive
percentile-based thresholds on velocity and gripper signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from robodata.segment.models import Segment

logger = logging.getLogger(__name__)

# Phase labels
IDLE = "idle"
REACHING = "reaching"
GRASPING = "grasping"
TRANSPORTING = "transporting"
PLACING = "placing"
RETRACTING = "retracting"
FINE_MANIPULATION = "fine_manipulation"
MOVING = "moving"  # Fallback when no gripper info
UNKNOWN = "unknown"


@dataclass
class _Thresholds:
    """Adaptive thresholds computed from episode-level statistics."""

    vel_low: float  # Below this = idle/fine manipulation
    vel_high: float  # Above this = fast motion
    gripper_close_thresh: float = 0.5  # Below = closed


def _compute_velocity(signal: np.ndarray) -> np.ndarray:
    """Compute per-frame velocity magnitude from position signal.

    Args:
        signal: Shape (T, D) position signal.

    Returns:
        Shape (T,) velocity magnitude (first frame = 0).
    """
    diff = np.diff(signal, axis=0)
    magnitudes = np.linalg.norm(diff, axis=1)
    return np.concatenate([[0.0], magnitudes])


def _compute_thresholds(velocity: np.ndarray) -> _Thresholds:
    """Compute adaptive thresholds from velocity distribution.

    Uses percentiles so thresholds self-calibrate to different robots
    and signal scales.
    """
    nonzero_vel = velocity[velocity > 1e-8]
    if len(nonzero_vel) < 5:
        return _Thresholds(vel_low=1e-6, vel_high=1e-3)

    return _Thresholds(
        vel_low=float(np.percentile(nonzero_vel, 20)),
        vel_high=float(np.percentile(nonzero_vel, 70)),
    )


def _segment_mean_velocity(velocity: np.ndarray, seg: Segment) -> float:
    """Mean velocity within a segment's frame range."""
    return float(np.mean(velocity[seg.start : seg.end]))


def _segment_velocity_std(velocity: np.ndarray, seg: Segment) -> float:
    """Velocity standard deviation within a segment."""
    return float(np.std(velocity[seg.start : seg.end]))


def _gripper_delta(gripper: np.ndarray, seg: Segment) -> float:
    """Net gripper change across a segment (end - start)."""
    g = gripper[seg.start : seg.end]
    if len(g) < 2:
        return 0.0
    return float(g[-1] - g[0])


def _mean_gripper(gripper: np.ndarray, seg: Segment) -> float:
    """Mean gripper value within a segment."""
    return float(np.mean(gripper[seg.start : seg.end]))


class PhaseLabeler:
    """Labels segments with semantic phase names using proprioception.

    Decision logic (with gripper):
        1. Low velocity + small gripper delta → idle
        2. Gripper closing significantly → grasping
        3. Gripper opening significantly → placing
        4. High velocity + gripper closed → transporting
        5. High velocity + gripper open → reaching
        6. Low-medium velocity + low std → fine_manipulation
        7. Otherwise → unknown

    Contextual pass:
        - A fast-moving segment after placing → retracting

    Without gripper:
        - Low velocity → idle
        - High velocity → moving
        - Low-medium velocity + low std → fine_manipulation
        - Otherwise → unknown
    """

    # Gripper delta threshold for detecting open/close events
    GRIPPER_DELTA_THRESH = 0.15

    def label_segments(
        self,
        segments: list[Segment],
        signal: np.ndarray,
        gripper: np.ndarray | None = None,
    ) -> list[Segment]:
        """Label each segment with a semantic phase.

        Args:
            segments: List of Segment objects (start/end already set).
            signal: Shape (T, D) position signal used for segmentation.
            gripper: Optional shape (T,) gripper state array in [0, 1].

        Returns:
            The same segment list with label fields populated.
        """
        if not segments:
            return segments

        velocity = _compute_velocity(signal)
        thresholds = _compute_thresholds(velocity)

        has_gripper = gripper is not None and len(gripper) > 0

        for seg in segments:
            mean_vel = _segment_mean_velocity(velocity, seg)
            vel_std = _segment_velocity_std(velocity, seg)

            if has_gripper:
                seg.label = self._classify_with_gripper(
                    seg, mean_vel, vel_std, thresholds, gripper  # type: ignore[arg-type]
                )
            else:
                seg.label = self._classify_without_gripper(
                    mean_vel, vel_std, thresholds
                )

        # Contextual pass: detect retracting
        if has_gripper:
            self._contextual_pass(segments, velocity, thresholds)

        return segments

    def _classify_with_gripper(
        self,
        seg: Segment,
        mean_vel: float,
        vel_std: float,
        thresh: _Thresholds,
        gripper: np.ndarray,
    ) -> str:
        g_delta = _gripper_delta(gripper, seg)
        g_mean = _mean_gripper(gripper, seg)

        # Idle: barely moving, gripper not changing
        if mean_vel < thresh.vel_low and abs(g_delta) < self.GRIPPER_DELTA_THRESH:
            return IDLE

        # Grasping: gripper closing (delta < -threshold)
        if g_delta < -self.GRIPPER_DELTA_THRESH:
            return GRASPING

        # Placing: gripper opening (delta > threshold)
        if g_delta > self.GRIPPER_DELTA_THRESH:
            return PLACING

        # High velocity: reaching or transporting depending on gripper
        if mean_vel >= thresh.vel_high:
            if g_mean < thresh.gripper_close_thresh:
                return TRANSPORTING
            return REACHING

        # Medium velocity with gripper closed → transporting
        if mean_vel > thresh.vel_low and g_mean < thresh.gripper_close_thresh:
            return TRANSPORTING

        # Medium velocity with gripper open → reaching
        if mean_vel > thresh.vel_low:
            return REACHING

        # Fine manipulation: low-medium velocity, low variance
        if vel_std < thresh.vel_low:
            return FINE_MANIPULATION

        return UNKNOWN

    def _classify_without_gripper(
        self,
        mean_vel: float,
        vel_std: float,
        thresh: _Thresholds,
    ) -> str:
        if mean_vel < thresh.vel_low:
            return IDLE

        if vel_std < thresh.vel_low:
            return FINE_MANIPULATION

        # Any meaningful motion without gripper context
        return MOVING

    def _contextual_pass(
        self,
        segments: list[Segment],
        velocity: np.ndarray,
        thresholds: _Thresholds,
    ) -> None:
        """Second pass: re-label based on surrounding context.

        Retracting = fast motion right after a placing segment.
        """
        for i, seg in enumerate(segments):
            if i == 0:
                continue
            prev = segments[i - 1]
            mean_vel = _segment_mean_velocity(velocity, seg)

            # Fast segment following a place → retract
            if prev.label == PLACING and mean_vel >= thresholds.vel_high and seg.label in (REACHING, MOVING, UNKNOWN):
                seg.label = RETRACTING


def extract_gripper_signal(frames) -> np.ndarray | None:
    """Extract gripper state from episode frames.

    Uses frame.gripper_state or frame.action_gripper fields.
    Does NOT guess from the last dimension of state/action vectors.

    Args:
        frames: List of Frame objects.

    Returns:
        Shape (T,) gripper array, or None if no gripper data found.
    """
    gripper_vals: list[float] = []

    for frame in frames:
        g = frame.gripper_state
        if g is None:
            g = frame.action_gripper
        if g is None:
            return None  # If any frame lacks gripper, bail out entirely
        gripper_vals.append(float(g))

    if not gripper_vals:
        return None

    return np.array(gripper_vals, dtype=np.float64)
