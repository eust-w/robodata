"""Quality metric functions for robotics episode data.

All functions operate on numpy arrays and return structured results.
No video/image processing — proprioception only.

References:
    - Hogan & Sternad (2009) — LDLJ smoothness metric
    - Sakr et al. "Consistency Matters" (ACM THRI, 2024) — path length, manipulability
    - Belkhale et al. "DemInf" (2025) — action entropy
    - Liu et al. "SCIZOR" (2025) — static episode detection
    - Kim et al. "OpenVLA" (2024) — dead action filtering
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from robodata.quality.config import QualityConfig
from robodata.quality.models import StaticResult, TimestampResult


# ── Metric 1: Dead Action Detection ─────────────────────────────


def dead_action_detection(
    actions: NDArray, config: QualityConfig
) -> tuple[float, list[tuple[int, int]]]:
    """Detect timesteps where all action dimensions are zero or constant.

    Args:
        actions: Shape (T, D) action array.
        config: Quality configuration.

    Returns:
        Tuple of (dead_fraction, dead_ranges) where dead_ranges is a list
        of (start_idx, end_idx) tuples marking dead segments.
    """
    eps = config.dead_action_eps

    # Zero actions
    is_zero = np.all(np.abs(actions) < eps, axis=1)

    # Constant actions (same as first timestep)
    is_constant = np.all(np.abs(actions - actions[0:1]) < eps, axis=1)

    is_dead = is_zero | is_constant
    dead_fraction = float(np.mean(is_dead))

    # Find contiguous dead ranges
    dead_ranges = _find_runs(is_dead)

    return dead_fraction, dead_ranges


# ── Metric 2: Log Dimensionless Jerk (LDLJ) ────────────────────


def log_dimensionless_jerk(
    positions: NDArray, dt: float
) -> float | None:
    """Compute LDLJ smoothness metric.

    More negative = jerkier (higher jerk relative to velocity).
    Typical range: -6 to -8 (smooth) to -15 to -25 (jerky).

    Args:
        positions: Shape (T, D) position array (joint angles or EE pose).
        dt: Time step in seconds (1/fps).

    Returns:
        LDLJ score, or None if input too short.
    """
    if len(positions) < 4 or dt <= 0:
        return None

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    t_total = (len(positions) - 1) * dt
    peak_vel = float(np.max(np.linalg.norm(velocity, axis=1)))

    if peak_vel < 1e-10 or t_total < 1e-10:
        return None

    jerk_magnitude_sq = float(np.sum(jerk**2) * dt)

    ldlj_arg = t_total**3 / peak_vel**2 * jerk_magnitude_sq
    ldlj = -np.log(max(ldlj_arg, 1e-10))

    return float(ldlj)


# ── Metric 3: Gripper Chatter Detection ─────────────────────────


def gripper_chatter(
    actions: NDArray, duration: float, config: QualityConfig
) -> tuple[int, float, bool]:
    """Count rapid gripper open/close transitions.

    Args:
        actions: Shape (T, D) action array.
        duration: Episode duration in seconds.
        config: Quality configuration.

    Returns:
        Tuple of (chatter_count, chatter_rate, is_chattery).
    """
    gripper_dim = config.gripper_dim
    gripper = actions[:, gripper_dim]

    # Auto-detect binarization threshold: midpoint of observed range
    g_min, g_max = float(np.min(gripper)), float(np.max(gripper))
    if g_max - g_min < 1e-6:
        # Gripper never moves
        return 0, 0.0, False

    threshold = (g_min + g_max) / 2.0
    binary = (gripper > threshold).astype(np.int32)
    transitions = int(np.sum(np.abs(np.diff(binary))))

    chatter_rate = transitions / max(duration, 1e-6)
    is_chattery = chatter_rate > config.chatter_threshold

    return transitions, chatter_rate, is_chattery


# ── Metric 4: Path Length ────────────────────────────────────────


def path_length(positions: NDArray) -> float:
    """Sum of step-to-step distances in joint or Cartesian space.

    Args:
        positions: Shape (T, D) position array.

    Returns:
        Total path length.
    """
    diffs = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(step_distances))


# ── Metric 5: Timestamp Regularity ──────────────────────────────


def timestamp_regularity(
    timestamps: NDArray, expected_fps: float
) -> TimestampResult:
    """Analyze timestamp regularity and detect dropped frames.

    Args:
        timestamps: Shape (T,) monotonic timestamp array in seconds.
        expected_fps: Expected recording frequency.

    Returns:
        TimestampResult with regularity statistics.
    """
    dt = np.diff(timestamps)
    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))

    jitter_ratio = dt_std / max(dt_mean, 1e-10)

    expected_dt = 1.0 / expected_fps if expected_fps > 0 else dt_mean
    gap_mask = dt > 2.0 * expected_dt
    gap_locations = np.where(gap_mask)[0].tolist()

    effective_fps = 1.0 / dt_mean if dt_mean > 0 else 0.0

    return TimestampResult(
        dt_mean=dt_mean,
        dt_std=dt_std,
        jitter_ratio=jitter_ratio,
        num_gaps=len(gap_locations),
        gap_locations=gap_locations,
        effective_fps=effective_fps,
    )


# ── Metric 6: Action Range Saturation ───────────────────────────


def action_saturation(
    actions: NDArray, config: QualityConfig
) -> tuple[NDArray, float, list[int]]:
    """Detect timesteps where actions hit their min/max bounds.

    Args:
        actions: Shape (T, D) action array.
        config: Quality configuration.

    Returns:
        Tuple of (saturation_per_dim, overall_saturation, saturated_dims).
    """
    if config.action_bounds is not None:
        a_min = np.full(actions.shape[1], config.action_bounds[0])
        a_max = np.full(actions.shape[1], config.action_bounds[1])
    else:
        a_min = np.min(actions, axis=0)
        a_max = np.max(actions, axis=0)

    range_per_dim = a_max - a_min
    # Skip dimensions with no range
    valid = range_per_dim > 1e-10
    margin = config.saturation_margin * range_per_dim

    at_min = actions <= (a_min + margin)
    at_max = actions >= (a_max - margin)
    saturated = at_min | at_max

    # Zero out invalid dims
    saturated[:, ~valid] = False

    sat_per_dim = np.mean(saturated, axis=0).astype(np.float64)
    # Mean across dims — one saturated gripper dim shouldn't dominate
    overall = float(np.mean(sat_per_dim[valid])) if np.any(valid) else 0.0
    flagged_dims = np.where(sat_per_dim > config.saturation_dim_flag)[0].tolist()

    return sat_per_dim, overall, flagged_dims


# ── Metric 7: Static Episode Detection ──────────────────────────


def static_detection(
    actions: NDArray, fps: float, config: QualityConfig
) -> StaticResult:
    """Detect periods where the robot is not moving.

    Args:
        actions: Shape (T, D) action array.
        fps: Recording frequency (for duration calculation).
        config: Quality configuration.

    Returns:
        StaticResult with static segments and statistics.
    """
    action_norms = np.linalg.norm(actions, axis=1)

    pct_value = np.percentile(action_norms, config.static_threshold_percentile)
    threshold = 0.01 * pct_value if pct_value > 1e-10 else 1e-6

    is_static = action_norms < threshold
    static_fraction = float(np.mean(is_static))

    # Find contiguous static runs
    runs = _find_runs(is_static)
    longest = max((end - start for start, end in runs), default=0)

    # Convert to segments with duration
    dt = 1.0 / fps if fps > 0 else 1.0
    segments = [(s, e, (e - s) * dt) for s, e in runs]

    return StaticResult(
        static_fraction=static_fraction,
        is_mostly_static=static_fraction > config.static_fraction_flag,
        longest_static_run=longest,
        static_segments=segments,
    )


# ── Metric 8: Action Distribution Entropy ───────────────────────


def action_entropy(actions: NDArray) -> tuple[NDArray, float]:
    """Compute Shannon entropy of action distribution per dimension.

    Low entropy = repetitive actions. Very high = random.

    Args:
        actions: Shape (T, D) action array.

    Returns:
        Tuple of (entropy_per_dim, mean_entropy).
    """
    t, d = actions.shape
    entropy_per_dim = np.zeros(d)

    for dim in range(d):
        values = actions[:, dim]
        val_range = np.ptp(values)

        if val_range < 1e-10:
            entropy_per_dim[dim] = 0.0
            continue

        # Sturges' rule for bin count
        n_bins = max(int(np.ceil(np.log2(t) + 1)), 2)
        counts, _ = np.histogram(values, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        # Normalized entropy (0-1 range)
        max_entropy = np.log2(n_bins)
        raw_entropy = float(-np.sum(probs * np.log2(probs)))
        entropy_per_dim[dim] = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    mean_entropy = float(np.mean(entropy_per_dim))
    return entropy_per_dim, mean_entropy


# ── Metric 10: Composite Quality Score ──────────────────────────


def composite_score(
    dead_fraction: float | None,
    ldlj_score: float | None,
    is_chattery: bool | None,
    chatter_rate: float | None,
    static_fraction: float | None,
    jitter_ratio: float | None,
    overall_saturation: float | None,
    mean_entropy: float | None,
    config: QualityConfig | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute weighted composite quality score.

    All thresholds, weights, and scoring parameters are read from config.

    Args:
        Individual metric results (None = not computed, skip).
        config: Quality configuration with all thresholds.

    Returns:
        Tuple of (overall_score 0-10, subscores dict, flags list).
    """
    if config is None:
        config = QualityConfig()

    subscores: dict[str, float] = {}
    flags: list[str] = []
    weights: dict[str, float] = {}

    # Dead actions
    if dead_fraction is not None:
        subscores["dead_actions"] = 1.0 - dead_fraction
        weights["dead_actions"] = config.weight_dead_actions
        if dead_fraction > config.dead_fraction_flag:
            flags.append("dead_actions")

    # Smoothness
    if ldlj_score is not None:
        subscores["smoothness"] = 1.0 / (
            1.0 + np.exp(-(ldlj_score - config.ldlj_sigmoid_center) / config.ldlj_sigmoid_scale)
        )
        weights["smoothness"] = config.weight_smoothness
        if ldlj_score < config.ldlj_flag:
            flags.append("jerky")

    # Gripper health
    if is_chattery is not None:
        if is_chattery and chatter_rate is not None:
            subscores["gripper_health"] = max(0.0, 1.0 - chatter_rate / 5.0)
        else:
            subscores["gripper_health"] = 1.0
        weights["gripper_health"] = config.weight_gripper_health
        if is_chattery:
            flags.append("gripper_chatter")

    # Timestamps
    if jitter_ratio is not None:
        if jitter_ratio < config.jitter_warn_ratio:
            subscores["timestamp_regularity"] = 1.0
        else:
            subscores["timestamp_regularity"] = max(0.0, 1.0 - jitter_ratio)
        weights["timestamp_regularity"] = config.weight_timestamp_regularity
        if jitter_ratio > config.jitter_warn_ratio:
            flags.append("timestamp_jitter")

    # Saturation
    if overall_saturation is not None:
        subscores["action_saturation"] = 1.0 - overall_saturation
        weights["action_saturation"] = config.weight_action_saturation
        if overall_saturation > config.saturation_flag:
            flags.append("saturated")

    # Static
    if static_fraction is not None:
        subscores["static_detection"] = 1.0 - static_fraction
        weights["static_detection"] = config.weight_static_detection
        if static_fraction > config.static_fraction_flag:
            flags.append("mostly_static")

    # Entropy (bell curve: penalize both too low and too high)
    if mean_entropy is not None:
        deviation = abs(mean_entropy - config.entropy_expected) / config.entropy_width
        subscores["action_diversity"] = max(0.0, 1.0 - deviation)
        weights["action_diversity"] = config.weight_action_diversity
        if mean_entropy < config.entropy_low_flag:
            flags.append("low_entropy")
        elif mean_entropy > config.entropy_high_flag:
            flags.append("high_entropy")

    # Weighted average
    if weights:
        total_weight = sum(weights.values())
        score = sum(subscores[k] * weights[k] for k in weights) / total_weight * 10.0
    else:
        score = 0.0

    return float(score), subscores, flags


# ── Helpers ──────────────────────────────────────────────────────


def _find_runs(mask: NDArray) -> list[tuple[int, int]]:
    """Find contiguous True runs in a boolean array.

    Returns list of (start_idx, end_idx) tuples (end exclusive).
    """
    if len(mask) == 0:
        return []

    runs: list[tuple[int, int]] = []
    padded = np.concatenate(([0], mask.astype(np.int32), [0]))
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        runs.append((int(s), int(e)))

    return runs
