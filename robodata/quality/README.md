# Robodata Quality Metrics

Episode-level quality scoring for robotics demonstration data using proprioception signals only (joint states, actions, timestamps). No video or image processing.

## Usage

### CLI

```bash
# Local dataset
robodata quality ./my_dataset/

# HuggingFace dataset
robodata quality hf://lerobot/aloha_sim_cube

# With options
robodata quality ./my_dataset --gripper-dim 6 --fps 30 --export report.json
robodata quality ./my_dataset --export-flagged flagged.json
robodata quality ./my_dataset --sample 50 --quick

# Known action bounds (tighter saturation detection)
robodata quality ./my_dataset --action-bounds -1.0,1.0

# Via stats command
robodata stats ./my_dataset --quality
```

### Python API

```python
from robodata.quality import QualityAnalyzer, QualityConfig

# Analyze a full dataset
analyzer = QualityAnalyzer(gripper_dim=-1, fps=30.0)
report = analyzer.analyze_dataset("./my_dataset")
print(report.overall_score)          # 7.8
print(report.flagged_episodes)       # {"jerky": ["ep_42", ...], ...}
report.to_json("quality_report.json")

# Analyze a single episode from numpy arrays
eq = analyzer.analyze_episode_arrays(
    episode_id="ep_0",
    actions=actions_array,     # (T, D)
    states=states_array,       # (T, D)
    timestamps=timestamps,     # (T,)
    fps=30.0,
)
print(eq.overall_score)              # 8.4
print(eq.flags)                      # []
print(eq.is_flagged)                 # False
```

## Metrics

### 1. Dead Action Detection
Detects timesteps where all action dimensions are zero or constant, indicating teleoperation disconnects or recording artifacts.

- **Output**: `dead_fraction` (0-1), `dead_ranges` (contiguous segments)
- **Reference**: Kim et al. "OpenVLA" (2024) — filters degenerate transitions

### 2. Log Dimensionless Jerk (LDLJ)
Gold-standard smoothness metric from motor control. Computes jerk (third derivative of position) normalized by movement duration and peak velocity.

- **Output**: `ldlj_score` (more negative = jerkier; smooth: -6 to -8, jerky: -15 to -25)
- **Scoring**: Sigmoid normalization with configurable center/scale
- **Reference**: Hogan & Sternad (2009); validated by Scano et al., Frontiers in Neurology (2018)

### 3. Gripper Chatter Detection
Counts rapid open/close transitions that indicate noisy teleoperation input.

- **Output**: `chatter_count`, `chatter_rate` (transitions/sec), `is_chattery`
- **Reference**: Sakr et al. "Consistency Matters" (ACM THRI, 2024)

### 4. Path Length
Sum of step-to-step distances in joint space. Useful as a relative metric to compare episodes of the same task (longer = more wandering/hesitation).

- **Output**: `joint_path_length`
- **Note**: Not included in composite score (relative metric, not absolute quality)
- **Reference**: Sakr et al. "Consistency Matters" (2024); Osa et al. (2018)

### 5. Timestamp Regularity
Detects recording issues: dropped frames, irregular timing, frequency drift.

- **Output**: `dt_mean`, `dt_std`, `jitter_ratio`, `num_gaps`, `gap_locations`, `effective_fps`
- **Reference**: General signal integrity; HuggingFace LeRobot video encoding blog (2024)

### 6. Action Range Saturation
Percentage of timesteps where action dimensions hit their min/max bounds, indicating hardware limits or clipping.

- **Output**: `saturation_per_dim`, `overall_saturation`, `saturated_dims`
- **Supports**: Known bounds via `--action-bounds` or auto-inferred from data

### 7. Static Episode Detection
Fraction of episode where the robot is essentially not moving, detecting idle periods.

- **Output**: `static_fraction`, `is_mostly_static`, `longest_static_run`, `static_segments`
- **Reference**: Liu et al. "SCIZOR" (2025) — 15.4% avg improvement after filtering static data

### 8. Action Distribution Entropy
Shannon entropy of action distribution per dimension, normalized to 0-1 range. Measures diversity within an episode.

- **Output**: `entropy_per_dim`, `mean_entropy`
- **Scoring**: Bell curve — penalizes both too low (repetitive) and too high (random)
- **Reference**: Belkhale et al. "DemInf" (2025); Belkhale & Sadigh "Data Quality in Imitation Learning" (NeurIPS, 2023)

### 9. Manipulability Index (Yoshikawa) — Deferred
Distance from singular configurations. Requires a kinematic model (URDF).

- **Status**: Not implemented. Needs robot-specific URDF and a kinematics library (pinocchio/ikpy).
- **Reference**: Yoshikawa (1985); validated in Sakr et al. (2024)

### 10. Composite Quality Score
Weighted combination of metrics 1-8, normalized to a 0-10 scale.

| Metric | Weight | Scoring |
|---|---|---|
| Smoothness (LDLJ) | 0.25 | Sigmoid normalization |
| Dead Actions | 0.15 | 1.0 - dead_fraction |
| Gripper Health | 0.15 | 1.0 if clean, penalized by chatter rate |
| Static Detection | 0.15 | 1.0 - static_fraction |
| Timestamp Regularity | 0.10 | 1.0 if jitter < threshold |
| Action Saturation | 0.10 | 1.0 - overall_saturation |
| Action Diversity | 0.10 | Bell curve around expected entropy |

All weights and thresholds are configurable via `QualityConfig`.

## Module Structure

```
robodata/quality/
    __init__.py       Public API exports
    config.py         QualityConfig dataclass (all thresholds, weights, params)
    metrics.py        Individual metric functions (pure numpy)
    analyzer.py       QualityAnalyzer orchestrator + dataset-level analysis
    models.py         EpisodeQuality, QualityReport, TimestampResult, StaticResult
```

## Adding a New Metric

To add a new quality metric, update these locations:

### 1. Add the metric function (`metrics.py`)

Write a pure function that takes numpy arrays and returns results. Follow the existing pattern:

```python
def my_new_metric(
    actions: NDArray, config: QualityConfig
) -> tuple[float, ...]:
    """One-line description.

    Args:
        actions: Shape (T, D) action array.
        config: Quality configuration.

    Returns:
        Tuple of result values.
    """
    # Implementation using vectorized numpy — no Python loops over timesteps
    ...
```

Requirements:
- **Numpy only** — no torch, tensorflow, or heavy dependencies
- **Return None** if the metric can't be computed (too few frames, missing data)
- **Handle NaN/Inf** gracefully — the analyzer sanitizes inputs, but be defensive
- Cite the paper in the module docstring at the top of `metrics.py`

### 2. Add result fields (`models.py`)

Add fields to `EpisodeQuality` for storing the metric's output:

```python
# Metric N: My New Metric
my_metric_value: float | None = None
```

If the metric returns a complex result, add a new dataclass (like `StaticResult` or `TimestampResult`) and reference it from `EpisodeQuality`. Include relevant fields in `to_dict()` for JSON export.

### 3. Wire into the analyzer (`analyzer.py`)

In `QualityAnalyzer.analyze_episode_arrays()`, call your metric and store the results:

```python
# N. My new metric
eq.my_metric_value = my_new_metric(actions, self.config)
```

Place it in the appropriate section (action-based, state-based, or timestamp-based).

### 4. Add scoring to composite score (`metrics.py`)

In `composite_score()`, add a scoring block that maps your metric's raw value to a 0-1 quality signal:

```python
# My new metric
if my_metric_value is not None:
    subscores["my_metric"] = ...  # 0-1 quality signal
    weights["my_metric"] = config.weight_my_metric
    if my_metric_value > config.my_metric_flag:
        flags.append("my_metric_issue")
```

### 5. Add config fields (`config.py`)

Add any thresholds, weights, and parameters to `QualityConfig`:

```python
# ── My new metric ──
my_metric_threshold: float = 0.5
weight_my_metric: float = 0.10
```

Adjust existing weights so they still sum to 1.0 (or close to it — the composite score normalizes by total weight).

### 6. Add recommendations (`analyzer.py`)

In `_generate_recommendations()`, add a human-readable flag and actionable suggestion:

```python
if "my_metric_issue" in flagged:
    n = len(flagged["my_metric_issue"])
    flags.append(f"{n} episodes with my metric issue")
    recs.append("Suggestion for the user")
```

### 7. Add tests (`tests/test_quality.py`)

Add a test class with at least:
- A test for the expected good case
- A test for the expected bad case
- Edge cases (empty data, constant values, etc.)

```python
class TestMyNewMetric:
    def test_good_case(self):
        ...

    def test_bad_case(self):
        ...
```
