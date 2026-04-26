# Robodata Segment

Episode segmentation via PELT changepoint detection on proprioception signals. Splits episodes into contiguous phases (sub-skills, regime changes, idle periods) without video or image processing.

## Usage

### CLI

```bash
# Basic segmentation
robodata segment ./my_dataset/

# HuggingFace dataset
robodata segment hf://lerobot/aloha_sim_cube

# Export JSON report and timeline visualization
robodata segment ./my_dataset --export segments.json --plot timeline.png

# Choose signal and tune PELT parameters
robodata segment ./my_dataset --signal action --penalty 5.0 --cost-model l2

# Sample a subset of episodes
robodata segment ./my_dataset --sample 20

# Disable per-dimension normalization
robodata segment ./my_dataset --no-normalize

# Use AIC penalty instead of BIC
robodata segment ./my_dataset --penalty aic --min-segment-length 15
```

### Python API

```python
from robodata.segment import SegmentAnalyzer, SegmentConfig

# Analyze a full dataset
analyzer = SegmentAnalyzer(penalty="bic", cost_model="rbf")
report = analyzer.segment_dataset("./my_dataset")
print(report.summary)                    # {'mean_segments': 3.2, ...}
report.to_json("segments.json")

# Segment a single episode from numpy arrays
import numpy as np
signal = np.random.randn(200, 7)         # (T, D) proprioception
result = analyzer.segment_episode_arrays(
    episode_id="ep_0",
    signal=signal,
    signal_name="observation.state",
    fps=30.0,
)
print(result.changepoints)               # [52, 131]
print(result.num_segments)               # 3
for seg in result.segments:
    print(f"  [{seg.start}:{seg.end}] {seg.duration_frames} frames")
```

## How It Works

### PELT (Pruned Exact Linear Time)

PELT is an exact changepoint detection algorithm that finds points where the statistical properties of a signal change abruptly. It minimizes:

```
sum(cost(segment_i)) + n_changepoints * penalty
```

- **Cost function** (`--cost-model`): Measures homogeneity within a segment. `rbf` (radial basis function) is the default — robust to scale differences across joint dimensions. Other options: `l2`, `l1`, `normal`, `ar`.
- **Penalty** (`--penalty`): Controls the number of changepoints. Higher penalty = fewer segments. `bic` (Bayesian Information Criterion) adapts to signal length and dimensionality. `aic` is less conservative. You can also pass a numeric value directly.

Reference: Killick, Fearnhead & Eckley, "Optimal Detection of Changepoints with a Linear Computational Cost", JASA (2012).

### Signal Selection

The `--signal` flag determines which proprioception field to segment on:

| `--signal` value | Frame field | Fallback |
|---|---|---|
| `observation.state` (default) | `frame.state` | `frame.joint_positions` |
| `qpos` / `joint_positions` | `frame.joint_positions` | — |
| `joint_velocities` | `frame.joint_velocities` | — |
| `action` | `frame.action` | — |

Currently limited to proprioception signals. Vision-based segmentation (e.g., on image embeddings) is not yet supported.

### Normalization

With `--normalize` (the default), each dimension is z-score normalized before PELT:

```
signal_normalized = (signal - mean) / (std + 1e-8)
```

This prevents high-magnitude dimensions (e.g., shoulder joint +/-3 rad) from dominating over low-magnitude ones (e.g., gripper 0-1). Disable with `--no-normalize` if your signal dimensions are already on a comparable scale.

### Penalty Selection Guide

| Penalty | Formula | When to use |
|---|---|---|
| `bic` | `ln(T) * D` | Default. Adapts to signal length and dimensionality. |
| `aic` | `2 * D` | Less conservative — detects more changepoints. |
| Numeric (e.g., `5.0`) | Used directly | Full manual control. Start low, increase to reduce segments. |

Where `T` = number of frames, `D` = signal dimensionality.

## Output Format

### JSON Report

```json
{
  "dataset_path": "./my_dataset",
  "num_episodes": 100,
  "config": {
    "signal": "observation.state",
    "penalty": "bic",
    "cost_model": "rbf",
    "min_segment_length": 10,
    "normalize": true
  },
  "computed_at": "2026-03-08T...",
  "summary": {
    "mean_segments": 3.2,
    "median_segments": 3,
    "min_segments": 1,
    "max_segments": 8,
    "total_changepoints": 220
  },
  "per_episode": [
    {
      "episode_id": "ep_0",
      "num_frames": 200,
      "signal_name": "observation.state",
      "signal_dim": 7,
      "changepoints": [45, 120],
      "segments": [
        {"start": 0, "end": 45, "duration_frames": 45, "duration_seconds": 1.5},
        {"start": 45, "end": 120, "duration_frames": 75, "duration_seconds": 2.5},
        {"start": 120, "end": 200, "duration_frames": 80, "duration_seconds": 2.67}
      ],
      "num_segments": 3,
      "fps": 30.0
    }
  ]
}
```

### Timeline PNG (`--plot`)

A horizontal timeline with one row per episode. Segments are colored rectangles, changepoints are vertical black lines. Useful for visually comparing segmentation consistency across episodes.

## Module Structure

```
robodata/segment/
    __init__.py       Public API exports
    config.py         SegmentConfig dataclass (signal, penalty, cost model, normalization)
    analyzer.py       SegmentAnalyzer — signal extraction, PELT execution, dataset orchestration
    models.py         Segment, EpisodeSegmentation, SegmentationReport (with JSON I/O)
    plot.py           Timeline visualization (matplotlib, optional dependency)
```

## Dependencies

- **Required**: `ruptures>=1.1.0` — install with `pip install robodata[segment]`
- **Optional for `--plot`**: `matplotlib>=3.7.0` — install with `pip install robodata[visualize]`
