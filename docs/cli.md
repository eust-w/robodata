# CLI 参考

Robodata 命令行工具完整参考。

## robodata inspect

分析数据集结构与 schema。

```bash
# Basic inspection
robodata inspect /path/to/dataset

# Inspect from HuggingFace Hub
robodata inspect hf://lerobot/pusht

# Quick inspect (metadata only, no download)
robodata inspect hf://lerobot/pusht --quick

# Generate a conversion config template
robodata inspect /path/to/dataset --generate-config config.yaml

# Output as JSON
robodata inspect /path/to/dataset --output json

# Deep scan (reads all episodes, slower but more accurate)
robodata inspect /path/to/dataset --deep

# Sample specific number of episodes
robodata inspect /path/to/dataset --samples 10
```

**输出包含：**
- Detected format
- Episode and frame counts
- Camera names and resolutions
- State/action dimensions
- Language instruction samples

---

## robodata convert

数据集格式转换。

```bash
# Basic conversion
robodata convert /path/to/input /path/to/output --format lerobot-v3

# From HuggingFace Hub
robodata convert hf://lerobot/pusht ./output --format lerobot-v3

# With camera name mapping
robodata convert input/ output/ --format lerobot-v3 --camera wrist_cam=img

# Multiple camera mappings
robodata convert input/ output/ --format lerobot-v3 \
    --camera agentview=front \
    --camera eye_in_hand=wrist

# Parallel processing (faster on multi-core systems)
robodata convert input/ output/ --format lerobot-v3 --workers 4

# Ray distributed execution (optional)
# Install: pip install "robodata[ray]"
robodata convert input/ output/ --format lerobot-v3 --executor ray --ray-address auto

# Using a config file
robodata convert input/ output/ --config config.yaml

# Dry run (preview without writing)
robodata convert input/ output/ --format lerobot-v3 --dry-run

# Open visualizer after conversion to compare
robodata convert input/ output/ --format lerobot-v3 --visualize

# Specify source format (if auto-detection fails)
robodata convert input/ output/ --format lerobot-v3 --source-format hdf5
```

**目标格式：**
- `lerobot-v3` - LeRobot v3 (recommended for HuggingFace)
- `rlds` - RLDS/TensorFlow Datasets
- `robodm` - RoboDM .vla format (up to 70x compression)

---

## robodata visualize

交互式数据集可视化。

```bash
# View a dataset
robodata visualize /path/to/dataset

# Compare two datasets side by side
robodata visualize /path/to/original --compare /path/to/converted

# Specify starting episode
robodata visualize /path/to/dataset --episode 5

# Use fast OpenCV backend (better for video playback)
robodata visualize /path/to/dataset --backend opencv
```

**Backends:**
- `matplotlib` (default) - Interactive with sliders, slower playback
- `opencv` - Fast video playback, keyboard controls

**Matplotlib controls:**
- Episode slider: Navigate between episodes
- Frame slider: Navigate frames within an episode
- Play/Pause button: Auto-play frames

**OpenCV controls:**
- `Space`: Play/Pause
- `Left/Right` or `A/D`: Previous/Next frame
- `Up/Down` or `W/S`: Previous/Next episode
- `+/-`: Increase/Decrease playback speed
- `Q` or `Escape`: Quit

---

## robodata quality

Episode-level quality scoring using proprioception data (no video processing).

```bash
# Basic quality report
robodata quality /path/to/dataset

# From HuggingFace Hub
robodata quality hf://lerobot/aloha_sim_cube

# With options
robodata quality /path/to/dataset --gripper-dim 6 --fps 30
robodata quality /path/to/dataset --export report.json
robodata quality /path/to/dataset --export-flagged flagged.json

# Quick mode (skip expensive metrics)
robodata quality /path/to/dataset --quick

# Known action bounds (tighter saturation detection)
robodata quality /path/to/dataset --action-bounds -1.0,1.0

# Sample subset
robodata quality /path/to/dataset --sample 50
```

**Metrics computed:**
- Smoothness (LDLJ) — jerk-based smoothness score
- Dead actions — zero/constant action detection
- Gripper chatter — rapid open/close transitions
- Static detection — idle periods
- Timestamp regularity — dropped frames, jitter
- Action saturation — time at hardware limits
- Action entropy — diversity vs repetitiveness
- Path length — wandering in joint space

**Output:** Overall score (0-10), per-metric subscores, flags, and actionable recommendations.

See [robodata/quality/README.md](../robodata/quality/README.md) for full metric details and paper references.

---

## robodata filter

Filter dataset episodes based on quality scores, flags, or episode IDs. Writes the same format — no conversion.

```bash
# Dry-run: preview what passes/fails
robodata filter /path/to/dataset --min-quality 6.0

# Write filtered dataset
robodata filter /path/to/dataset /path/to/output --min-quality 6.0

# Exclude by flags
robodata filter /path/to/dataset /path/to/output --exclude-flags jerky,mostly_static

# Use pre-computed quality report (faster)
robodata quality /path/to/dataset --export report.json
robodata filter /path/to/dataset /path/to/output --from-report report.json --min-quality 7.0

# Explicit episode lists
robodata filter /path/to/dataset /path/to/output --include-episodes ep_000,ep_001
robodata filter /path/to/dataset /path/to/output --exclude-episodes ep_003,ep_010
```

**Filter criteria:**
- `--min-quality` — Keep episodes scoring >= threshold (0-10)
- `--exclude-flags` — Exclude episodes with matching flags (comma-separated)
- `--include-episodes` / `--exclude-episodes` — Explicit episode selection
- `--from-report` — Skip re-analysis, use existing quality report JSON

**Output:** Summary of kept/excluded episodes with reasons. Dry-run if no output path given.

See [robodata/filter/README.md](../robodata/filter/README.md) for full details.

---

## robodata segment

Episode segmentation via PELT changepoint detection on proprioception signals. Splits episodes into contiguous phases (sub-skills, regime changes, idle periods) without video processing.

```bash
# Basic segmentation
robodata segment /path/to/dataset

# From HuggingFace Hub
robodata segment hf://lerobot/droid_100

# Export JSON report and timeline visualization
robodata segment /path/to/dataset --export segments.json --plot timeline.png

# Choose signal and tune PELT parameters
robodata segment /path/to/dataset --signal action --penalty 5.0 --cost-model l2

# Sample a subset of episodes
robodata segment /path/to/dataset --sample 20

# Disable per-dimension normalization
robodata segment /path/to/dataset --no-normalize

# Use AIC penalty instead of BIC
robodata segment /path/to/dataset --penalty aic --min-segment-length 15
```

**Options:**
- `--signal`: Proprioception signal to segment on (`observation.state`, `qpos`, `action`, `joint_positions`, `joint_velocities`)
- `--penalty`: PELT penalty — `bic` (default, adapts to signal length/dim), `aic` (less conservative), or a numeric value
- `--cost-model`: Cost function for segment homogeneity (`rbf`, `l2`, `l1`, `normal`, `ar`)
- `--min-segment-length`: Minimum segment length in frames (default: 10)
- `--normalize / --no-normalize`: Z-score normalize per dimension (default: enabled)
- `--sample`: Analyze a random subset of episodes
- `--export`: Save JSON report with per-episode segmentation results
- `--plot`: Generate a timeline PNG visualization
- `--format`: Format hint for dataset loading

**Output:** Per-episode changepoints, segment boundaries with frame/time durations, and dataset-level summary statistics (mean/median/min/max segments).

**Dependencies:** Requires `ruptures>=1.1.0` — install with `pip install "robodata[segment]"`. Timeline plotting requires `matplotlib>=3.7.0`.

See [robodata/segment/README.md](../robodata/segment/README.md) for algorithm details and penalty selection guide.

---

## robodata stats

Compute dataset statistics.

```bash
# Basic statistics
robodata stats /path/to/dataset

# With distribution plots (requires matplotlib)
robodata stats /path/to/dataset --plot

# Export to JSON
robodata stats /path/to/dataset --output stats.json

# Sample subset of episodes (faster)
robodata stats /path/to/dataset --sample 100

# Include quality metrics
robodata stats /path/to/dataset --quality
```

**Statistics include:**
- Episode counts (total, min/max/mean frames)
- Coverage metrics (language, success labels, rewards)
- Action/state distributions (min, max, mean, std per dimension)
- Quality metrics (with `--quality` flag)

---

## robodata export-video

从任意数据集格式导出相机视频。

```bash
# Export first episode (all cameras as grid)
robodata export-video /path/to/dataset -o demo.mp4

# Export specific episode
robodata export-video /path/to/dataset -e 5 -o episode5.mp4

# Export specific camera only
robodata export-video /path/to/dataset -c wrist_cam -o wrist.mp4

# Export all episodes to a directory
robodata export-video /path/to/dataset --all -o ./videos/

# From HuggingFace Hub
robodata export-video hf://lerobot/pusht -o pusht_demo.mp4

# Override FPS
robodata export-video /path/to/dataset -o demo.mp4 --fps 30

# Force grid layout for multiple cameras
robodata export-video /path/to/dataset -o demo.mp4 --grid
```

**Options:**
- `-e, --episode`: Episode index (default: 0)
- `-c, --camera`: Export only this camera
- `-a, --all`: Export all episodes
- `-f, --fps`: Override frames per second
- `-g, --grid`: Combine cameras into grid layout

---

## robodata hub

在 HuggingFace Hub 中搜索与下载数据集。

```bash
# List popular robotics datasets
robodata hub

# Search by query
robodata hub "robot manipulation"
robodata hub "droid"

# Filter by author/organization
robodata hub --author lerobot
robodata hub --author berkeley-humanoid

# Download a dataset
robodata hub --download lerobot/pusht
```

下载的数据集默认缓存目录为 `~/.cache/robodata/datasets/`。

---

## robodata formats

列出支持的格式与能力。

```bash
robodata formats
```

Shows read/write/visualize support for each format.

---

## robodata version

显示版本信息。

```bash
robodata version
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBODATA_CACHE_DIR` | Dataset cache location | `~/.cache/robodata` |
| `ROBODATA_LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Examples

### Convert Open-X dataset for LeRobot training

```bash
# Download from HuggingFace
robodata hub --download openvla/droid_100

# Inspect to understand structure
robodata inspect ~/.cache/robodata/datasets/openvla/droid_100

# Convert to LeRobot v3
robodata convert ~/.cache/robodata/datasets/openvla/droid_100 \
    ./droid_lerobot \
    --format lerobot-v3 \
    --workers 4
```

### Convert LeRobot dataset for OpenVLA/Octo training

```bash
# LeRobot → RLDS
robodata convert hf://lerobot/pusht ./pusht_rlds --format rlds
```

### Convert to RoboDM for efficient storage

```bash
# Any format → RoboDM (up to 70x compression)
robodata convert hf://lerobot/pusht ./pusht_robodm --format robodm

# Visualize with fast OpenCV backend
robodata visualize ./pusht_robodm --backend opencv
```

### Verify conversion quality

```bash
# Compare original vs converted
robodata visualize original_dataset/ --compare converted_dataset/

# Check statistics match
robodata stats original_dataset/ --output original.json
robodata stats converted_dataset/ --output converted.json
diff original.json converted.json
```
