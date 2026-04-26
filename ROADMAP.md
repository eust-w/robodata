# Robodata Roadmap

## Current Status (v0.1)

### Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| RLDS | ✅ | ✅ | Open X-Embodiment, OpenVLA, Octo |
| LeRobot v2 | ✅ | - | HuggingFace robotics |
| LeRobot v3 | ✅ | ✅ | Target format for all conversions |
| Zarr | ✅ | - | Diffusion Policy datasets |
| HDF5 | ✅ | - | robomimic, ACT/ALOHA datasets |
| ROS bags | ✅ | - | Raw robot recordings |

### Core Features
- [x] Format auto-detection
- [x] CLI (`robodata inspect`, `convert`, `visualize`, `formats`, `hub`)
- [x] YAML configuration files
- [x] Parallel episode processing (`--workers N`)
- [x] Unified visualizer
- [x] HuggingFace Hub integration (`hf://` URLs)

---

## Completed (v0.1.1)

### ✅ HuggingFace Hub Integration
**Status:** Done

```bash
# Download and inspect
robodata inspect hf://lerobot/pusht

# Download and convert
robodata convert hf://openvla/modified_libero_rlds output/ --format lerobot-v3

# Search datasets
robodata hub "robot manipulation"
robodata hub --author lerobot
```

Features:
- [x] `hf://` URL scheme for HuggingFace datasets
- [x] Automatic download and caching (`~/.cache/robodata/datasets/`)
- [x] Dataset search/discovery (`robodata hub` command)
- [ ] Streaming support for large datasets (future)

---

## Completed (v0.2)

### ✅ RLDS Writer
**Status:** Done

```bash
# Convert LeRobot to RLDS
robodata convert lerobot_dataset/ output/ --format rlds

# Round-trip conversion (RLDS → LeRobot v3 → RLDS)
robodata convert original_rlds/ temp/ --format lerobot-v3
robodata convert temp/ reconverted_rlds/ --format rlds
```

Features:
- [x] Write RLDS format (TFRecord)
- [x] Generate proper episode metadata (dataset_info.json, features.json)
- [x] Support OXE-compatible schema
- [x] Round-trip conversion verified

---

## High Priority (v0.3)

### 1. Validation Tools
**Why:** Ensure converted datasets are correct and complete.

```bash
robodata validate output/ --check-frames --check-videos
```

- [ ] Frame count validation
- [ ] Video integrity checks
- [ ] Schema compatibility checks
- [ ] Compare source vs converted statistics

---

## Medium Priority (v0.3)

### 4. Additional Formats

#### DROID Dataset
- Large-scale bimanual manipulation dataset
- Already converted to LeRobot: `IPEC-COMMUNITY/droid_lerobot`

#### BridgeData V2
- 60k+ trajectories from WidowX robots
- Core component of Octo training

#### LIBERO
- Simulation benchmark used by OpenVLA and Pi0
- Pre-converted versions exist on HF Hub

### 5. Streaming & Large Dataset Support
- [ ] Lazy loading for datasets > RAM
- [ ] Chunked processing
- [ ] Resume interrupted conversions

### ✅ Statistics & Analysis
```bash
robodata stats dataset/ --plot
robodata stats dataset/ --sample 100 --output stats.json
```
- [x] Action/state distributions (min, max, mean, std per dimension)
- [x] Episode length histograms
- [x] Coverage metrics (language, success labels, rewards)
- [x] JSON export for programmatic access
- [x] Matplotlib visualization with --plot flag

---

## Low Priority (v0.4+)

### 7. HuggingFace Upload
```bash
robodata upload output/ --repo my-org/my-dataset
```

### 8. Data Augmentation
- [ ] Image augmentations during conversion
- [ ] Action noise injection
- [ ] Temporal subsampling

### 9. Format Converters
- [x] HDF5 reader (robomimic, ACT/ALOHA) ✅ Done
- [ ] MuJoCo dataset support
- [ ] Isaac Gym recordings

---

## VLA Model Compatibility Matrix

| Model | Training Format | Fine-tune Format | Robodata Support |
|-------|----------------|------------------|---------------|
| OpenVLA | RLDS | RLDS | ✅ Read |
| Octo | RLDS | RLDS | ✅ Read |
| Pi0/OpenPI | Proprietary | LeRobot v2 | ✅ Read/Write |
| ACT | HDF5 | HDF5 | ✅ Read |
| Diffusion Policy | Zarr | Zarr | ✅ Read |

---

## References

- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) - 1M+ trajectories, RLDS format
- [OpenVLA](https://github.com/openvla/openvla) - Uses RLDS, fine-tunes on RLDS
- [Octo](https://github.com/octo-models/octo) - Uses RLDS from OXE
- [OpenPI (Pi0)](https://github.com/Physical-Intelligence/openpi) - Uses LeRobot for fine-tuning
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robotics standard
