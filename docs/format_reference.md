# Robotics Dataset Format Reference

A comprehensive guide to robotics dataset formats supported by Robodata.

## Format Comparison

| Format | Container | Video | Tabular | Compression | Random Access | Ecosystem |
|--------|-----------|-------|---------|-------------|---------------|-----------|
| RLDS | TFRecord | Per-frame PNG/JPEG | Protocol Buffers | Low | Poor | TensorFlow, Open-X |
| LeRobot v2/v3 | Parquet + MP4 | H.264 video | Apache Parquet | Medium | Good | HuggingFace |
| RoboDM | Matroska (.vla) | H.265 video | Pickle streams | High (70x) | Good | Berkeley |
| Zarr | Zarr chunks | Per-frame arrays | Zarr arrays | Medium | Excellent | Diffusion Policy |
| HDF5 | Single .hdf5 | Per-frame arrays | HDF5 datasets | Low-Medium | Good | robomimic, ACT |
| Rosbag | .bag / MCAP | Compressed msgs | ROS messages | Varies | Poor | ROS |

---

## RLDS (Reinforcement Learning Datasets)

**Used by:** Open-X Embodiment, Octo, RT-X, OpenVLA

### Structure
```
dataset/
├── dataset_info.json
├── features.json
└── 1.0.0/
    ├── dataset_info.json
    └── rlds_spec-train.tfrecord-00000-of-00001
```

### Key Characteristics
- **Container:** TensorFlow TFRecord (Protocol Buffers)
- **Episode structure:** Nested `steps` containing observations, actions, rewards
- **Images:** Stored as individual encoded frames (PNG/JPEG) per timestep
- **Metadata:** TFDS-style `dataset_info.json` with feature specs

### Schema Example
```python
{
    "steps": {
        "observation": {
            "image": tf.Tensor(shape=[H, W, 3], dtype=uint8),
            "state": tf.Tensor(shape=[N], dtype=float32),
        },
        "action": tf.Tensor(shape=[M], dtype=float32),
        "reward": tf.Tensor(shape=[], dtype=float32),
        "is_terminal": tf.Tensor(shape=[], dtype=bool),
        "language_instruction": tf.Tensor(shape=[], dtype=string),
    }
}
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Standard for Open-X ecosystem | Large file sizes (no video compression) |
| Rich metadata support | Requires TensorFlow |
| Language instruction support | Slow random access |

---

## LeRobot v2/v3

**Used by:** HuggingFace LeRobot, GR00T (NVIDIA Isaac)

### Structure
```
dataset/
├── meta/
│   ├── info.json           # Dataset metadata, feature specs
│   ├── episodes.jsonl      # Episode index and lengths
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.{camera}/
            ├── episode_000000.mp4
            └── ...
```

### Key Characteristics
- **Container:** Apache Parquet (tabular) + MP4 (video)
- **Video codec:** H.264 (yuv420p)
- **Chunking:** Episodes grouped into chunks (default 1000)
- **Version:** `codebase_version` in info.json (`"v2.0"` or `"v2.1"`)

### info.json Schema
```json
{
    "codebase_version": "v2.0",
    "robot_type": "koch",
    "fps": 30.0,
    "total_episodes": 100,
    "total_frames": 50000,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [14]},
        "observation.images.top": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "video_info": {"video.fps": 30.0, "video.codec": "h264"}
        }
    }
}
```

### Parquet Columns
| Column | Type | Description |
|--------|------|-------------|
| `index` | int64 | Global frame index |
| `episode_index` | int64 | Episode number |
| `frame_index` | int64 | Frame within episode |
| `timestamp` | float64 | Time in seconds |
| `observation.state` | float32[] | Robot state vector |
| `action` | float32[] | Action vector |
| `task_index` | int64 | Index into tasks.jsonl |

### Pros & Cons
| Pros | Cons |
|------|------|
| Good compression (H.264) | Multiple files per episode |
| Fast columnar queries | Requires pyarrow + av |
| HuggingFace integration | |

---

## RoboDM (.vla)

**Used by:** Berkeley Automation, OpenVLA research

### Structure
```
dataset/
├── trajectory_000000.vla
├── trajectory_000001.vla
├── ...
└── metadata.json
```

### Key Characteristics
- **Container:** EBML/Matroska (same as .mkv)
- **Video codec:** H.265/HEVC (default), H.264, AV1, or lossless
- **Non-video:** Pickle-serialized numpy arrays in rawvideo streams
- **Single file:** One .vla per episode (self-contained)

### Internal Structure (EBML)
```
.vla file (Matroska container)
├── Track 1: observation/images/ego_view (H.265 video)
├── Track 2: observation/images/wrist (H.265 video)
├── Track 3: observation/state (rawvideo + pickle)
├── Track 4: action (rawvideo + pickle)
└── ...
```

### Hierarchical Keys
```python
trajectory = robodm.Trajectory("episode.vla", mode="r")
data = trajectory.load()
# Keys: observation/images/ego_view, observation/state, action
```

### Compression Comparison
| Source Format | Source Size | RoboDM Size | Ratio |
|---------------|-------------|-------------|-------|
| Zarr (pusht) | 32 MB | 12 MB | 2.7x |
| LeRobot (aloha) | 500 MB | ~70 MB | 7x |
| Raw numpy | 1 GB | ~15 MB | 70x |

### Pros & Cons
| Pros | Cons |
|------|------|
| Best compression (H.265) | Requires manual install |
| Single file per episode | Slower decode than H.264 |
| Standard tools (ffprobe) | Less ecosystem support |

---

## Zarr

**Used by:** Diffusion Policy, UMI, robomimic (some)

### Structure
```
dataset.zarr/
├── .zattrs              # Root attributes (metadata)
├── .zgroup              # Group marker
├── data/
│   ├── .zarray          # Array metadata
│   ├── 0                # Chunk files
│   ├── 1
│   └── ...
├── action/
│   ├── .zarray
│   └── ...
└── img/
    └── {camera}/
        ├── .zarray
        └── ...
```

### Key Characteristics
- **Container:** Zarr (chunked N-dimensional arrays)
- **Images:** Stored as 4D arrays `(episode, time, H, W, C)`
- **Compression:** Blosc, LZ4, or Zstd per-chunk
- **Random access:** Excellent (chunk-level seeking)

### .zattrs Example
```json
{
    "fps": 10,
    "num_episodes": 206,
    "episode_ends": [100, 200, 300, ...]
}
```

### Episode Indexing
```python
import zarr
z = zarr.open("dataset.zarr", "r")
episode_ends = z.attrs["episode_ends"]
# Episode 5: frames from episode_ends[4] to episode_ends[5]
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Excellent random access | No video compression |
| Cloud-native (S3, GCS) | Large storage footprint |
| Simple array model | Images as raw arrays |

---

## HDF5

**Used by:** robomimic, ACT, ALOHA (original)

### Structure
```
dataset.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions          (N, action_dim)
│   │   ├── obs/
│   │   │   ├── agentview_image  (N, H, W, C)
│   │   │   ├── robot0_eef_pos   (N, 3)
│   │   │   └── ...
│   │   ├── rewards          (N,)
│   │   └── dones            (N,)
│   └── demo_1/
│       └── ...
└── mask/
    ├── train              [0, 1, 2, ...]
    └── valid              [100, 101, ...]
```

### Key Characteristics
- **Container:** Single HDF5 file
- **Images:** Raw numpy arrays (optionally gzip compressed)
- **Hierarchy:** `/data/demo_{i}/obs/{key}` structure
- **Masks:** Train/valid splits as index arrays

### Reading Example
```python
import h5py
with h5py.File("dataset.hdf5", "r") as f:
    demo = f["data/demo_0"]
    images = demo["obs/agentview_image"][:]  # (N, H, W, C)
    actions = demo["actions"][:]              # (N, action_dim)
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Single file | No video compression |
| Mature ecosystem | Large file sizes |
| Good random access | Memory-mapped limitations |

---

## Rosbag

**Used by:** ROS-based robots, real-world data collection

### ROS1 (.bag)
```
recording.bag
├── /camera/image_raw      (sensor_msgs/Image)
├── /joint_states          (sensor_msgs/JointState)
├── /cmd_vel               (geometry_msgs/Twist)
└── ...
```

### ROS2 (MCAP)
```
recording.mcap
├── /camera/image_raw      (sensor_msgs/msg/Image)
├── /joint_states          (sensor_msgs/msg/JointState)
└── ...
```

### Key Characteristics
- **Container:** Bag (ROS1) or MCAP (ROS2)
- **Messages:** ROS message types with timestamps
- **Topics:** Organized by ROS topic names
- **Compression:** Optional LZ4/BZ2 per-message

### Pros & Cons
| Pros | Cons |
|------|------|
| Native ROS format | Not ML-friendly |
| Timestamps preserved | Topic-based, not episode-based |
| Standard robot format | Requires episode segmentation |

---

## GR00T (NVIDIA Isaac)

**Based on:** LeRobot v2 with NVIDIA-specific extensions

### Differences from LeRobot
| Aspect | LeRobot | GR00T |
|--------|---------|-------|
| `robot_type` | `"koch"`, `"aloha"` | `"GR1ArmsOnly"`, `"SO100DualArm"` |
| State naming | Generic | `motor_0`, `motor_1`, ... |
| Annotations | `task` | `annotation.human.action.task_description` |
| Validity | - | `annotation.human.validity` |
| Embodiment | - | Links to URDF/robot model |

### GR00T-Specific Features
```json
{
    "features": {
        "observation.state": {
            "dtype": "float64",
            "shape": [44],
            "names": ["motor_0", "motor_1", ..., "motor_43"]
        },
        "annotation.human.validity": {
            "dtype": "int64",
            "shape": [1]
        }
    }
}
```

### Robodata Compatibility
GR00T datasets are read/written using the LeRobot reader/writer:
```bash
robodata inspect hf://nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
robodata convert groot_dataset/ ./output --format lerobot-v3
```

---

## Format Selection Guide

### Choose RLDS when:
- Training Octo, RT-X, or OpenVLA
- Contributing to Open-X Embodiment
- Need TensorFlow ecosystem integration

### Choose LeRobot when:
- Using HuggingFace ecosystem
- Training LeRobot policies
- Need good balance of compression and compatibility

### Choose RoboDM when:
- Storage is limited
- Archiving large datasets
- Need maximum compression

### Choose Zarr when:
- Training Diffusion Policy
- Need cloud-native storage (S3)
- Require fast random access

### Choose HDF5 when:
- Training robomimic or original ACT
- Need single-file simplicity
- Working with existing HDF5 datasets

---

## Conversion Matrix

What Robodata can convert between:

```
           TO →
FROM ↓     RLDS   LeRobot   RoboDM
─────────────────────────────────
RLDS        -       ✓         ✓
LeRobot     ✓       -         ✓
RoboDM      ✓       ✓         -
Zarr        ✓       ✓         ✓
HDF5        ✓       ✓         ✓
Rosbag      ✓       ✓         ✓
```

### Example Conversions
```bash
# ALOHA HDF5 → LeRobot for HuggingFace
robodata convert aloha.hdf5 ./output --format lerobot-v3

# Open-X RLDS → RoboDM for archival
robodata convert hf://openvla/droid ./droid_robodm --format robodm

# Diffusion Policy Zarr → RLDS for Octo training
robodata convert pusht.zarr ./pusht_rlds --format rlds
```
