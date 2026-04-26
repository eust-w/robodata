# Model-to-Format Reference

Quick reference for which robotics foundation models use which data formats.

## Format Summary

| Model | Organization | Primary Format | Framework | Notes |
|-------|--------------|----------------|-----------|-------|
| **RT-1** | Google | RLDS | TensorFlow | Original Open-X format |
| **RT-2** | Google | RLDS | TensorFlow/JAX | VLM-based |
| **RT-X** | Google | RLDS | TensorFlow | Cross-embodiment |
| **Octo** | Berkeley | RLDS | JAX/Flax | Open-X pretrained |
| **OpenVLA** | Stanford/Berkeley | RLDS | PyTorch | Fine-tunes on RLDS |
| **ACT** | Stanford | HDF5 | PyTorch | ALOHA demonstrations |
| **Diffusion Policy** | Columbia | Zarr | PyTorch | Also supports HDF5 |
| **LeRobot** | HuggingFace | LeRobot v2 | PyTorch | Parquet + MP4 |
| **SmolVLA** | HuggingFace | LeRobot v2 | PyTorch | Lightweight VLA |
| **π₀ (Pi-zero)** | Physical Intelligence | Custom | PyTorch | Proprietary format |
| **RoboCasa** | UT Austin | HDF5 | PyTorch | robomimic-based |
| **MimicGen** | NVIDIA | HDF5 | PyTorch | robomimic-based |
| **robomimic** | Stanford | HDF5 | PyTorch | Benchmark suite |

---

## Detailed Breakdown

### RLDS-Based Models

**RLDS** (Reinforcement Learning Datasets) is the TensorFlow Datasets format used by Google's robotics team and the Open-X Embodiment project.

#### RT-1 / RT-2 / RT-X (Google DeepMind)
- **Format:** RLDS (TFRecord)
- **Framework:** TensorFlow / JAX
- **Dataset:** Open-X Embodiment
- **Data source:** `gs://gresearch/robotics/` or TFDS
- **Robodata conversion:** `robodata convert hf://dataset ./output --format lerobot-v3`

#### Octo (Berkeley AI Research)
- **Format:** RLDS
- **Framework:** JAX / Flax
- **Pretrained on:** Open-X Embodiment (800k+ episodes)
- **Fine-tuning:** Expects RLDS format
- **Repo:** [octo-models/octo](https://github.com/octo-models/octo)
- **Robodata use case:** Convert LeRobot datasets to RLDS for Octo fine-tuning

#### OpenVLA (Stanford/Berkeley)
- **Format:** RLDS (for training data)
- **Framework:** PyTorch (model), but data loading via RLDS
- **Pretrained on:** Open-X Embodiment
- **Repo:** [openvla/openvla](https://github.com/openvla/openvla)
- **Robodata use case:** Convert custom datasets to RLDS for OpenVLA fine-tuning

---

### LeRobot-Based Models

**LeRobot** is HuggingFace's robotics framework using Parquet + MP4 storage.

#### LeRobot (HuggingFace)
- **Format:** LeRobot v2 (current), v1 (legacy)
- **Framework:** PyTorch
- **Structure:** `data/` (Parquet) + `videos/` (MP4)
- **Hub:** [huggingface.co/lerobot](https://huggingface.co/lerobot)
- **Repo:** [huggingface/lerobot](https://github.com/huggingface/lerobot)

#### SmolVLA (HuggingFace)
- **Format:** LeRobot v2
- **Framework:** PyTorch
- **Description:** Lightweight Vision-Language-Action model
- **Training data:** LeRobot Hub datasets
- **Robodata use case:** Convert RLDS/Open-X to LeRobot for SmolVLA training

---

### HDF5-Based Models

**HDF5** is used by Stanford's robomimic ecosystem and ALOHA-related projects.

#### ACT (Action Chunking with Transformers)
- **Format:** HDF5
- **Framework:** PyTorch
- **Dataset structure:** robomimic-style
- **Used for:** ALOHA bimanual manipulation
- **Repo:** [tonyzhaozh/act](https://github.com/tonyzhaozh/act)
- **Robodata use case:** Convert ALOHA HDF5 to LeRobot for broader compatibility

#### robomimic
- **Format:** HDF5
- **Framework:** PyTorch
- **Structure:** `/data/demo_0/obs/`, `/data/demo_0/actions/`
- **Repo:** [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)
- **Datasets:** Can, Lift, Square, Transport, etc.

#### RoboCasa (UT Austin)
- **Format:** HDF5 (robomimic-compatible)
- **Framework:** PyTorch
- **Description:** Large-scale simulation for household robots
- **Repo:** [robocasa/robocasa](https://github.com/robocasa/robocasa)

#### MimicGen (NVIDIA)
- **Format:** HDF5
- **Framework:** PyTorch
- **Description:** Automated data generation from demonstrations
- **Repo:** [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen)

---

### Zarr-Based Models

**Zarr** is a chunked array format popular in diffusion-based policies.

#### Diffusion Policy (Columbia)
- **Format:** Zarr (primary), HDF5 (also supported)
- **Framework:** PyTorch
- **Structure:** Chunked arrays with metadata
- **Repo:** [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- **Robodata use case:** Convert Zarr to LeRobot or vice versa

#### UMI (Universal Manipulation Interface)
- **Format:** Zarr
- **Framework:** PyTorch
- **Description:** Portable data collection system
- **Repo:** [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)

---

### Proprietary/Custom Formats

#### π₀ / Pi-zero (Physical Intelligence)
- **Format:** Proprietary (not publicly documented)
- **Framework:** PyTorch
- **Note:** Commercial model, format details not released

#### DROID (Berkeley/Toyota)
- **Format:** RLDS (publicly released subset)
- **Framework:** Varies
- **Hub:** Available on HuggingFace
- **Robodata use case:** `robodata convert hf://droid_100 ./output --format lerobot-v3`

---

## Common Conversion Paths

### For OpenVLA/Octo Fine-tuning (need RLDS)
```bash
# LeRobot → RLDS
robodata convert hf://lerobot/aloha_sim ./output --format rlds

# HDF5 → RLDS
robodata convert ./robomimic_data ./output --format rlds
```

### For LeRobot/SmolVLA Training (need LeRobot v2/v3)
```bash
# RLDS/Open-X → LeRobot
robodata convert hf://openvla/droid_100 ./output --format lerobot-v3

# Zarr → LeRobot
robodata convert ./diffusion_policy_data ./output --format lerobot-v3

# HDF5 → LeRobot
robodata convert ./aloha_demos ./output --format lerobot-v3
```

### For Diffusion Policy (need Zarr)
```bash
# LeRobot → Zarr
robodata convert hf://lerobot/pusht ./output --format zarr
```

---

## Format Feature Comparison

| Feature | RLDS | LeRobot v2 | HDF5 | Zarr |
|---------|------|------------|------|------|
| **Images** | Encoded bytes | MP4 video | Raw arrays | Chunked arrays |
| **Compression** | Optional GZIP | H.264 video | Optional | Blosc/Zstd |
| **Streaming** | Yes (TFRecord) | Yes (Parquet) | No | Yes |
| **Random access** | By episode | By frame | Full | Chunked |
| **Cloud-native** | GCS/S3 | HuggingFace Hub | No | Yes |
| **Typical size** | Large (raw) | Compact (video) | Large (raw) | Medium |

---

## When to Use Each Format

| Use Case | Recommended Format |
|----------|-------------------|
| Training OpenVLA/Octo | RLDS |
| Training with LeRobot/SmolVLA | LeRobot v2/v3 |
| Publishing to HuggingFace Hub | LeRobot v2 |
| Diffusion Policy experiments | Zarr |
| Working with ALOHA/robomimic | HDF5 |
| Maximum compatibility | Convert to multiple via Robodata |
