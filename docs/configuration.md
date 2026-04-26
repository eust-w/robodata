# 配置指南

Robodata 支持使用 YAML 配置文件来自定义转换行为，适用于以下场景：

- 你的数据集字段命名不标准
- 你希望在输出中重命名相机
- 你需要做一些转换前后的处理（例如夹爪值归一化）
- 你需要用同一套配置批量转换多个数据集

## 快速开始

1. **从数据集生成配置模板**：
   ```bash
   robodata inspect my_dataset/ --generate-config config.yaml
   ```

2. **编辑配置**（根据你的数据字段/相机命名进行映射）：
   ```yaml
   target_format: lerobot-v3
   fps: 30
   robot_type: franka

   cameras:
     wrist_camera: wrist_cam
     overhead_camera: front_cam

   fields:
     action: data/actions
     state: data/proprio
   ```

3. **执行转换**：
   ```bash
   robodata convert my_dataset/ output/ --config config.yaml
   ```

## 配置参考

### 基础配置

```yaml
# Target format (required)
target_format: lerobot-v3

# Source format (auto-detected if not specified)
source_format: rlds

# Frames per second (required if not in source metadata)
fps: 30

# Robot identifier
robot_type: franka
```

### 相机映射

将源数据集的相机名称映射到目标名称。Robodata 支持模糊匹配，因此你不需要写入常见前缀（例如 `steps/observation/`）。

```yaml
cameras:
  # source_name: target_name
  agentview_image: front_cam
  robot0_eye_in_hand_image: wrist_cam

  # These all match the same source camera:
  # - steps/observation/wrist_camera
  # - observation.images.wrist_camera
  # - wrist_camera
  wrist_camera: wrist_cam
```

### 字段映射

将源字段路径映射到规范字段名：

```yaml
fields:
  # Simple mapping (source path)
  action: steps/action
  state: observation/robot_state

  # With explicit target name
  gripper:
    source: observation/gripper_pos
    target: gripper_state

  # With transform (future feature)
  gripper_binary:
    source: observation/gripper_pos
    target: gripper_state
    transform: binary  # Normalize to 0/1
```

### 视频配置

```yaml
video:
  codec: h264          # Video codec (default: h264)
  crf: 23              # Quality: lower = better, 18-28 typical
  compress: true       # Enable compression
```

### 行为配置

```yaml
fail_on_error: false   # Stop on first error vs continue
skip_existing: true    # Skip episodes that already exist in output
include_depth: true    # Include depth camera streams
```

### 性能配置

```yaml
num_workers: 4         # episode 并行处理的 worker 数（默认 1）
executor: local        # local|ray，默认 local
ray_address: auto      # 可选：Ray 地址（例如 auto / ray://host:10001）
```

当 `executor: local` 时，Robodata 使用本地 `ProcessPoolExecutor` 进行多进程并行。

当 `executor: ray` 时，Robodata 会把每个 episode 的转换任务提交到 Ray 集群执行（Ray 为可选依赖）。

## CLI 覆盖规则

命令行参数会覆盖配置文件中的同名设置：

```bash
# config 中是 fps: 30，但 CLI 可以覆盖为 60
robodata convert input/ output/ --config config.yaml --fps 60

# 追加相机映射（会覆盖/合并 config 内的 cameras 映射）
robodata convert input/ output/ --config config.yaml --camera extra_cam=side_cam

# 使用 4 个本地多进程 worker
robodata convert input/ output/ --config config.yaml --workers 4

# 使用 Ray 执行（需要安装可选依赖：pip install "robodata[ray]"）
robodata convert input/ output/ --config config.yaml --executor ray --ray-address auto
```

## 配置示例

### RLDS (robomimic) to LeRobot v3

```yaml
target_format: lerobot-v3
fps: 20
robot_type: panda

cameras:
  agentview_image: front_cam
  robot0_eye_in_hand_image: wrist_cam

fields:
  action: steps/action
  state: steps/observation/robot0_eef_pos
```

### Zarr (Diffusion Policy) to LeRobot v3

```yaml
target_format: lerobot-v3
fps: 10
robot_type: pusht

cameras:
  img: observation_image

fields:
  action: action
  state: state
```

### Custom Dataset

```yaml
target_format: lerobot-v3
fps: 30
robot_type: franka

cameras:
  cam_wrist: wrist_cam
  cam_front: front_cam
  cam_side: side_cam

fields:
  action: robot/commanded_joint_positions
  state:
    source: robot/measured_joint_positions
    target: state
  gripper:
    source: robot/gripper_position
    target: gripper_state

video:
  crf: 20  # Higher quality

fail_on_error: true  # Stop on any error
```

## Dry Run

Preview what a conversion will do without writing files:

```bash
robodata convert input/ output/ --config config.yaml --dry-run
```

Output shows:
- Detected source format and schema
- Camera mappings that will be applied
- Field mappings configured
- Target format and settings
