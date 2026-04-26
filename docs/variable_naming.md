# 变量命名规范

## 概述

本文档定义了 Robodata 项目中的变量命名规范，确保代码的可读性和一致性。

## 命名原则

### 1. 清晰性和描述性
变量名应该清楚地表达其用途和含义，避免使用模糊或过于简短的名称。

### 2. 一致性
在整个项目中保持一致的命名风格。

### 3. 可读性
变量名应该易于阅读和理解。

## 命名规范

### 1. 变量名（snake_case）

所有变量名使用小写字母和下划线的组合：

```python
# ✅ 好的命名
episode_count = 10
frame_index = 0
dataset_path = Path("/data/dataset")
camera_images = {}
segment_data = []

# ❌ 避免的命名
cnt = 10
idx = 0
path = Path("/data/dataset")
imgs = {}
data = []
```

### 2. 循环变量

即使是循环变量，也应该使用描述性名称：

```python
# ✅ 好的命名
for episode_idx, episode in enumerate(episodes):
    for frame_idx, frame in enumerate(episode.frames()):
        process_frame(frame_idx, frame)

# ❌ 避免的命名
for i, e in enumerate(episodes):
    for j, f in enumerate(e.frames()):
        process_frame(j, f)
```

### 3. 函数参数

函数参数应该使用描述性名称：

```python
# ✅ 好的命名
def process_episode(
    episode_data: EpisodeData,
    output_path: Path,
    config: ProcessingConfig
) -> ProcessResult:
    pass

# ❌ 避免的命名
def process(data, path, cfg):
    pass
```

### 4. 临时变量

即使是临时变量，也应该有清晰的名称：

```python
# ✅ 好的命名
actions_subset = actions[:, :8]
states_subset = states[:, :8]
segment_lengths = [seg.end_frame - seg.start_frame for seg in segments]

# ❌ 避免的命名
a = actions[:, :8]
s = states[:, :8]
lengths = [seg.end_frame - seg.start_frame for seg in segments]
```

### 5. 布尔变量

布尔变量应该以 `is_`, `has_`, `can_`, `should_`, `will_` 等开头：

```python
# ✅ 好的命名
is_valid = True
has_images = False
can_process = True
should_skip = False
will_export = True

# ❌ 避免的命名
valid = True
images = False
process = True
skip = False
export = True
```

### 6. 集合变量

集合变量应该使用复数形式：

```python
# ✅ 好的命名
episodes = []
frames = []
cameras = {}
segments = []

# ❌ 避免的命名
episode_list = []
frame_array = []
camera_dict = []
segment_collection = []
```

### 7. 索引和计数器

索引和计数器应该使用描述性名称：

```python
# ✅ 好的命名
episode_index = 0
frame_counter = 1
segment_count = len(segments)

# ❌ 避免的命名
i = 0
j = 1
cnt = len(segments)
```

## 特定场景命名

### 1. 图像处理

```python
# ✅ 好的命名
red_channel, green_channel, blue_channel = pixel_values
image_height, image_width = image.shape
pixel_data = np.array(image)

# ❌ 避免的命名
r, g, b = pixel_values
h, w = image.shape
data = np.array(image)
```

### 2. 数据处理

```python
# ✅ 好的命名
input_data = load_data()
processed_data = transform(input_data)
output_data = save(processed_data)

# ❌ 避免的命名
inp = load_data()
proc = transform(inp)
out = save(proc)
```

### 3. 文件和路径

```python
# ✅ 好的命名
source_directory = Path("/source")
output_file = Path("/output/result.json")
config_path = Path("/config/settings.yaml")

# ❌ 避免的命名
src = Path("/source")
out = Path("/output/result.json")
cfg = Path("/config/settings.yaml")
```

## 避免的命名模式

### 1. 单字母变量（除了数学上下文）

```python
# ❌ 避免的命名
for i in range(10):
    for j in range(5):
        x = i * j

# ✅ 好的命名（除非是数学公式）
for row_idx in range(10):
    for col_idx in range(5):
        product = row_idx * col_idx
```

### 2. 缩写（除非是广泛认知的）

```python
# ❌ 避免的命名
num = 10
temp = get_temporary_data()
conf = load_config()

# ✅ 好的命名
number = 10
temporary_data = get_temporary_data()
configuration = load_config()
```

### 3. 模糊的名称

```python
# ❌ 避免的命名
data = load_data()
result = process(data)
value = calculate()

# ✅ 好的命名
dataset_data = load_data()
processing_result = process(dataset_data)
calculated_value = calculate()
```

## 例外情况

### 1. 数学上下文

在数学计算中，可以使用传统的数学变量名：

```python
# ✅ 可接受的命名
for x in range(10):
    y = x * 2 + 1
    z = math.sqrt(y)
```

### 2. 坐标

在坐标系统中，可以使用 x, y, z：

```python
# ✅ 可接受的命名
x_coord, y_coord = point.position
z_coord = point.height
```

### 3. 颜色通道

在图像处理中，可以使用 r, g, b：

```python
# ✅ 可接受的命名
red_value, green_value, blue_value = pixel
```

## 重构指南

### 1. 识别问题变量

查找以下模式：
- 单字母变量（除了循环计数器）
- 缩写变量名
- 模糊的变量名
- 不一致的命名风格

### 2. 重构步骤

1. **识别变量用途**：理解变量的实际用途
2. **选择描述性名称**：选择能清楚表达用途的名称
3. **全局替换**：在整个作用域内替换变量名
4. **测试验证**：确保重构后功能正常

### 3. 工具辅助

使用代码质量检查工具：
```bash
python -m robodata.dev_tools quality
```

## 示例对比

### 处理前
```python
def proc(d, p, cfg):
    cnt = 0
    for i, e in enumerate(d):
        if cfg.filter(e):
            cnt += 1
            save(e, p)
    return cnt
```

### 处理后
```python
def process_dataset(
    dataset_data: list[Episode],
    output_path: Path,
    config: ProcessingConfig
) -> int:
    processed_count = 0
    for episode_idx, episode in enumerate(dataset_data):
        if config.should_filter_episode(episode):
            processed_count += 1
            save_episode(episode, output_path)
    return processed_count
```

## 检查清单

在代码审查时，检查以下项目：

- [ ] 所有变量名都是描述性的
- [ ] 循环变量有清晰的名称
- [ ] 布尔变量以适当的词缀开头
- [ ] 集合变量使用复数形式
- [ ] 没有单字母变量（除了数学上下文）
- [ ] 没有不必要的缩写
- [ ] 命名风格一致
- [ ] 变量名表达实际用途

## 总结

良好的变量命名是代码可读性的基础。通过遵循这些规范，我们可以：
- 提高代码的可读性
- 减少理解代码的时间
- 降低维护成本
- 提高团队协作效率

记住：代码是写给人读的，其次才是给机器执行的。
