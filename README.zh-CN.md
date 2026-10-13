<div align="center">
  <img src="docs/readme/hero.svg" alt="Robodata — workflow overview" width="100%">
  <h1>Robodata</h1>
  <p><strong>让机器人示教数据更容易读取、转换和理解。</strong></p>
  <p><img alt="Python: 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square">
  <img alt="Status: Alpha" src="https://img.shields.io/badge/Status-Alpha-8b7fd1?style=flat-square">
  <img alt="Core: Episode / Frame" src="https://img.shields.io/badge/Core-Episode%20%2F%20Frame-6366f1?style=flat-square"></p>
  <p>
    <a href="#overview">项目亮点</a> ·
    <a href="#quick-start">快速开始</a> ·
    <a href="#workflow">工作流</a> ·
    <a href="#validation">验证与边界</a> ·
    <a href="#documentation">文档</a>
  </p>
  <p><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>
</div>

---

<a id="overview"></a>
## 为机器人数据提供共同语言

**Robodata** 将数据集检查、格式转换、质量评分与 episode 分段集中到一套 Python 工具中。不同 reader 和 writer 通过统一的 `Episode` / `Frame` 表达连接，新增适配器可以复用已有处理流程。

| 转换格式 | 理解示教 | 筛选数据 |
| :--- | :--- | :--- |
| RLDS、LeRobot、Zarr、HDF5、Rosbag、RoboDM | 查看结构、评估运动质量、识别时序阶段 | 数据集目录、episode 筛选与输出检查 |

### Episode 分段示例

![Episode 分段时间线示例](docs/assets/segment_timeline.png)

*仓库现有示例。分段检测的是本体感知信号的统计变化，不等于已经理解每段动作的任务语义。*

<a id="quick-start"></a>
## 快速开始

> [!WARNING]
> **已知启动阻塞：**当前 [validation 模块](robodata/validation/engine.py) 的同名类字段遮蔽了 `dataclasses.field`，CLI 检查在导入阶段报 `TypeError: 'NoneType' object is not callable`。以下命令用于说明预期接口，需要先修复该代码问题才能正常运行。本次文档更新不修改实现。

需要 **Python 3.10+**。先安装核心，再按数据格式选择可选依赖。

```bash
git clone https://github.com/eust-w/robodata.git
cd robodata
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
robodata --help
robodata registry list
```

以 HDF5 → LeRobot 为例：

```bash
python -m pip install -e '.[lerobot,hdf5,video,visualize]'
robodata inspect /path/to/dataset
robodata convert /path/to/hdf5_dataset ./converted --format lerobot-v3
```

路径需替换为真实、受支持的数据集。使用 Hub 来源时再安装 `.[hub]`；远程数据可能还需要登录、下载和单独接受使用条款。

<a id="workflow"></a>
## 统一中间表示

```text
RLDS / LeRobot / Zarr / HDF5 / Rosbag / RoboDM
                       │ Reader
                       ▼
                 Episode / Frame
                       │
            检查 / 评分 / 筛选 / 分段
                       │ Writer
                       ▼
                  转换后的数据集
```

| 格式 | 读取 | 写入 | 依赖组 / 说明 |
| :--- | :---: | :---: | :--- |
| RLDS | ✓ | ✓ | `rlds`，依赖 TensorFlow / TFDS |
| LeRobot v2 / v3 | ✓ | ✓ | `lerobot`，视频功能按需补依赖 |
| GR00T | ✓ | — | 带 embodiment 元数据的 LeRobot 风格输入 |
| Zarr | ✓ | — | `zarr` |
| HDF5 | ✓ | — | `hdf5` |
| ROS bags | ✓ | — | `rosbag` |
| RoboDM | ✓ | ✓ | `robodm`；现有集成说明建议按需从上游源码安装 |

完整约束见 [格式参考](docs/format_reference.md) 和 [模型格式对应表](docs/model_formats.md)。

### 检查 → 评分 → 筛选 → 分段

```bash
robodata quality /path/to/dataset --export quality.json
robodata filter /path/to/dataset --min-quality 6.0
robodata filter /path/to/dataset ./filtered --min-quality 6.0
python -m pip install -e '.[segment,visualize]'
robodata segment /path/to/dataset --export segments.json --plot timeline.png
```

不指定输出目录的 filter 命令用于预览。评分覆盖平滑性、无效动作、夹爪抖动、静止片段、时间戳规律性、饱和、动作熵和路径长度；它们是诊断指标，不能替代任务成功标签或训练效果验证。

### Python 接口

```python
import robodata

info = robodata.inspect("/path/to/dataset")
print(info.format, info.num_episodes, info.cameras)
robodata.convert("/path/to/source", "./output", target_format="lerobot-v3")
```

<a id="validation"></a>
## 验证与当前范围

```bash
python -m pip install -e '.[dev]'
python -m pytest
robodata registry validate
```

格式相关测试需要对应可选依赖。项目仍为 **Alpha**：先用小样本确认相机映射、时间戳、动作/状态语义和转换结果，再处理完整训练集。

数据集合并/拆分、更完整的深度保留、GR00T writer、自动转换等价性检查仍是后续方向，不应由当前支持表推导为已完成。

<a id="documentation"></a>
## 文档导航

| 入口 | 内容 |
| :--- | :--- |
| [CLI](docs/cli.md) · [配置](docs/configuration.md) | 命令与转换参数 |
| [架构](docs/architecture.md) | 统一模型与适配器 |
| [质量指标](robodata/quality/README.md) | 指标定义和参考文献 |
| [筛选](robodata/filter/README.md) · [分段](robodata/segment/README.md) | Episode 分析 |
| [目录贡献指南](robodata/registry/CONTRIBUTING.md) | 数据集条目维护 |

## 许可状态

早期 README 声明 MIT，但当前仓库没有根 `LICENSE` 文件，仍需维护者补齐正式许可文本。数据集和依赖的许可独立适用。
