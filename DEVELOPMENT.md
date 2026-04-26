# Robodata Development Guide

## 概述

Robodata 是一个高性能的机器人数据集转换和处理框架。本文档为开发者提供详细的开发指南。

## 架构概览

### 核心模块

- **`robodata/core/`** - 核心功能和工具
  - `models.py` - 数据模型定义
  - `exceptions.py` - 异常类定义
  - `manifest.py` - 数据集清单管理
  - `typing_utils.py` - 类型工具
  - `cache.py` - 高级缓存系统
  - `parallel.py` - 并行处理框架
  - `error_handling.py` - 错误处理工具
  - `constants.py` - 常量定义
  - `config_manager.py` - 配置管理
  - `quality.py` - 代码质量检查

- **`robodata/formats/`** - 格式支持
  - `registry.py` - 格式注册表
  - `common.py` - 共享工具函数
  - `[format]/reader.py` - 格式读取器
  - `[format]/writer.py` - 格式写入器

- **`robodata/convert/`** - 数据转换
  - `converter.py` - 主转换器
  - `__init__.py` - 公共 API

- **`robodata/cli/`** - 命令行界面
  - `main.py` - CLI 入口点
  - `commands/` - 子命令模块

- **`robodata/visualize/`** - 可视化工具
  - `unified_viewer.py` - 统一查看器
  - `cv_viewer.py` - OpenCV 查看器
  - `web_viewer.py` - Web 查看器

## 开发环境设置

### 1. 克隆仓库

```bash
git clone <repository-url>
cd robodata
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 运行测试

```bash
pytest -q
```

## 代码质量

### 质量检查工具

使用内置的开发工具进行代码质量检查：

```bash
# 运行所有质量检查
python -m robodata.dev_tools quality

# 只检查错误和警告
python -m robodata.dev_tools quality --severity warning

# 检查特定目录
python -m robodata.dev_tools quality robodata/formats
```

### 代码格式化

```bash
# 检查格式
python -m robodata.dev_tools format

# 自动格式化
black robodata/
```

### 依赖检查

```bash
# 检查依赖关系
python -m robodata.dev_tools deps
```

## 编码规范

### 命名约定

- **类名**: PascalCase (`DatasetReader`)
- **函数/变量名**: snake_case (`read_episodes`)
- **常量**: UPPER_SNAKE_CASE (`DEFAULT_CACHE_SIZE`)
- **私有函数**: 前缀下划线 (`_internal_function`)

### 类型注解

所有公共函数和方法都应该有类型注解：

```python
def read_episodes(self, path: Path) -> Iterator[Episode]:
    """Read episodes from dataset.
    
    Args:
        path: Path to dataset directory
        
    Returns:
        Iterator of Episode objects
    """
    pass
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def process_data(self, data: np.ndarray, threshold: float = 0.5) -> bool:
    """Process data with threshold.
    
    Args:
        data: Input data array
        threshold: Processing threshold
        
    Returns:
        True if processing succeeded
        
    Raises:
        ValueError: If data is invalid
    """
    pass
```

## 添加新格式支持

### 1. 创建格式模块

在 `robodata/formats/` 下创建新目录：

```
robodata/formats/myformat/
├── __init__.py
├── reader.py
└── writer.py
```

### 2. 实现读取器

```python
# robodata/formats/myformat/reader.py
from robodata.formats.registry import FormatReader

class MyFormatReader(FormatReader):
    def read_episodes(self, path: Path) -> Iterator[Episode]:
        # 实现读取逻辑
        pass
        
    def inspect(self, path: Path) -> DatasetInfo:
        # 实现检查逻辑
        pass
```

### 3. 注册格式

```python
# robodata/formats/myformat/__init__.py
from robodata.formats.registry import register_format
from .reader import MyFormatReader
from .writer import MyFormatWriter

register_format("myformat", MyFormatReader, MyFormatWriter)
```

## 性能优化

### 缓存使用

使用内置的缓存系统：

```python
from robodata.core.cache import get_image_cache

cache = get_image_cache()
image = cache.get_image("key")
if image is None:
    image = load_image()
    cache.put_image("key", image)
```

### 并行处理

使用并行处理框架：

```python
from robodata.core.parallel import parallel_map

results = parallel_map(
    process_item,
    items,
    num_workers=4,
    use_processes=True
)
```

### 内存管理

注册内存清理回调：

```python
from robodata.core.cache import get_memory_manager

memory_manager = get_memory_manager()
memory_manager.register_cleanup_callback(cleanup_function)
```

## 配置管理

### 环境变量

支持以下环境变量：

- `ROBODATA_CACHE_SIZE` - 缓存大小
- `ROBODATA_MEMORY_LIMIT_MB` - 内存限制
- `ROBODATA_MAX_WORKERS` - 最大工作进程数
- `ROBODATA_DEBUG` - 调试模式
- `ROBODATA_LOG_LEVEL` - 日志级别

### 配置文件

创建配置文件 `robodata.yaml`：

```yaml
cache:
  max_size: 1000
  max_size_bytes: 1073741824
  enable_compression: true

performance:
  memory_limit_mb: 4096
  max_workers_cpu: 8
  enable_parallel: true

visualization:
  max_episodes: 50
  default_fps: 10.0

logging:
  level: INFO
  enable_console: true
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest -q

# 运行特定测试
pytest tests/test_converter.py

# 运行带覆盖率的测试
pytest --cov=robodata
```

### 编写测试

使用 pytest 和标准测试模式：

```python
def test_format_reader():
    reader = MyFormatReader()
    episodes = list(reader.read_episodes(test_data_path))
    assert len(episodes) > 0
```

## 发布流程

### 1. 版本更新

更新 `pyproject.toml` 中的版本号。

### 2. 质量检查

```bash
python -m robodata.dev_tools quality --severity error
python -m robodata.dev_tools format
```

### 3. 运行测试

```bash
pytest -q
```

### 4. 构建包

```bash
python -m build
```

### 5. 发布

```bash
python -m twine upload dist/*
```

## 调试

### 启用调试模式

```bash
export ROBODATA_DEBUG=1
python -m robodata convert dataset output
```

### 日志配置

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 常见问题

### Q: 如何添加新的数据类型支持？

A: 在 `robodata/core/models.py` 中添加新的数据类型，然后在相应的格式读取器中实现转换逻辑。

### Q: 如何优化内存使用？

A: 使用缓存系统、并行处理框架，并配置适当的内存限制。

### Q: 如何处理大型数据集？

A: 使用流式处理、分块读取和并行处理来处理大型数据集。

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 编写测试
4. 确保代码质量检查通过
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
