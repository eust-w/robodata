# Robodata v2.0 使用指南和最佳实践

## 📖 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [使用示例](#使用示例)
4. [最佳实践](#最佳实践)
5. [性能优化](#性能优化)
6. [故障排除](#故障排除)
7. [高级用法](#高级用法)

## 🚀 快速开始

### 安装和设置

```bash
# 安装 Robodata v2.0
pip install robodata[v2]

# 或从源码安装
git clone https://github.com/robodata/robodata.git
cd robodata
pip install -e ".[all]"
```

### 基本使用

```python
import robodata

# 检查数据集
info = robodata.inspect("./my_dataset")
print(f"Format: {info.format}, Episodes: {info.num_episodes}")

# 转换数据集
result = robodata.convert(
    "./my_dataset", 
    "./output", 
    target_format="lerobot-v3"
)

# 质量评估
quality_report = robodata.quality("./output")
print(f"Average quality score: {quality_report.average_score}")
```

## 🧠 核心概念

### 1. 数据流水线 (Pipeline)

Robodata v2.0 引入了统一的数据处理流水线概念：

```python
from robodata.core.pipeline import ProcessingPipeline, ProcessingConfig, Processor

# 创建自定义处理器
class MyProcessor(Processor):
    def process_item(self, item):
        return processed_item
    
    def setup(self):
        print("Processor ready")
    
    def cleanup(self):
        print("Processor cleaned up")

# 配置流水线
config = ProcessingConfig(
    mode=ProcessingMode.PARALLEL,
    num_workers=4,
    fail_fast=False
)

# 创建并运行流水线
pipeline = ProcessingPipeline(MyProcessor(), config)
result = pipeline.execute(source_data, sink_data)
```

### 2. 结构化日志

所有日志都使用结构化格式：

```python
from robodata.core.logging_system import get_logger

logger = get_logger(__name__)
logger.info("Processing episode", episode_id="ep_001", format="rlds")
logger.error("Conversion failed", error="Invalid data", episode_id="ep_001")
```

### 3. 配置管理

分层配置系统支持多种配置源：

```python
from robodata.core.config_manager_v2 import get_config, set_config, temporary_config

# 获取配置
num_workers = get_config("conversion.num_workers", 1)

# 设置配置
set_config("conversion.num_workers", 8)

# 临时配置
with temporary_config(num_workers=16, fail_fast=True):
    # 临时配置生效
    result = convert_dataset(...)
```

### 4. 性能监控

实时性能监控和分析：

```python
from robodata.core.performance_monitor import profile_operation, get_performance_report

# 监控特定操作
with profile_operation("episode_conversion"):
    result = convert_episode(...)

# 获取性能报告
report = get_performance_report()
print(f"CPU usage: {report['resources']['cpu_usage']['current']}%")
```

## 💡 使用示例

### 示例 1: 基本数据转换

```python
import robodata
from robodata.core.config_manager_v2 import temporary_config

# 使用临时配置优化性能
with temporary_config(
    conversion_num_workers=8,
    performance_memory_limit_mb=4096,
    cache_max_size=2000
):
    # 转换数据集
    result = robodata.convert(
        source="./large_dataset",
        output="./converted_output",
        target_format="lerobot-v3"
    )
    
    print(f"Converted {result.episodes_converted} episodes")
    print(f"Failed: {result.episodes_failed} episodes")
```

### 示例 2: 数据质量分析

```python
from robodata.validation.engine import validate_dataset
from robodata.analytics.engine import analyze_dataset, AnalysisType

# 加载数据集
dataset_info = robodata.inspect("./my_dataset")
episodes = list(load_episodes("./my_dataset"))

# 验证数据质量
validation_result = validate_dataset(dataset_info, episodes)
print(f"Validation passed: {validation_result.success}")
print(f"Issues found: {len(validation_result.issues)}")

# 深度分析
analysis_result = analyze_dataset(
    dataset_info, 
    episodes,
    analysis_types=[AnalysisType.STATISTICAL, AnalysisType.QUALITY]
)

# 查看洞察
for insight in analysis_result.high_insights:
    print(f"🔍 {insight.title}: {insight.description}")
```

### 示例 3: 数据预处理

```python
from robodata.preprocessing.pipeline import (
    preprocess_episodes, 
    PreprocessingConfig,
    AugmentationType
)

# 配置预处理
config = PreprocessingConfig(
    enable_cleaning=True,
    enable_normalization=True,
    enable_augmentation=True,
    normalization_config={"method": "standard"},
    augmentation_config={
        "factor": 3,
        "types": [AugmentationType.NOISE_ADDITION, AugmentationType.DROPOUT],
        "noise": {"action_std": 0.02, "state_std": 0.01},
        "dropout": {"action_rate": 0.1, "state_rate": 0.05}
    }
)

# 预处理数据
result = preprocess_episodes(episodes, config)
print(f"Original episodes: {result.statistics['data_cleaner']['original_episodes']}")
print(f"Augmented episodes: {result.statistics['data_augmenter']['augmented_episodes']}")
```

### 示例 4: 自定义处理器

```python
from robodata.core.pipeline import Processor, ProcessingPipeline
from robodata.core.performance_monitor import performance_monitor

class CustomFilterProcessor(Processor):
    """自定义过滤器处理器"""
    
    def __init__(self, min_quality_score=6.0):
        self.min_quality_score = min_quality_score
        self.filtered_count = 0
    
    def setup(self):
        self.logger = get_logger(__name__)
        self.logger.info("Custom filter processor ready")
    
    def cleanup(self):
        self.logger.info(f"Filtered {self.filtered_count} items")
    
    @performance_monitor("custom_filter")
    def process_item(self, episode):
        # 评估质量
        quality_score = evaluate_episode_quality(episode)
        
        if quality_score >= self.min_quality_score:
            return episode
        else:
            self.filtered_count += 1
            return None  # 过滤掉低质量数据

# 使用自定义处理器
processor = CustomFilterProcessor(min_quality_score=7.0)
pipeline = ProcessingPipeline(processor)
result = pipeline.execute(episodes)
```

### 示例 5: 错误处理和恢复

```python
from robodata.core.error_handling_v2 import (
    error_context, 
    ErrorCategory,
    ErrorHandler
)

# 全局错误处理
error_handler = ErrorHandler()

# 注册错误恢复策略
def recover_from_format_error(error_report):
    """恢复格式错误的策略"""
    if "corrupted" in error_report.exception.args[0]:
        # 尝试修复损坏的数据
        return attempt_data_repair(error_report.context.source_path)
    return False

error_handler.register_recovery_strategy(ErrorCategory.FORMAT, recover_from_format_error)

# 使用错误上下文
try:
    with error_context("dataset_conversion", source_path="./data"):
        result = convert_dataset(...)
except Exception as e:
    print(f"Conversion failed: {e}")
    # 错误已被自动处理和记录
```

## 🎯 最佳实践

### 1. 配置管理

```python
# ✅ 好的做法：使用分层配置
from robodata.core.config_manager_v2 import get_config, set_config

# 设置全局默认值
set_config("conversion.num_workers", 4)
set_config("cache.max_size", 1000)

# 在特定操作中使用临时配置
with temporary_config(num_workers=8):
    result = convert_large_dataset(...)

# ❌ 避免：硬编码配置
def convert_data():
    num_workers = 4  # 硬编码，不灵活
    ...
```

### 2. 性能优化

```python
# ✅ 好的做法：使用性能监控
from robodata.core.performance_monitor import (
    start_performance_monitoring,
    profile_operation
)

# 启动全局监控
start_performance_monitoring()

# 监控关键操作
with profile_operation("critical_conversion"):
    result = convert_dataset(...)

# 定期检查性能报告
report = get_performance_report()
if report['resources']['memory_usage']['current'] > 80:
    print("⚠️ High memory usage detected!")

# ❌ 避免：无监控的大量数据处理
def process_large_dataset():
    # 没有性能监控，难以发现问题
    for episode in episodes:
        process_episode(episode)
```

### 3. 错误处理

```python
# ✅ 好的做法：结构化错误处理
from robodata.core.error_handling_v2 import error_context, handle_errors

@handle_errors("episode_processing")
def process_episode(episode):
    with error_context("episode_processing", episode_id=episode.episode_id):
        # 处理逻辑
        return processed_episode

# ❌ 避免：裸露异常处理
def process_episode(episode):
    try:
        # 处理逻辑
        pass
    except:  # 太宽泛，隐藏了错误
        pass
```

### 4. 日志记录

```python
# ✅ 好的做法：结构化日志
from robodata.core.logging_system import get_logger

logger = get_logger(__name__)

def process_episode(episode):
    logger.info("Starting episode processing", 
                episode_id=episode.episode_id, 
                frames=len(list(episode.load_frames())))
    
    try:
        result = process_episode_data(episode)
        logger.info("Episode processed successfully", 
                    episode_id=episode.episode_id,
                    processing_time=result.duration)
        return result
    except Exception as e:
        logger.error("Episode processing failed", 
                    episode_id=episode.episode_id,
                    error=str(e),
                    exc_info=True)
        raise

# ❌ 避免：使用 print 语句
def process_episode(episode):
    print(f"Processing episode {episode.episode_id}")  # 不结构化
    try:
        result = process_episode_data(episode)
        print("Success")  # 没有上下文
        return result
    except Exception as e:
        print(f"Error: {e}")  # 缺少详细信息
```

### 5. 缓存使用

```python
# ✅ 好的做法：智能缓存
from robodata.core.cache_v2 import create_cache, CacheStrategy, cached

# 创建适合的缓存
cache = create_cache(
    "episode_processing",
    max_size=500,
    strategy=CacheStrategy.LRU,
    enable_disk_cache=True
)

# 使用缓存装饰器
@cached("expensive_computation", ttl_seconds=3600)
def compute_expensive_feature(episode_id):
    # 耗时的计算
    return expensive_calculation(episode_id)

# ❌ 避免：无缓存或过度缓存
def compute_feature(episode_id):
    # 每次都重新计算，效率低
    return expensive_calculation(episode_id)
```

## ⚡ 性能优化

### 1. 内存优化

```python
# 使用流式处理处理大数据集
from robodata.core.pipeline import ProcessingConfig, ProcessingMode

config = ProcessingConfig(
    mode=ProcessingMode.SEQUENTIAL,  # 减少内存使用
    chunk_size=100,  # 小块处理
    memory_limit_mb=1024  # 内存限制
)

# 启用磁盘缓存
from robodata.core.cache_v2 import create_cache, CacheStrategy

cache = create_cache(
    "processing",
    max_size=100,
    strategy=CacheStrategy.LRU,
    enable_disk_cache=True,  # 使用磁盘缓存
    max_disk_size_mb=2048
)
```

### 2. 并行处理

```python
# 根据系统资源自动调整并行度
from robodata.core.performance_monitor import get_performance_report

report = get_performance_report()
cpu_cores = report['resources'].get('cpu_cores', 4)
memory_usage = report['resources']['memory_usage']['current']

# 动态调整工作进程数
if memory_usage < 50:
    num_workers = min(cpu_cores, 8)
else:
    num_workers = max(1, cpu_cores // 2)

config = ProcessingConfig(
    mode=ProcessingMode.PARALLEL,
    num_workers=num_workers
)
```

### 3. IO 优化

```python
# 批量操作减少 IO 开销
from robodata.preprocessing.pipeline import DataSampler

config = PreprocessingConfig(
    sampling_config={
        "target_fps": 10,  # 降低帧率
        "min_episode_length": 50  # 过滤短序列
    }
)

# 使用 SSD 缓存目录
from robodata.core.config_manager_v2 import set_config

set_config("cache.disk_cache_dir", "/tmp/robodata_cache")  # 使用高速存储
```

## 🔧 故障排除

### 常见问题和解决方案

#### 1. 内存不足错误

```python
# 问题：MemoryError 或 OutOfMemoryError
# 解决方案：启用内存优化配置

from robodata.core.config_manager_v2 import temporary_config

with temporary_config(
    performance_memory_limit_mb=1024,  # 限制内存使用
    conversion_num_workers=1,  # 减少并行进程
    cache_max_size=100,  # 减少缓存大小
    cache_enable_disk_cache=True  # 启用磁盘缓存
):
    result = convert_dataset(...)
```

#### 2. 转换速度慢

```python
# 问题：转换过程非常缓慢
# 解决方案：性能分析和优化

from robodata.core.performance_monitor import get_performance_report

# 获取性能报告
report = get_performance_report()

# 检查瓶颈
if report['resources']['cpu_usage']['current'] < 50:
    print("💡 建议：增加工作进程数")
    
if report['resources']['memory_usage']['current'] > 80:
    print("💡 建议：减少内存使用或启用磁盘缓存")

# 自动优化
optimizations = report.get('auto_optimizations', {})
if optimizations:
    print("🚀 自动优化建议：")
    for key, value in optimizations.items():
        print(f"  {key}: {value}")
```

#### 3. 数据质量问题

```python
# 问题：数据验证失败
# 解决方案：使用数据清洗和预处理

from robodata.validation.engine import validate_dataset
from robodata.preprocessing.pipeline import clean_episodes

# 验证数据
validation_result = validate_dataset(dataset_info, episodes)

if not validation_result.success:
    print("🔍 数据质量问题：")
    for issue in validation_result.critical_issues:
        print(f"  - {issue.message}")
    
    # 清洗数据
    cleaning_result = clean_episodes(episodes)
    cleaned_episodes = cleaning_result.processed_episodes
    
    print(f"✅ 清洗完成：{len(cleaned_episodes)} 个有效片段")
```

#### 4. 格式不兼容

```python
# 问题：UnsupportedFormatError
# 解决方案：检查格式支持和格式检测

from robodata.formats.registry import FormatRegistry

# 检查支持的格式
print("支持的读取格式：", list(FormatRegistry._readers.keys()))
print("支持的写入格式：", list(FormatRegistry._writers.keys()))

# 手动指定格式
result = robodata.convert(
    source="./ambiguous_dataset",
    output="./output",
    source_format="hdf5",  # 手动指定源格式
    target_format="lerobot-v3"
)
```

## 🚀 高级用法

### 1. 自定义格式支持

```python
from robodata.formats.registry import FormatRegistry
from robodata.core.protocols import FormatReader, FormatWriter

@FormatRegistry.register_reader("my_format")
class MyFormatReader(FormatReader):
    @classmethod
    def can_read(cls, path):
        return (path / "my_format_marker").exists()
    
    def inspect(self, path):
        # 实现检查逻辑
        return dataset_info
    
    def read_episodes(self, path):
        # 实现读取逻辑
        yield episode

# 使用自定义格式
info = robodata.inspect("./my_dataset", source_format="my_format")
```

### 2. 分布式处理

```python
# 使用 Ray 进行分布式处理
from robodata.core.config_manager_v2 import temporary_config

with temporary_config(
    performance_ray_address="ray://head-node:10001",
    conversion_num_workers=16,
    enable_distributed=True
):
    result = convert_large_dataset(...)
```

### 3. 实时监控

```python
# 设置实时监控
from robodata.core.performance_monitor import start_performance_monitoring
import time

start_performance_monitoring()

# 长时间运行的任务
for batch in dataset_batches:
    result = process_batch(batch)
    
    # 定期检查性能
    report = get_performance_report()
    if report['resources']['memory_usage']['current'] > 90:
        print("⚠️ 内存使用过高，考虑优化")
        break
    
    time.sleep(1)  # 避免过度监控
```

### 4. 插件系统

```python
# 创建自定义插件
from robodata.core.pipeline import Processor

class QualityFilterPlugin(Processor):
    """质量过滤插件"""
    
    def __init__(self, min_quality=6.0):
        self.min_quality = min_quality
    
    def process_item(self, episode):
        quality = compute_quality_score(episode)
        if quality >= self.min_quality:
            return episode
        return None

# 注册插件
pipeline = PreprocessingPipeline()
pipeline.add_preprocessor(QualityFilterPlugin(min_quality=7.0))
```

## 📚 更多资源

- [API 文档](https://robodata.readthedocs.io/)
- [架构设计](docs/architecture.md)
- [迁移指南](scripts/migrate_to_v2.py)
- [示例代码](examples/)
- [社区论坛](https://discuss.robodata.org)

---

💡 **提示**: 遇到问题时，首先查看日志输出，然后使用性能监控工具分析瓶颈，最后参考故障排除部分寻找解决方案。

🎯 **目标**: Robodata v2.0 旨在提供高效、可靠、易用的机器人数据集处理工具。遵循这些最佳实践可以最大化系统的性能和稳定性。
