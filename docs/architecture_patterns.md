# 架构设计模式文档

## 概述

本文档描述了 Robodata 项目中使用的架构设计模式和优化策略。通过应用这些设计模式，我们实现了高度模块化、可扩展和可维护的代码架构。

## 设计模式概览

### 1. 工厂模式 (Factory Pattern)

#### 位置
- `robodata/formats/factory.py`

#### 目的
创建和管理格式读取器和写入器的实例，提供统一的接口来获取不同格式的处理器。

#### 实现
```python
class FormatFactory:
    def get_reader(self, format_name: str) -> FormatReader
    def get_writer(self, format_name: str) -> FormatWriter
    def detect_format(self, path: Path) -> str
    def register_reader(self, format_name: str, reader_class: Type[FormatReader])
```

#### 优势
- **解耦创建逻辑** - 客户端代码无需了解具体的创建过程
- **统一接口** - 通过工厂统一获取不同格式的处理器
- **易于扩展** - 新增格式只需注册到工厂中
- **缓存优化** - 工厂缓存实例，避免重复创建

#### 使用示例
```python
factory = FormatFactory()
reader = factory.get_reader("zarr")
writer = factory.get_writer("lerobot-v3")
```

### 2. 策略模式 (Strategy Pattern)

#### 位置
- `robodata/convert/strategies.py`

#### 目的
封装不同的数据转换策略，允许在运行时选择最适合的转换方法。

#### 实现
```python
class ConversionStrategyBase(ABC):
    @abstractmethod
    def execute(self, context, reader, writer) -> Dict[str, Any]

class SequentialConversionStrategy(ConversionStrategyBase)
class ParallelConversionStrategy(ConversionStrategyBase)
class StreamingConversionStrategy(ConversionStrategyBase)
class BatchConversionStrategy(ConversionStrategyBase)
```

#### 优势
- **算法族封装** - 不同的转换策略被封装在独立的类中
- **运行时选择** - 可以根据数据集特征动态选择策略
- **易于扩展** - 新增策略无需修改现有代码
- **性能优化** - 针对不同场景优化的专用策略

#### 策略选择逻辑
```python
def select_strategy(context, hint=None):
    if hint:
        return strategies[hint]
    
    episode_count = context.episode_count
    total_frames = context.total_frames
    
    if episode_count < 10:
        return SequentialConversionStrategy()
    elif episode_count > 100 and parallel_config:
        return ParallelConversionStrategy()
    elif total_frames > 100000:
        return StreamingConversionStrategy()
    else:
        return BatchConversionStrategy()
```

### 3. 观察者模式 (Observer Pattern)

#### 位置
- `robodata/convert/observers.py`

#### 目的
实现事件驱动的架构，允许组件监听和响应转换过程中的各种事件。

#### 实现
```python
class ConversionObserver(Protocol):
    def on_event(self, event: ConversionEvent) -> None

class ProgressObserver(ConversionObserverBase)
class MetricsObserver(ConversionObserverBase)
class LoggingObserver(ConversionObserverBase)
class FileObserver(ConversionObserverBase)

class ConversionEventDispatcher:
    def add_observer(self, observer: ConversionObserver)
    def dispatch_event(self, event: ConversionEvent)
```

#### 优势
- **松耦合** - 观察者和被观察者之间松耦合
- **可扩展** - 可以轻松添加新的观察者类型
- **实时监控** - 实时监控转换进度和状态
- **多播支持** - 一个事件可以通知多个观察者

#### 事件类型
```python
class ConversionEventType(Enum):
    STARTED = "started"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    EPISODE_FAILED = "episode_failed"
    PROGRESS_UPDATE = "progress_update"
    WARNING = "warning"
    ERROR = "error"
    COMPLETED = "completed"
    FINALIZED = "finalized"
```

### 4. 命令模式 (Command Pattern)

#### 位置
- `robodata/convert/commands.py`

#### 目的
将操作封装为命令对象，支持撤销、重做和批量操作。

#### 实现
```python
class Command(Protocol):
    def execute(self) -> CommandResult
    def undo(self) -> CommandResult

class ConvertDatasetCommand(ConversionCommand)
class InspectDatasetCommand(ConversionCommand)
class ValidateDatasetCommand(ConversionCommand)

class CommandInvoker:
    def execute_command(self, command: Command) -> CommandResult
    def undo_last_command(self) -> Optional[CommandResult]
```

#### 优势
- **操作封装** - 每个操作都被封装为独立的命令
- **撤销支持** - 支持操作的撤销和重做
- **批量操作** - 可以批量执行多个命令
- **历史记录** - 保留操作历史，便于审计和调试

#### 命令执行流程
```python
invoker = CommandInvoker()
command = ConvertDatasetCommand(source, output, "zarr", "lerobot-v3", config)
result = invoker.execute_command(command)

if not result.success:
    invoker.undo_last_command()
```

### 5. 建造者模式 (Builder Pattern)

#### 位置
- `robodata/convert/builder.py`

#### 目的
简化复杂对象的构建过程，提供流畅的API来配置转换操作。

#### 实现
```python
class ConversionBuilder:
    def source(self, path) -> "ConversionBuilder"
    def output(self, path) -> "ConversionBuilder"
    def target_format(self, format_name) -> "ConversionBuilder"
    def parallel(self, num_workers=None) -> "ConversionBuilder"
    def with_logging(self, log_level="INFO") -> "ConversionBuilder"
    def build(self) -> ConvertDatasetCommand

class PipelineBuilder:
    def convert(self, source, output, target_format) -> "PipelineBuilder"
    def inspect(self, dataset_path) -> "PipelineBuilder"
    def validate(self, dataset_path) -> "PipelineBuilder"
    def build(self) -> ConversionPipeline
```

#### 优势
- **流畅API** - 提供链式调用的流畅接口
- **参数验证** - 在构建时验证参数的有效性
- **复杂配置** - 简化复杂配置的构建过程
- **类型安全** - 编译时类型检查和IDE支持

#### 使用示例
```python
command = (ConversionBuilder()
    .source("/data/dataset")
    .output("/data/output")
    .target_format("lerobot-v3")
    .parallel(num_workers=4)
    .with_logging()
    .with_progress()
    .build())

pipeline = (PipelineBuilder()
    .validate("/data/dataset")
    .convert("/data/dataset", "/data/output", "lerobot-v3")
    .with_metrics()
    .build())
```

### 6. 模板方法模式 (Template Method Pattern)

#### 位置
- `robodata/formats/base.py`

#### 目的
定义算法骨架，让子类重写特定步骤而不改变算法结构。

#### 实现
```python
class BaseReader(ABC):
    def get_episode_count(self, path: Path) -> int:
        # 模板方法：定义默认实现
        try:
            count = 0
            for _ in self.read_episodes(path):
                count += 1
            return count
        except Exception as e:
            self._logger.warning(f"Could not count episodes: {e}")
            return -1
    
    @abstractmethod
    def read_episodes(self, path: Path) -> Iterator[Episode]:
        # 抽象方法：子类必须实现
        pass
```

#### 优势
- **代码复用** - 公共逻辑在基类中实现
- **一致性** - 保证所有子类的行为一致
- **易于扩展** - 子类只需实现特定的抽象方法
- **维护性** - 修改公共逻辑只需修改基类

### 7. 协议模式 (Protocol Pattern)

#### 位置
- `robodata/formats/base.py`
- `robodata/convert/observers.py`

#### 目的
定义接口契约，支持鸭子类型和结构化子类型。

#### 实现
```python
@runtime_checkable
class FormatReader(Protocol):
    @property
    def format_name(self) -> str: ...
    
    @classmethod
    def can_read(cls, path: Path) -> bool: ...
    
    def inspect(self, path: Path) -> DatasetInfo: ...
    
    def read_episodes(self, path: Path) -> Iterator[Episode]: ...

@runtime_checkable
class ConversionObserver(Protocol):
    def on_event(self, event: ConversionEvent) -> None: ...
```

#### 优势
- **类型安全** - 编译时类型检查
- **灵活性** - 支持多种实现方式
- **测试友好** - 易于创建测试替身
- **文档化** - 协议本身就是接口文档

## 架构层次

### 1. 表示层 (Presentation Layer)
```
robodata/cli/
├── main.py              # CLI入口点
├── commands/            # CLI命令模块
└── visualize/           # 可视化组件
```

### 2. 应用层 (Application Layer)
```
robodata/convert/
├── converter.py         # 主转换器（门面模式）
├── strategies.py        # 策略模式实现
├── commands.py          # 命令模式实现
├── observers.py         # 观察者模式实现
└── builder.py           # 建造者模式实现
```

### 3. 领域层 (Domain Layer)
```
robodata/core/
├── models.py            # 领域模型
├── exceptions.py        # 领域异常
├── protocols.py         # 领域协议
└── constants.py         # 领域常量
```

### 4. 基础设施层 (Infrastructure Layer)
```
robodata/formats/
├── factory.py           # 工厂模式实现
├── base.py              # 基础抽象类
├── registry.py          # 格式注册表
└── [format]/            # 具体格式实现
```

## 性能优化架构

### 1. 缓存架构
```
robodata/core/cache.py
├── LRUCache              # LRU缓存实现
├── ImageCache            # 图像专用缓存
├── MemoryManager         # 内存管理器
└── CacheContext          # 缓存上下文管理
```

### 2. 并行处理架构
```
robodata/core/parallel.py
├── WorkerPool            # 工作池抽象
├── ParallelConfig        # 并行配置
├── BatchProcessor        # 批处理器
└── ProgressTracker       # 进度跟踪器
```

### 3. 配置管理架构
```
robodata/core/config_manager.py
├── RobodataConfig         # 主配置类
├── CacheConfig           # 缓存配置
├── PerformanceConfig     # 性能配置
└── ConfigContext         # 配置上下文
```

## 数据流架构

### 1. 转换数据流
```
Source Dataset → Reader → Strategy → Writer → Target Dataset
     ↓               ↓         ↓        ↓
   Format       Episodes  Parallel  Episodes
 Detection     Loading   Processing Writing
```

### 2. 事件流
```
Conversion Event → Event Dispatcher → Multiple Observers
       ↓                ↓                    ↓
   Event Data      Event Routing      Logging/Metrics/Progress
```

### 3. 命令流
```
Client Code → Builder → Command → Invoker → Execution
     ↓         ↓        ↓        ↓         ↓
   Builder   Command  History  Results  Rollback
```

## 扩展点设计

### 1. 格式扩展
- 实现 `FormatReader` 和 `FormatWriter` 协议
- 使用装饰器注册到工厂
- 继承 `BaseReader`/`BaseWriter` 获得通用功能

### 2. 策略扩展
- 继承 `ConversionStrategyBase`
- 实现 `execute` 方法
- 注册到 `ConversionStrategySelector`

### 3. 观察者扩展
- 实现 `ConversionObserver` 协议
- 注册到事件分发器
- 响应特定事件类型

### 4. 命令扩展
- 继承 `ConversionCommand`
- 实现 `execute` 和 `undo` 方法
- 集成到命令调用器

## 质量保证

### 1. 类型安全
- 广泛使用协议和类型注解
- 运行时类型检查
- IDE智能提示支持

### 2. 错误处理
- 统一的异常层次结构
- 上下文管理器模式
- 优雅的错误恢复

### 3. 测试支持
- 依赖注入设计
- 模拟对象友好
- 集成测试框架

### 4. 文档化
- 完整的文档字符串
- 类型注解即文档
- 架构决策记录

## 性能特征

### 1. 可扩展性
- 水平扩展：并行处理支持
- 垂直扩展：缓存和内存优化
- 格式扩展：插件化架构

### 2. 可维护性
- 模块化设计
- 清晰的职责分离
- 标准化的接口

### 3. 可测试性
- 依赖注入
- 接口抽象
- 模拟友好

### 4. 可观测性
- 事件驱动架构
- 丰富的指标收集
- 详细的日志记录

## 最佳实践

### 1. 设计原则
- **单一职责** - 每个类只有一个变化的理由
- **开闭原则** - 对扩展开放，对修改关闭
- **里氏替换** - 子类可以替换父类
- **接口隔离** - 客户端不应依赖不需要的接口
- **依赖倒置** - 依赖抽象而非具体实现

### 2. 编码规范
- 使用描述性的变量名
- 保持函数简洁
- 编写清晰的文档字符串
- 遵循类型注解规范

### 3. 架构决策
- 优先组合而非继承
- 使用协议定义接口
- 实现优雅的错误处理
- 保持向后兼容性

## 总结

通过应用这些设计模式和架构原则，Robodata 项目实现了：

✅ **高度模块化** - 清晰的模块边界和职责分离  
✅ **易于扩展** - 新功能可以通过插件方式添加  
✅ **性能优化** - 针对不同场景的优化策略  
✅ **类型安全** - 编译时类型检查和运行时验证  
✅ **可测试性** - 友好的测试架构和模拟支持  
✅ **可观测性** - 完整的事件监控和指标收集  
✅ **可维护性** - 清晰的代码结构和文档  

这种架构设计为 Robodata 项目提供了坚实的技术基础，支持未来的功能扩展和性能优化需求。
