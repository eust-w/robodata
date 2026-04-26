# Robodata 命名规范

## 变量命名规范

### 1. 避免 单字母变量名
- **避免**: `for k in keys`, `for v in values`
- **推荐**: `for key in keys`, `for value in values`

### 2. 使用描述性名称
- **避免**: `tmp_path`, `data_group`, `arr`
- **推荐**: `temp_path`, `dataset_group`, `array_data`

### 3. 函数命名
- 使用动词或动词短语
- **避免**: `_get_data()`, `_process_item()`
- **推荐**: `_load_dataset()`, `_process_episode()`

### 4. 常量命名
- 使用 `UPPER_SNAKE_CASE`
- **示例**: `DEFAULT_FPS`, `MAX_CACHE_SIZE`

### 5. 私有函数命名
- 内部工具函数：使用 `_` 前缀但描述性命名
- **避免**: `_extract()`, `_process()`
- **推荐**: `_extract_camera_name()`, `_process_episode_data()`

### 6. 类命名
- 使用 `PascalCase`
- **示例**: `DatasetReader`, `ConversionConfig`

## 代码组织规范

### 1. 共享工具函数
- 将重复的工具函数提取到 `robodata.formats.common` 模块
- **示例**: `numpy_dtype_to_dtype()`, `extract_camera_name_from_key()`

### 2. 导入顺序
1. 标准库
2. 第三方库
3. 本地模块（robodata.*）

### 3. 类型注解
- 优先使用 `Optional[T]` 而不是 `T | None`（Typer 兼容性）
- 使用具体的类型而不是 `Any`

## 实施中的改进

### 已完成的改进
1. **重复工具函数统一**
   - `_numpy_dtype_to_dtype()` → `numpy_dtype_to_dtype()` (common.py)
   - `_extract_camera_name()` → `extract_camera_name_from_key()` (common.py)

2. **文件命名优化**
   - `ray_backend.py` → `ray.py` (简化命名)

### 待改进的命名
1. **变量命名改进**
   - `tmp_path` → `temp_path`
   - `data_group` → `dataset_group`
   - `arr` → `array_data`
   - `f` → `file_handle` (在 HDF5 读取器中)

2. **函数命名改进**
   - `get_*()` 方法可以更具体
   - 私有方法使用更描述性的名称
