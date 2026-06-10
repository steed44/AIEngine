# Phase 4 — Qt 上位机设计文档

## 目标

构建一个基于 Qt5 的精简上位机应用程序 `AICoreUI.exe`，调用 aicore.dll 的 C API 进行单张图片推理并显示检测结果。

## 功能范围（精简版）

- 菜单：文件 → 打开图片 (Ctrl+O)、退出
- 选择图片后自动加载默认流水线配置，调用推理引擎
- 显示原图 + 检测框（含标签和置信度）
- 侧栏显示检测结果 JSON

## 不包含

- 摄像头/视频流接入
- 配置编辑
- 多流水线管理
- 性能仪表盘

## 架构

```
┌─────────────────────────────────────────────────┐
│                   MainWindow                     │
│  ┌──────────┐  ┌──────────────────┐ ┌─────────┐ │
│  │ 菜单栏    │  │                  │ │         │ │
│  │ File     │  │  QLabel          │ │ QTextEdit│ │
│  │  ├ Open  │  │  (图片+检测框)    │ │ (检测   │ │
│  │  ├ Exit  │  │                  │ │  结果   │ │
│  └──────────┘  │  QScrollArea 包裹 │ │  JSON)  │ │
│                │                  │ │         │ │
│                └──────────────────┘ └─────────┘ │
└─────────────────────────────────────────────────┘
```

## 数据流

### 内存管理

调用 `aicore_result_to_json` 返回的 JSON 字符串指针指向内部静态缓冲区，**调用方不得 free**。每次 `StoreError` 调用会覆盖前一次内容，调用方应**立即复制**所需字符串。

`aicore_result_free` 释放的是 `AICoreResult` 句柄（即 result 对象），而非 JSON 字符串。

```cpp
const char* jsonStr = aicore_result_to_json(result);
QString json = QString::fromUtf8(jsonStr); // 立即复制
aicore_result_free(result);                // 释放句柄
```

### 生命周期

- **启动时**：读取 `pipeline_config.json`（与应用同目录，不存在则使用硬编码默认配置）→ `aicore_pipeline_create(configJson, &error)` 创建一次 pipeline，复用至窗口关闭。若创建失败，显示错误信息并禁用"打开图片"按钮
- **打开图片**：复用已有 pipeline，每次执行 `aicore_pipeline_execute`
- **关闭窗口**：`aicore_pipeline_destroy(pipeline)` 释放资源

### 线程模型

推理调用 `aicore_pipeline_execute` 通过 `QtConcurrent::run` 在后台线程执行（该函数内部无线程限制，可安全在后台线程调用）。完成后通过信号/槽更新 UI 组件。执行期间界面显示"推理中..."提示。

### 单次推理流

1. 用户选择图片 → QFileDialog → QImage
2. `QImage → raw unsigned char* (RGB)` 
3. 调用 `aicore_pipeline_execute(pipeline, data, w, h, 3, &result, &error)`
4. 调用 `aicore_result_to_json(result)` 获取 JSON 字符串
5. 解析 JSON → 提取 detections 列表
6. QPainter 在图片上绘制 bounding boxes + label + confidence
7. QLabel::setPixmap 显示绘制后的图片
8. QTextEdit 显示 JSON 结果

### 坐标系

bbox 中的 (x, y, w, h) 为像素坐标，相对于原图尺寸。QPainter 直接映射到 QPixmap 的像素位置。

## C API 引用

Qt 应用通过链接 `aicore.lib`（导入库）并包含以下头文件调用推理引擎：

| 头文件 | 内容 |
|--------|------|
| `include/api/aicore_api.h` | C API 函数声明（7 个导出函数） |

## 文件结构

```
aicore/gui/
  main.cpp            ← QApplication + MainWindow 启动
  main_window.h       ← MainWindow 类声明
  main_window.cpp     ← MainWindow 实现
```

## CMake 集成

添加到 `aicore/CMakeLists.txt`：

```cmake
set(CMAKE_AUTOMOC ON)
find_package(Qt5 REQUIRED COMPONENTS Widgets Concurrent)

add_executable(AICoreUI
    gui/main.cpp
    gui/main_window.cpp
)
target_include_directories(AICoreUI PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(AICoreUI PRIVATE Qt5::Widgets Qt5::Concurrent aicore)
```

需要设置 `CMAKE_PREFIX_PATH` 指向 `C:/Qt/Qt5.12.11/5.12.11/msvc2017_64`。

### JSON 结果格式

`aicore_result_to_json` 返回的 JSON 格式：

```json
{
  "timestamp": 1718000000,
  "latency_ms": 45.6,
  "status": 0,
  "detections": [
    {
      "node_id": "det_1",
      "label": "defect",
      "confidence": 0.95,
      "bbox": {"x": 10, "y": 20, "w": 100, "h": 200}
    }
  ]
}
```

`status` 字段：0=OK, 1=Skip, 2=ErrorConfigParse, ...

### 默认流水线配置

应用内硬编码一个最小流水线配置 JSON：

```json
{
    "pipeline": {
        "name": "default",
        "max_concurrency": 1,
        "enable_profiling": true,
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "params": {}
            }
        ],
        "edges": []
    }
}
```

用户也可通过配置文件 `pipeline_config.json`（与应用同目录）覆盖默认配置。默认配置仅有 input 节点（无检测节点），覆盖为实际 pipeline 后才能看到检测框结果。

### 图片格式转换

QImage 加载图片 → 转换为 RGB888 格式 → 获取原始字节指针传给 C API。

```cpp
QImage img(path);
QImage rgb = img.convertToFormat(QImage::Format_RGB888);
const unsigned char* data = rgb.bits();
int w = rgb.width(), h = rgb.height();
```

### 检测框绘制

解析 JSON 中 detections 数组，每个元素包含 bbox (x,y,w,h)、label、confidence。使用 QPainter 在原图上绘制矩形框 + 文字标签。

### 错误处理

- 图片加载失败 → QMessageBox::warning
- 流水线创建失败 → 显示 errorOut 内容
- 推理失败 → 显示错误信息
- JSON 解析失败 → 显示原始结果字符串

## 测试

手动测试：
1. 启动 AICoreUI.exe
2. 文件 → 打开 → 选择测试图片
3. 验证图片显示正确
4. 验证检测框绘制正确（位置、标签、置信度）
5. 验证 JSON 侧栏显示正确
