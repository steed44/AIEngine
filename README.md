# AIEngine — 工业检测 C++ AI 推理引擎

基于 NVIDIA GPU 工控机的工业异常检测纯 C++ AI 引擎。支持 YOLOv8 目标检测和 PatchCore 异常检测，提供推理加速、模型优化和训练流水线。

## 架构

```
AIEngine/
├── aicore/                  # 主工程（纯源码，不含编译产物）
│   ├── include/             # 公共头文件
│   ├── src/                 # 实现文件
│   ├── cli/                 # CLI 入口
│   ├── gui/                 # Qt 上位机
│   ├── samples/             # 示例代码
│   ├── scripts/             # Python 导出脚本
│   ├── tests/               # GTest 单元测试
│   └── CMakeLists.txt
├── config/                  # 配置文件集中存放
│   ├── config_optimize.json
│   ├── config_rois.json
│   └── config_train.json
├── data/                    # 数据 / 权重 / 测试输出
│   ├── yolo_weights/        # YOLO 模型权重
│   └── test_output/         # 测试产物（*.bin）
├── docs/                    # 文档 / 设计规格
│   ├── 基于NVIDIA...技术方案.md
│   ├── knowledge-graph/
│   └── superpowers/
├── out/                     # 编译输出（独立于源码）
│   └── build/
├── graphify-out/            # 语义知识图谱
├── scripts/                 # Python 辅助脚本
└── .gitignore
```

## 编译目标

| 目标 | 说明 |
|------|------|
| `aicore.dll` | 推理核心 DLL（7 个 C API 导出） |
| `aicore_optimizer.dll` | 模型优化工具 DLL |
| `aicore_trainer.dll` | 训练模块 DLL |
| `AICoreUI.exe` | Qt 5 上位机（打开图片 → 推理 → 显示检测框） |
| `ModelOptimizer.exe` | 模型优化 CLI |
| `AICoreTrainer.exe` | 训练 CLI |
| `PatchCoreTrain.exe` | PatchCore 训练 CLI（文件夹 → memory bank） |
| `RoiTrain.exe` | 多 ROI 训练 CLI |
| `RoiInfer.exe` | 多 ROI 推理 CLI |
| `aicore_tests.exe` | GTest 单元测试（71 个） |

## 环境要求

| 组件 | 版本 |
|------|------|
| Visual Studio | 2022 (v143) |
| CMake | ≥ 3.20 |
| C++ 标准 | C++17 |
| OpenCV | 4.10.0（通过 vcpkg 安装） |
| Qt 5 | 5.12.11（msvc2017_64） |
| GTest | 1.16.0（通过 vcpkg 安装） |
| LibTorch | 2.5.1 CUDA 11.8（可选，GPU backbone） |
| CUDA | 11.8（可选，TensorRT/GPU 后端） |

## 快速开始

### 1. 配置 CMake

```powershell
cd aicore
cmake -S . -B ../out/build -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE="D:/work/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DOpenCV_DIR="D:/work/vcpkg/vcpkg/installed/x64-windows/share/opencv4" `
  -DCMAKE_PREFIX_PATH="C:/Qt/Qt5.12.11/5.12.11/msvc2017_64"
```

> **注意**：根据你的实际环境调整 vcpkg 路径和 Qt 路径。

### 2. 编译

```powershell
cmake --build ../out/build --config Release
```

Debug 模式：

```powershell
cmake --build ../out/build --config Debug
```

### 3. 运行测试

```powershell
..\out\build\tests\Release\aicore_tests.exe
```

## 使用方式

### AICoreUI（Qt 上位机）

```powershell
..\out\build\Release\AICoreUI.exe
```

- 菜单 `文件 → 打开图片` 选择待检测图像
- 左侧显示原图 + 检测框（YOLO 目标检测）或异常热力图（PatchCore）
- 右侧显示检测结果 JSON

### PatchCore 训练

```powershell
..\out\build\Release\PatchCoreTrain.exe --data ./normal_images/ --model wideresnet.onnx --output memory_bank.bin
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | — | 正常样本图片文件夹（必填） |
| `--model` | — | Backbone ONNX 模型路径（必填） |
| `--output` | memory_bank.bin | Memory bank 输出路径 |
| `--input_size` | 224 | 输入图片尺寸 |
| `--layers` | layer2,layer3 | Backbone 特征提取层名（逗号分隔） |
| `--coreset` | 0.1 | Coreset 保留比例 |
| `--backend` | libtorch | backbone 类型（libtorch / opencv_dnn / model_backend） |

### 多 ROI 训练

```powershell
..\out\build\Release\RoiTrain.exe --config ..\config\config_rois.json
```

ROI 配置文件示例：

```json
{
    "rois": [
        {
            "name": "roi_defect",
            "data_dir": "./train_roi1/",
            "model_path": "wideresnet.onnx",
            "backend_type": "libtorch",
            "input_size": 224,
            "backbone_layers": "layer2,layer3",
            "coreset_fraction": 0.1,
            "output_path": "memory_bank_roi1.bin"
        },
        {
            "name": "roi_scratch",
            "data_dir": "./train_roi2/",
            "model_path": "resnet18.onnx",
            "backend_type": "opencv_dnn",
            "input_size": 224,
            "backbone_layers": "layer2,layer3",
            "coreset_fraction": 0.1,
            "output_path": "memory_bank_roi2.bin"
        }
    ]
}
```

多 ROI 共享同一 backbone 模型，每个 ROI 独立 MemoryBank。训练大图（>100MB）自动启用流式模式，避免爆内存。

### 多 ROI 推理

```powershell
..\out\build\Release\RoiInfer.exe --config ..\config\config_rois.json --image test.jpg
```

### YOLO 推理流水线

创建 JSON 配置文件，通过 `aicore_pipeline_create` C API 加载：

```json
{
    "pipeline": {
        "name": "yolo_demo",
        "max_concurrency": 1,
        "enable_profiling": true,
        "nodes": [
            {"id": "resize", "type": "resize", "params": {"width": "640", "height": "640"}},
            {"id": "normalize", "type": "normalize", "params": {}},
            {"id": "detect", "type": "model", "model_path": "yolov8n.engine", "backend": "tensorrt"},
            {"id": "nms", "type": "nms", "params": {"iou_threshold": "0.45", "confidence_threshold": "0.5"}}
        ],
        "edges": [
            {"from": "input", "to": "resize"},
            {"from": "resize", "to": "normalize"},
            {"from": "normalize", "to": "detect"},
            {"from": "detect", "to": "nms"}
        ]
    }
}
```

### PatchCore 推理流水线

```json
{
    "pipeline": {
        "name": "patchcore_infer",
        "nodes": [
            {"id": "patchcore", "type": "patchcore",
             "params": {
                 "backbone": "opencv_dnn",
                 "model_path": "wideresnet.onnx",
                 "memory_bank_path": "memory_bank.bin",
                 "input_size": "224",
                 "backbone_layers": "layer2,layer3",
                 "anomaly_threshold": "0.5"
             }}
        ],
        "edges": []
    }
}
```

## Backbone 类型

三种 backbone 实现可配置：

| 类型 | 后端 | GPU | 适用场景 |
|------|------|-----|----------|
| `libtorch` | LibTorch 2.5.1 | ✅ CUDA | 主推，GPU 加速推理 |
| `opencv_dnn` | OpenCV DNN | ❌ CPU | 备选，无 GPU 环境 |
| `model_backend` | IModelBackend 接口 | 取决于后端 | 通用接口，通过 backendType 指定具体后端（tensorrt/onnxruntime/libtorch） |

通过 `CreateBackbone()` 工厂统一创建，多 ROI 共享 backbone 避免重复加载。

## C API

```c
AICorePipeline aicore_pipeline_create(const char* configJson, const char** errorOut);
int            aicore_pipeline_execute(AICorePipeline pipeline, const unsigned char* imageData,
                                       int width, int height, int channels,
                                       AICoreResult* resultOut, const char** errorOut);
const char*    aicore_result_to_json(AICoreResult result);
void           aicore_result_free(AICoreResult result);
void           aicore_pipeline_destroy(AICorePipeline pipeline);
int            aicore_pipeline_get_state(AICorePipeline pipeline);
const char*    aicore_version();
```

JSON 结果格式：

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
            "bbox": {"x": 10, "y": 20, "w": 100, "h": 200},
            "anomaly_score": 0.87
        }
    ]
}
```

## 流水线引擎

基于 DAG（有向无环图）的流水线执行器：

- **节点类型**：`model`, `resize`, `normalize`, `nms`, `merge`, `composite`, `patchcore`
- **配置驱动**：JSON 配置文件定义节点拓扑
- **线程安全**：内置 ThreadPool，支持异步执行
- **EnginePool**：GPU 引擎池，支持多模型轮换

## PatchCore 算法

PatchCore（WACV 2022）是一种基于特征存储的工业异常检测方法：

1. **训练**：通过预训练 CNN backbone 提取正常样本的 patch 级特征 → Coreset 降采样 → 保存为 memory bank
2. **推理**：提取输入图片的 patch 特征 → Memory Bank 最近邻搜索 → 异常得分图

### 架构设计

```
config.json → MultiRoiNode
                 ├── IBackbone（工厂创建，三种实现）
                 ├── ROI 1 → MemoryBank 1 → heatmap 1
                 ├── ROI 2 → MemoryBank 2 → heatmap 2
                 └── ...
```

- **IBackbone 抽象**：统一接口，工厂 `CreateBackbone()` 根据 `backendType` 创建具体实现
- **多 ROI 共享 backbone**：同一模型只加载一次，不同 ROI 各有独立 MemoryBank
- **流式训练**：大图（>100MB）自动切换流式模式，逐批加载，峰值内存 ~1.2GB

## 知识图谱

项目知识图谱使用 [graphify](https://opencode.ai) 自动构建：

- `docs/knowledge-graph/`：静态代码架构图（DOT/PNG/JSON-LD）
- `graphify-out/`：完整语义图谱（684 节点，27 个社区）
  - `graph.html`：可交互 HTML 图谱
  - `graph.json`：JSON 格式图数据
  - `GRAPH_REPORT.md`：分析报告（God Nodes、Surprising Connections、Suggested Questions）

查询图谱（需 graphify 内置在 opencode 中）：

```
opencode /graphify 查询 社区结构
opencode /graphify 解释 MultiRoiNode 和 IBackbone 的关系
```

## 设计文档

| 文档 | 说明 |
|------|------|
| `docs/superpowers/specs/2026-06-10-aicore-engine-design.md` | 推理引擎设计 |
| `docs/superpowers/specs/2026-06-10-aicore-phase2-optimizer-design.md` | 模型优化器设计 |
| `docs/superpowers/specs/2026-06-10-aicore-phase3-trainer-design.md` | 训练模块设计 |
| `docs/superpowers/specs/2026-06-10-aicore-patchcore-design.md` | PatchCore 设计 |
| `docs/superpowers/specs/2026-06-10-aicore-scheduler-design.md` | GPU 调度器设计 |
| `docs/superpowers/specs/2026-06-10-aicore-inference-server-design.md` | 推理服务器设计 |
| `docs/superpowers/plans/2026-06-10-aicore-phase2-optimizer.md` | Phase 2 实现计划 |
| `docs/superpowers/plans/2026-06-10-aicore-phase3-trainer.md` | Phase 3 实现计划 |
| `docs/基于NVIDIA GPU工控机的工业检测纯C++ AI引擎技术方案.md` | 整体技术方案 |

## 开发

### 添加新节点类型

1. 继承 `IProcessor` 实现节点类
2. 在 `pipeline_builder.cpp` 中添加类型分支
3. 在包含 JSON 解析中添加参数
4. 添加 GTest 单元测试

### 项目规范

- C++17，命名空间 `aicore`
- 变量名 camelCase，函数名动词开头
- 注释使用中文
- DLL 导出宏：`AICORE_API` / `AICORE_OPTIMIZER_API` / `AICORE_TRAINER_API` / `AICORE_C_API`

## License

MIT
