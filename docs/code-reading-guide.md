# AIEngine 代码阅读手册

> 从零理解 AIEngine 工业检测 AI 推理引擎的完整代码结构。
> 适合第一次接触本项目的开发者。

---

## 目录

1. [项目全景图](#1-项目全景图)
2. [核心接口（必读 4 文件）](#2-核心接口必读-4-文件)
3. [数据流向（一条数据怎么跑完）](#3-数据流向一条数据怎么跑完)
4. [重点实现文件（3 个）](#4-重点实现文件3-个)
5. [三条主线模块](#5-三条主线模块)
6. [PatchCore 异常检测数据流](#6-patchcore-异常检测数据流)
7. [YOLO 训练数据流](#7-yolo-训练数据流)
8. [测试文件阅读顺序](#8-测试文件阅读顺序)
9. [设计模式速查](#9-设计模式速查)
10. [实用阅读技巧](#10-实用阅读技巧)
11. [推荐阅读路径（2小时版）](#11-推荐阅读路径2小时版)
12. [避坑指南](#12-避坑指南)

---

## 1. 项目全景图

```
AIEngine/
├── aicore/                          ← 主工程目录
│   ├── include/                     ← 公共头文件（只看这里了解接口）
│   │   ├── core/                    ← 核心接口
│   │   │   ├── types.h              ← Tensor, Status, DataType, MemoryType
│   │   │   ├── frame.h              ← Frame（cv::Mat + 元数据）
│   │   │   ├── processor.h          ← IProcessor（处理器抽象）
│   │   │   ├── pipeline.h           ← IPipeline（流水线抽象）
│   │   │   └── model_backend.h      ← IModelBackend（推理后端抽象）
│   │   ├── api/                     ← C API 入口
│   │   │   ├── aicore_api.h         ← 对外导出的 C 函数
│   │   │   └── scheduler_api.h      ← GPU 调度器 C API
│   │   ├── backend/                 ← 推理后端
│   │   │   └── backend_factory.h    ← 后端工厂
│   │   ├── patchcore/               ← PatchCore 异常检测模块
│   │   ├── pipeline/                ← DAG 节点实现
│   │   ├── preprocess/              ← 预处理节点
│   │   ├── postprocess/             ← 后处理节点（YOLO 解码、NMS）
│   │   ├── utils/                   ← 工具函数（绘图等）
│   │   ├── engine/                  ← 引擎核心 + 线程池
│   │   ├── server/                  ← 推理服务器
│   │   └── trainer/                 ← 训练模块
│   ├── src/                         ← 对应实现（.cpp/.cu）
│   ├── cli/                         ← 命令行入口（6个exe）
│   ├── gui/                         ← Qt 上位机
│   ├── tests/                       ← GTest 测试（286个）
│   ├── samples/                     ← 示例代码
│   ├── scripts/                     ← Python 辅助脚本
│   └── CMakeLists.txt               ← 构建配置
├── config/                          ← 配置文件
├── data/                            ← 数据/权重/测试输出
├── out/build/                       ← 编译产物
├── CMakePresets.json                ← CMake 预设
└── README.md                        ← 项目说明
```

### 编译产物一览

| 产物 | 说明 | 依赖 |
|------|------|------|
| `aicore.dll` | 推理核心 DLL | OpenCV, CUDA |
| `aicore_trainer.dll` | 训练模块 DLL | LibTorch |
| `aicore_optimizer.dll` | 模型优化 DLL | Python |
| `AICoreUI.exe` | Qt 上位机 | Qt5 |
| `aicore_tests.exe` | 单元测试 | GTest |

---

## 2. 核心接口（必读 4 文件）

按以下顺序读这 4 个头文件，建立心智模型。每个文件不超过 50 行。

### 2.1 `include/core/types.h` — 基础类型

```cpp
// 核心数据结构
struct Tensor {
    DataType dtype;           // kUInt8, kFloat32, kFloat16
    std::vector<int64_t> shape;  // 如 {N, C, H, W}
    MemoryType memory;        // kCPU, kGPU, kPinned
    void* data;               // ⚠️ 裸指针，不管理生命周期！
    size_t bytes;             // 字节数
    size_t allocId;           // 分配 ID（内存池追踪用）
};

struct Status {
    StatusCode code;          // OK, ErrorModelLoad, ErrorModelInfer...
    std::string message;
    operator bool() const;    // if(status) 判断成功
};

enum class MemoryType { kCPU, kGPU, kPinned };
enum class DataType     { kUInt8, kFloat32, kFloat16 };
```

**关键点**：
- `Status` 支持隐式 bool 转换：`if (s) { /* 成功 */ }`
- `Tensor::data` 是裸指针 — 调用方负责内存管理
- `StatusCode` 覆盖了推理全链路的错误场景

### 2.2 `include/core/frame.h` — 帧数据结构

```cpp
struct Frame {
    cv::Mat image;                        // OpenCV 图像矩阵
    uint64_t frameId;                     // 帧序号
    uint64_t timestamp;                   // 时间戳（ms）
    std::string sourceId;                 // 来源标识
    std::map<std::string, float> roiMap;  // ROI 分数映射

    Frame() = default;
    Frame(cv::Mat img, uint64_t id = 0);  // 自动记录时间戳
    bool empty() const;
    int width() const;
    int height() const;
};
```

**关键点**：
- `Frame` 是 pipeline 中传递的基本数据单元
- 内部持有 `cv::Mat`，拷贝时浅拷贝（引用计数）
- 构造函数自动记录 `steady_clock` 时间戳

### 2.3 `include/core/processor.h` — 处理器接口

```cpp
using NodeConfig = std::unordered_map<std::string, std::string>;

class IProcessor {
public:
    virtual ~IProcessor() = default;

    // 初始化（传入配置参数）
    virtual Status Init(const NodeConfig& config) = 0;
    // 处理：输入帧 → 输出帧
    virtual Status Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) = 0;
    // 节点标识
    virtual std::string GetName() const = 0;
    virtual std::string GetType() const = 0;

    // 可选：注入线程池
    virtual void SetThreadPool(ThreadPool* pool);
};
```

**关键点**：
- 所有处理逻辑（缩放、推理、NMS...）都实现此接口
- `Process` 接受帧列表、返回帧列表 — 支持多输入/多输出
- `GetType()` 返回字符串如 `"resize"`, `"model"`, `"nms"` — 用于 JSON 配置匹配

### 2.4 `include/core/pipeline.h` — 流水线接口

```cpp
enum class PipelineState {
    kCreated, kReady, kRunning, kStopped, kError
};

class IPipeline {
public:
    virtual ~IPipeline() = default;

    // 根据 JSON 配置构建拓扑
    virtual Status Build(const std::string& configJson) = 0;
    // 同步执行：输入一帧，阻塞等待结果
    virtual Status Execute(const Frame& input, Result& output) = 0;
    // 异步执行：输入一帧，回调返回结果
    virtual Status ExecuteAsync(const Frame& input,
                                std::function<void(const Result&)> callback) = 0;
    // 等待所有异步任务完成
    virtual Status WaitAll() = 0;
    // 停止处理
    virtual void Stop() = 0;
    // 查询状态
    virtual PipelineState GetState() const = 0;
    virtual std::string GetConfig() const = 0;
};
```

**关键点**：
- `Build()` 从 JSON 配置创建 DAG 拓扑
- `Execute()` 是同步入口 — 阻塞直到整条管线处理完
- `ExecuteAsync()` 是异步入口 — 通过回调接收结果

---

## 3. 数据流向（一条数据怎么跑完）

### 3.1 完整调用链

```
用户调用 C API
    │
    ├─ aicore_pipeline_create(configJson, &errorOut)
    │     │
    │     ├─ ① ConfigParser.Parse(configJson) → PipelineConfig
    │     │     解析 JSON 中的 nodes[] 和 edges[]
    │     │
    │     ├─ ② PipelineBuilder.Build(config, pipeline)
    │     │     遍历 config.nodes，按 type 创建 IProcessor
    │     │     遍历 config.edges，建立节点间连接
    │     │
    │     └─ ③ 返回 IPipeline* 不透明句柄
    │
    ├─ aicore_pipeline_execute(pipeline, imageData, w, h, c, &resultOut, &errorOut)
    │     │
    │     ├─ ① 将 raw pixels → cv::Mat → Frame
    │     │
    │     ├─ ② pipeline->Execute(Frame)
    │     │     │
    │     │     ├─ PipelineImpl::Execute()
    │     │     │     │
│     │     │     ├─ 按拓扑排序遍历节点
│     │     │     ├─ 每个节点：node->Process(inputs, outputs)
│     │     │     │     │
│     │     │     │     ├─ LetterboxNode::Process()  ← cv::resize + padding
│     │     │     │     ├─ ModelNode::Process()     ← 调后端推理
│     │     │     │     │     │
│     │     │     │     │     ├─ BackendFactory::Create(type)
│     │     │     │     │     ├─ backend->Infer(inputs, outputs)
│     │     │     │     │     │     │
│     │     │     │     │     │     ├─ ONNXRuntimeBackend::Infer()
│     │     │     │     │     │     │     Ort::Session::Run()
│     │     │     │     │     │     │
│     │     │     │     │     │     ├─ LibTorchBackend::Infer()
│     │     │     │     │     │     │     module_->forward(ivInputs)
│     │     │     │     │     │     │
│     │     │     │     │     │     └─ TensorRTBackend (stub)
│     │     │     │     │
│     │     │     │     ├─ YoloDecodeNode::Process() ← rawOutputs → 检测框
│     │     │     │     │     支持 v5/v8/v11，DFL 解算 + 坐标逆变换
│     │     │     │     │
│     │     │     │     ├─ NmsNode::Process()       ← NMSCommon() 非极大值抑制
│     │     │     │     │     NMSCommon（共享工具，IouBox 内联）
│     │     │     │     │
│     │     │     │     ├─ FusionNode::Process()    ← YOLO 检测 + PatchCore 异常评分
│     │     │     │     │     两路输入：[detections, original_image]
│     │     │     │     │     每检测框 → crop → backbone → MemoryBank NN → anomaly_score
│     │     │     │     │
│     │     │     │     └─ DrawRoiAnomaly() / DrawDetections() ← draw_utils.h 共享绘图工具
    │     │     │     │
    │     │     │     └─ 收集所有叶子节点输出 → Result
    │     │
    │     └─ ③ new Result(std::move(result)) → resultOut
    │
    └─ aicore_result_to_json(result) → JSON 字符串
          │
          ├─ 序列化 detections[], measurements, anomalyMap
          └─ 返回静态缓存的 C 字符串
```

### 3.2 JSON 配置示例

```json
{
    "pipeline": {
        "name": "yolo_patchcore_fusion",
        "max_concurrency": 4,
        "nodes": [
            {"id": "letterbox", "type": "letterbox", "params": {"width": "640", "height": "640"}},
            {"id": "detect",    "type": "model",     "model_path": "yolov8n.engine", "backend": "tensorrt"},
            {"id": "yolo_decode", "type": "yolo_decode",
             "params": {"confidence_threshold": "0.5", "num_classes": "80", "version": "v8"}},
            {"id": "nms",       "type": "nms",       "params": {"iou_threshold": "0.45", "confidence_threshold": "0.5"}},
            {"id": "fusion",    "type": "fusion",
             "params": {"backbone_type": "opencv_dnn", "model_path": "wideresnet.onnx",
                        "memory_bank_path": "bank.bin", "anomaly_threshold": "0.5",
                        "input_size": "224", "backbone_layers": "layer2,layer3"}}
        ],
        "edges": [
            {"from": "input",    "to": "letterbox"},
            {"from": "letterbox", "to": "detect"},
            {"from": "detect",   "to": "yolo_decode"},
            {"from": "yolo_decode", "to": "nms"},
            {"from": "nms",      "to": "fusion"}
        ]
    }
}
```

---

## 4. 重点实现文件（3 个）

### 4.1 `src/config/pipeline_builder.cpp` — 管线构建器

**职责**：根据 JSON 配置创建完整的处理器拓扑图。

**关键逻辑**：
```cpp
// 阶段一：按节点类型创建所有处理器
for (auto& pc : config.nodes) {
    std::shared_ptr<IProcessor> processor;
    
    if (pc.type == "model") {
        auto backend = BackendFactory::Create(pc.backend);
        ModelInfo info{...};
        backend->Load(info);
        processor = std::make_shared<ModelNode>(std::move(backend));
    }
    else if (pc.type == "letterbox")      processor = std::make_shared<LetterboxNode>();
    else if (pc.type == "yolo_decode")    processor = std::make_shared<YoloDecodeNode>();
    else if (pc.type == "nms")            processor = std::make_shared<NmsNode>();
    else if (pc.type == "patchcore")      processor = std::make_shared<PatchCoreNode>();
    else if (pc.type == "multi_roi")      processor = std::make_shared<MultiRoiNode>();
    else if (pc.type == "fusion")         processor = std::make_shared<FusionNode>();
    else if (pc.type == "merge")          processor = std::make_shared<MergeNode>();
    else if (pc.type == "composite")      processor = std::make_shared<CompositeNode>();
    else if (pc.type == "resize")         processor = std::make_shared<ResizeNode>();
    else if (pc.type == "normalize")      processor = std::make_shared<NormalizeNode>();
    else return Error("unknown node type: " + pc.type);
    
    processor->Init(pc.params);
    impl->AddNode(pc.id, processor, {});
}
// 阶段二：按 edges 建立节点间连接
for (auto& edge : config.edges) {
    impl->AddEdge(edge.from, edge.to);
}
```

**阅读要点**：
- 新增节点类型只需在 switch 中加一个分支
- 节点通过 `AddNode(id, processor, inputs)` 注册 DAG 拓扑
- `AddEdge(from, to)` 建立单向数据依赖
- `PipelineBuilder` 包含 15+ 个头文件 — 是设计意图，它需要知道所有节点类型

### 4.2 `src/pipeline/pipeline_impl.cpp` — DAG 执行引擎

**职责**：实现 `IPipeline` 接口，负责拓扑排序和执行调度。

**关键逻辑**：
```cpp
Status PipelineImpl::Execute(const Frame& input, Result& output) {
    // 1. 拓扑排序确定执行顺序
    auto order = TopologicalSort();
    
    // 2. 按序执行每个节点
    std::vector<Frame> currentFrames = {input};
    
    for (auto* node : order) {
        std::vector<Frame> nextFrames;
        node->Process(currentFrames, nextFrames);
        currentFrames = std::move(nextFrames);
    }
    
    // 3. 收集最终输出
    output = CollectResults(currentFrames);
    return Status{};
}
```

**阅读要点**：
- 拓扑排序保证 DAG 的正确执行顺序
- 每帧数据沿边流动，经过每个节点时被处理
- 支持并行执行（通过 `ThreadPool`）

### 4.3 `src/pipeline/model_node.cpp` — 推理节点

**职责**：封装模型推理，连接预处理和后处理。

**关键逻辑**：
```cpp
Status ModelNode::Process(const std::vector<Frame>& inputs,
                          std::vector<Frame>& outputs) {
    for (auto& frame : inputs) {
        // 1. 准备输入张量
        std::vector<Tensor> tensors;
        for (auto& f : frame.rois) {
            Tensor t = ConvertToTensor(f.image);
            tensors.push_back(t);
        }
        
        // 2. 调用后端推理
        std::vector<Tensor> results;
        auto s = backend_->Infer(tensors, results);
        if (!s) return s;
        
        // 3. 将结果转回 Frame
        for (auto& r : results) {
            Frame out = ConvertFromTensor(r);
            outputs.push_back(out);
        }
    }
    return Status{};
}
```

**阅读要点**：
- `cv::Mat → Tensor` 的格式转换是关键
- `Tensor → cv::Mat` 的反向转换涉及内存拷贝
- 支持多 ROI 推理（一张图多个检测区域）

---

## 5. 三条主线模块

### 主线 A：推理管线（必读）

| 文件 | 说明 |
|------|------|
| `include/api/aicore_api.h` | C API 入口，14 个导出函数 |
| `src/api/aicore_api.cpp` | C API 实现 |
| `src/config/pipeline_builder.cpp` | JSON → DAG 拓扑 |
| `src/pipeline/pipeline_impl.cpp` | DAG 执行引擎 |
| `src/pipeline/model_node.cpp` | 推理节点 |
| `src/backend/backend_factory.cpp` | 后端工厂 |

### 主线 B：PatchCore 异常检测（按需）

| 文件 | 说明 |
|------|------|
| `include/patchcore/patchcore_node.h` | PatchCoreNode 接口 |
| `include/patchcore/memory_bank.h` | MemoryBank 特征存储 |
| `include/patchcore/coreset_sampler.h` | Coreset 降采样 |
| `include/patchcore/backbone.h` | IBackbone 抽象 |
| `src/patchcore/nn_search.cu` | CUDA 最近邻搜索 |
| `src/patchcore/tiered_memory_bank.cpp` | GPU/CPU 分层存储 |

### 主线 C：YOLO 训练（按需）

| 文件 | 说明 |
|------|------|
| `include/trainer/model/yolo_model.h` | YOLOv8 架构定义 |
| `src/trainer/model/yolo_model.cpp` | YOLOv8 前向传播 |
| `include/trainer/model/yolo_loss.h` | 损失函数 |
| `src/trainer/data/yolo_data.cpp` | 数据增强 |
| `include/trainer/training/yolo_trainer.h` | 训练循环 |

### 主线 D：YOLO 推理管线 + PatchCore 融合

| 文件 | 说明 |
|------|------|
| `include/preprocess/letterbox_node.h` | Letterbox 预处理（等比例缩放+填充） |
| `include/postprocess/yolo_decode_node.h` | YOLO 解码（v5/v8/v11，DFL 解算） |
| `include/postprocess/nms_node.h` | NMS 节点 |
| `include/utils/draw_utils.h` | 绘图工具（DrawDetections / DrawRoiAnomaly） |
| `include/pipeline/fusion_node.h` | YOLO+PatchCore 融合节点 |
| `include/patchcore/patchcore_node.h` | PatchCoreNode 接口 |
| `include/patchcore/memory_bank.h` | MemoryBank 特征存储 |
| `include/patchcore/backbone.h` | IBackbone 抽象 |
| `src/postprocess/nms_common.cpp` | NMS 共享工具（IouBox + NMSCommon） |
| `src/preprocess/letterbox_node.cpp` | 等比例缩放 + 填充到目标尺寸 |
| `src/postprocess/yolo_decode_node.cpp` | rawOutputs → 检测框解码 |
| `src/pipeline/fusion_node.cpp` | 双路输入：detections + image → anomaly_score |

---

## 6. PatchCore 异常检测数据流

### 6.1 训练阶段

```
PatchCoreTrain.exe
    │
    ├─ FolderDataset.Load(normal_images/)
    │     扫描文件夹，加载所有正常样本图片
    │
    ├─ Backbone.Extract(image) → features
    │     用预训练 CNN（ResNet/WideResNet）提取 patch 级特征
    │     截取指定中间层（layer2, layer3）的输出
    │
    ├─ CoresetSampler.FastSample(pool, fraction=0.1)
    │     贪心降采样：选最分散的 10% 特征点
    │     时间复杂度 O(n²) → 优化为 O(n log n)
    │
    └─ MemoryBank.Store(features)
          写入 .bin 文件
          格式：[magic(4B)][version(4B)][norm_params(2x4B)][features(NxD)]
```

### 6.2 推理阶段

```
RoiInfer.exe / patchcore_node
    │
    ├─ Backbone.Extract(query_image) → query_features
    │
    ├─ MemoryBank.Load(memory_bank.bin)
    │
    ├─ NearestNeighbor(query_features, memory_bank)
    │     GPU CUDA kernel 并行计算 L2 距离
    │     返回每个 query patch 到 memory bank 的最小距离
    │
    ├─ ComputeAnomalyMap(distances, image_size)
    │     距离图 → 异常热力图（高分 = 异常）
    │     可选：高斯平滑、阈值过滤
    │
    └─ ColorizeAnomalyMap(heatmap) → cv::Mat
          距离值 → RGB 颜色（冷色=正常，暖色=异常）
```

### 6.3 关键数据结构

```cpp
// MemoryBank — 特征存储
struct MemoryBank {
    std::vector<float> features;    // N × D 展平数组
    int64_t numFeatures;            // N
    int64_t featureDim;             // D
    float normMean;                 // 归一化参数
    float normStd;                  // 归一化参数
    bool isGpuLoaded;               // 是否已加载到 GPU
};

// PatchCoreNode — 异常检测节点
class PatchCoreNode : public IProcessor {
    std::shared_ptr<IBackbone> backbone_;
    std::shared_ptr<MemoryBank> memoryBank_;
    float anomalyThreshold_;
};
```

---

## 7. YOLO 训练数据流

### 7.1 训练循环

```
AICoreTrainer.exe
    │
    ├─ YOLOTrainer.Init(config)
    │     ├─ YOLODataset.Load(train_images/, train_labels/)
    │     │     解析 COCO 格式标注
    │     │     建立 image ↔ label 配对
    │     │
    │     ├─ DataLoader(dataset, batch_size=16, shuffle=true)
    │     │     多线程数据加载
    │     │
    │     └─ YOLOv8Model.Build(config)
    │           构建 YOLOv8-n 架构
    │
    ├─ TrainingLoop.Run()
    │     for epoch = 0 to epochs-1:
    │         for batch in DataLoader:
    │             predictions = model.Forward(batch.images)
    │             loss = yolo_loss(predictions, batch.labels)
    │             loss.backward()
    │             optimizer.step()
    │             
    │             if epoch % save_interval == 0:
    │                 checkpoint.Save(epoch, model)
    │             
    │             progress_callback(loss, epoch, batch)
    │
    └─ YOLOTrainer.Train() 返回
```

### 7.2 YOLOv8 模型架构

```
YOLOv8Model (YOLOv8-n 微型版)
    │
    ├─ Backbone（主干网络）
    │   ├─ ConvBnSiLU(3→16, s=2)     P1: 320×320
    │   ├─ ConvBnSiLU(16→32, s=2)    P2: 160×160
    │   ├─ C2f(32, 32, shortcut)
    │   ├─ ConvBnSiLU(32→64, s=2)    P3: 80×80
    │   ├─ C2f(64, 64, shortcut×2)
    │   ├─ ConvBnSiLU(64→128, s=2)   P4: 40×40
    │   ├─ C2f(128, 128, shortcut×2)
    │   ├─ ConvBnSiLU(128→128, s=2)  P5: 20×20
    │   ├─ C2f(128, 128, shortcut)
    │   └─ SPPF(128, 128, k=5)
    │
    ├─ Neck（颈部网络，FPN+PAN）
    │   ├─ Upsample → concat(P4) → C2f(256→64)
    │   ├─ Upsample → concat(P3) → C2f(128→32)
    │   ├─ ConvBnSiLU(32→64, s=2) → concat(neck_P4) → C2f(128→64)
    │   └─ ConvBnSiLU(64→128, s=2) → concat(neck_P5) → C2f(256→128)
    │
    └─ Detect Head（检测头）
        ├─ cvReg[i]: Conv2d(ch[i]→4×16)    边界框回归
        ├─ cvCls[i]: Conv2d(ch[i]→nc)      类别分类
        └─ DFL: Distribution Focal Loss     边界框细化
```

### 7.3 关键组件

```cpp
// C2f — Cross Stage Partial with 2 convolutions
struct C2fImpl : torch::nn::Module {
    torch::nn::Conv2d cv1;   // 1×1 卷积降维
    torch::nn::Conv2d cv2;   // 1×1 卷积升维
    torch::nn::ModuleList m; // Bottleneck 列表
};

// Bottleneck — 残差块
struct BottleneckImpl : torch::nn::Module {
    ConvBnSiLU cv1;   // Conv + BN + SiLU
    ConvBnSiLU cv2;   // Conv + BN + SiLU
    bool add;         // 是否 shortcut 相加
};

// Detect — 解耦检测头
struct DetectImpl : torch::nn::Module {
    int nc;                    // 类别数
    int nl;                    // 检测尺度数（3）
    std::vector<torch::nn::Conv2d> cvReg;  // 回归头
    std::vector<torch::nn::Conv2d> cvCls;  // 分类头
    torch::Tensor proj;        // DFL 投影矩阵
};
```

---

## 8. 测试文件阅读顺序

测试是最好的文档——它们展示了接口的正确使用方式。

### 8.1 推荐顺序

| 顺序 | 文件 | 行数 | 学习重点 |
|------|------|------|----------|
| 1 | `test_pipeline.cpp` | ~100 | DAG 构建和执行 |
| 2 | `test_processor.cpp` | ~50 | IProcessor 接口用法 |
| 3 | `test_backend_factory.cpp` | ~50 | 后端工厂模式 |
| 4 | `test_inference_server.cpp` | ~170 | 推理服务器生命周期 |
| 5 | `test_letterbox_node.cpp` | ~40 | YOLO 预处理（Letterbox） |
| 6 | `test_yolo_decode_node.cpp` | ~100 | YOLO 解码 + NMS 测试 |
| 7 | `test_nms_node.cpp` | ~120 | NMSCommon（IouBox/NMSCommon/NmsNode 测试） |
| 8 | `test_fusion_node.cpp` | ~50 | YOLO+PatchCore 融合节点 |
| 9 | `test_patchcore.cpp` | ~750 | PatchCore 全流程（最大） |
| 10 | `test_yolo_model.cpp` | ~60 | YOLO 前向传播形状验证 |
| 11 | `test_yolo_trainer.cpp` | ~166 | YOLO 训练循环 |

### 8.2 如何读测试

```cpp
// 读测试时关注这三个部分：
TEST(Suite, TestCaseName) {
    // 1. 准备（Setup）— 创建对象、加载数据
    auto pipeline = CreateTestPipeline();
    auto frame = MakeTestFrame();
    
    // 2. 执行（Act）— 调用被测试的接口
    auto result = pipeline->Execute(frame);
    
    // 3. 验证（Assert）— 检查结果是否符合预期
    EXPECT_EQ(result.detections.size(), 1);
    EXPECT_TRUE(result.status == StatusCode::OK);
}
```

---

## 9. 设计模式速查

### 9.1 工厂模式

```cpp
// 后端工厂 — 根据字符串创建不同后端
class BackendFactory {
    static std::unique_ptr<IModelBackend> Create(BackendType type);
};
// 调用：auto backend = BackendFactory::Create(BackendType::kONNXRuntime);
```

**为什么用**：解耦后端创建逻辑，新增后端只需加 switch 分支。

### 9.2 策略模式

```cpp
// IProcessor 是策略接口
class IProcessor { virtual Status Process(...) = 0; };
// 具体策略：ResizeNode, NormalizeNode, ModelNode, NmsNode...
```

**为什么用**：pipeline 不需要知道每个节点的具体实现，只需调用 `Process()`。

### 9.3 DAG 编排

```json
{
    "nodes": [{"id": "a", "type": "resize"}, {"id": "b", "type": "model"}],
    "edges": [{"from": "a", "to": "b"}]
}
```

**为什么用**：配置驱动，无需改代码就能改变处理流程。

### 9.4 RAII 资源管理

```cpp
// 智能指针管理生命周期
std::unique_ptr<IPipeline> pipeline;        // pipeline 独占所有权
std::shared_ptr<IProcessor> processor;      // 节点可被多个地方引用
std::shared_ptr<EnginePool> pool;           // 引擎池共享
```

---

## 10. 实用阅读技巧

### 10.1 从入口倒推

不要从头文件开始读。先找入口，再顺藤摸瓜：

```powershell
# 看 C API 怎么被调用的
grep -r "aicore_pipeline" aicore/

# 看测试怎么用 IPipeline
grep -r "Build\|Execute" aicore/tests/

# 看 CLI 怎么启动
cat aicore/cli/patchcore_train_main.cpp
```

### 10.2 用 LSP 跳转

- `Ctrl + 点击` 函数名 → 跳转到定义
- `IProcessor::Process` → 找到 `ResizeNode::Process` → 看具体实现
- `BackendFactory::Create` → 看三种后端的工厂模式

### 10.3 追踪数据流

```
Frame (cv::Mat) → detections + rawOutputs + original image
  │
  ├─ 进入 LetterboxNode::Process()
  │   └─ 等比例缩放 + 填充到 640×640，记录 scale/pad 到 roiMap
  │
  ├─ 进入 ModelNode::Process()
  │   ├─ cv::Mat → Tensor (NCHW, float32)
  │   ├─ backend->Infer(Tensor → Tensor)
  │   └─ rawOutputs 写入 Frame
  │
  ├─ 进入 YoloDecodeNode::Process()
  │   ├─ rawOutputs → DFL 解算 / sigmoid
  │   ├─ 坐标逆变换（letterbox 反向）
  │   └─ → detections（带 confidence 和 bbox）
  │
  ├─ 进入 NmsNode::Process()
  │   ├─ NMSCommon() 按类别分组 + IouBox 去重
  │   └─ → 过滤后的 detections
  │
  ├─ 进入 FusionNode::Process() [需要两路输入]
  │   ├─ inputs[0].detections + inputs[1].image
  │   ├─ 每框裁剪 → backbone.Extract → MemoryBank NN
  │   └─ → enrich detections with anomaly_score + is_anomaly
  │
  └─ DrawDetections() [draw_utils.h]
      └─ 在原图上叠加检测框 + 标签 + 异常得分
```

### 10.4 理解错误处理

```cpp
// 所有可能失败的函数都返回 Status
Status s = pipeline->Execute(frame, result);
if (!s) {
    // 失败：s.code 是错误码，s.message 是描述
    fprintf(stderr, "Error: %s\n", s.message.c_str());
    return -1;
}

// Status 支持隐式 bool 转换
if (s) { /* 成功 */ }
if (!s) { /* 失败 */ }
```

---

## 11. 推荐阅读路径（2小时版）

### Day 1（30分钟）：接口层

```
只读头文件，不读实现：
  1. include/core/types.h
  2. include/core/frame.h
  3. include/core/processor.h
  4. include/core/pipeline.h
  5. include/backend/backend_factory.h

目标：能画出数据在 pipeline 中流动的草图。
```

### Day 2（45分钟）：核心实现

```
读实现文件，理解数据怎么流转：
  1. src/config/pipeline_builder.cpp
  2. src/pipeline/pipeline_impl.cpp
  3. src/pipeline/model_node.cpp

目标：理解 JSON 配置 → DAG 拓扑 → 节点执行的完整链路。
```

### Day 2.5（30分钟）：YOLO 推理管线

```
  1. include/preprocess/letterbox_node.h → src/preprocess/letterbox_node.cpp
  2. include/postprocess/yolo_decode_node.h → src/postprocess/yolo_decode_node.cpp
  3. src/postprocess/nms_common.cpp（NMSCommon 共享工具）
  4. include/utils/draw_utils.h

目标：理解 YOLO 检测 pipeline 完整数据流。
```

### Day 2.6（20分钟）：PatchCore 融合

```
  1. include/pipeline/fusion_node.h → src/pipeline/fusion_node.cpp
  2. include/patchcore/backbone.h
  3. include/patchcore/memory_bank.h
  4. src/patchcore/multi_roi_node.cpp（DrawRoiAnomaly → draw_utils 改造点）

目标：理解 YOLO 检测框 + PatchCore 异常评分的融合方式。
```

### Day 3（30分钟）：后端实现

```
理解推理后端怎么接入：
  1. src/backend/backend_factory.cpp
  2. src/backend/onnxruntime_backend.cpp (174行)
  3. src/backend/backend_factory.cpp 中的 LibTorchBackend (内联)

目标：理解 Tensor 格式转换和推理调用。
```

### Day 4（15分钟）：验证理解

```
跑测试，确认理解正确：
  .\out\build\aicore\tests\Release\aicore_tests.exe

看到 286/286 PASS 就对了。
```

---

## 12. 避坑指南

### ❌ 不要从这里开始

| 文件 | 原因 |
|------|------|
| `gui/training_dialog.cpp` (493行) | Qt GUI 代码，绕路 |
| `gui/main_window.cpp` | 同上 |
| `cli/*.cpp` | 只是 `Init() → Run()` 的壳 |
| `scripts/*.py` | 训练辅助脚本，不是引擎核心 |
| `patchcore/` 内部实现 | 独立模块，先懂 pipeline 再看 |

### ✅ 应该从这里开始

```
types.h → frame.h → processor.h → pipeline.h
    ↓
pipeline_builder.cpp → pipeline_impl.cpp → model_node.cpp
    ↓
backend_factory.cpp → onnxruntime_backend.cpp
    ↓
跑测试验证
```

### ⚠️ 常见陷阱

1. **`Tensor::data` 是裸指针**
   - 不管理内存生命周期
   - 调用方必须确保数据在使用期间有效
   - 后端 `Infer()` 返回的 `Tensor` 数据由后端分配，调用方负责释放

2. **`Status` 不是异常**
   - 不抛出 `std::exception`
   - 通过返回值传递错误
   - 用 `if (!s)` 判断失败

3. **`Frame` 拷贝是浅拷贝**
   - `cv::Mat` 内部使用引用计数
   - 拷贝 `Frame` 不会复制图像数据
   - 修改 `frame.image` 会影响所有引用

4. **`pipeline_builder.cpp` 包含 15+ 个头文件**
   - 这是设计意图，不是 bug
   - 构建器需要知道所有节点类型（model / letterbox / yolo_decode / nms / fusion / merge...）
   - 新增节点只需加 switch 分支 + include

5. **`onnxruntime_backend.cpp` 中的 `new float[]`**
   - 已修复为 memcpy 拷贝
   - 调用方通过 `allocId` 判断是否需要释放
   - `allocId == 1` 表示 backend 分配，需 `delete[]`

---

## 附录：常用命令

### 构建

```powershell
# 配置（首次）
cmake --preset default

# 构建
cmake --build --preset debug
cmake --build --preset release

# 测试
ctest --preset debug
```

### 运行

```powershell
# 单元测试
.\out\build\aicore\tests\Release\aicore_tests.exe

# PatchCore 训练
.\out\build\Release\PatchCoreTrain.exe --data ./normal_images/ --output memory_bank.bin

# 多 ROI 推理
.\out\build\Release\RoiInfer.exe --config ../config/config_rois.json --image test.jpg

# Qt 上位机
.\out\build\Release\AICoreUI.exe
```

### 调试

```powershell
# 只跑某个测试套件
.\out\build\aicore\tests\Release\aicore_tests.exe --gtest_filter="PipelineBuilderTest.*"

# 看详细输出
.\out\build\aicore\tests\Release\aicore_tests.exe --gtest_print_time=1

# 跑 YOLO 相关测试（含推理管线）
.\out\build\aicore\tests\Release\aicore_tests.exe --gtest_filter="*YOLO*:*Yolo*:*Letterbox*:*Nms*:*NMS*"

# 跑 FusionNode 测试
.\out\build\aicore\tests\Release\aicore_tests.exe --gtest_filter="*Fusion*"

# 跑 PatchCore 测试
.\out\build\aicore\tests\Release\aicore_tests.exe --gtest_filter="*PatchCore*:*ROI*:*MemoryBank*:*Roi*"
```
