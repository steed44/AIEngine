# PatchCore 异常检测集成设计文档

## 目标

在现有流水线引擎中集成 PatchCore 异常检测算法，支持训练（从正常样本构建 memory bank）和推理（异常得分图 + 图像级得分）。

## 功能范围

- 训练：从 IDataset 或本地文件夹读取正常图像，提取特征，coreset 降采样，保存 memory bank
- 推理：作为流水线节点 `"patchcore"`，加载 backbone + memory bank，输出异常得分图
- Backbone 支持：OpenCV::dnn（ONNX 模型）和 IModelBackend 接口两种方式

## 不包含

- 多 GPU 训练加速
- 在线学习 / memory bank 增量更新
- 其他异常检测算法（SPADE, PaDiM 等）

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                         Pipeline DAG                         │
│  [input] → [resize] → [patchcore] → [output]                │
│                             │                                │
│                    ┌────────┴────────┐                       │
│                    │  PatchCoreNode   │                       │
│                    │  (IProcessor)    │                       │
│                    │                  │                       │
│              ┌─────┴─────┐    ┌──────┴──────┐                │
│              │  Backbone  │    │ MemoryBank   │                │
│              │ cv::dnn /  │    │ NN search    │                │
│              │ IModelBkd  │    │ + scoring    │                │
│              └───────────┘    └─────────────┘                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Training Pipeline                       │
│  [IDataset/Folder] → [Backbone] → [FeaturePool] → [Coreset] → [MemoryBank.bin] │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
include/patchcore/
  patchcore_node.h         ← PatchCoreNode (IProcessor)
  memory_bank.h            ← MemoryBank（特征存储 + NN 搜索）
  coreset_sampler.h        ← CoresetSampler（贪心降采样）
  patchcore_trainer.h      ← PatchCoreTrainer（训练入口）

src/patchcore/
  patchcore_node.cpp
  memory_bank.cpp
  coreset_sampler.cpp
  patchcore_trainer.cpp
```

## 核心类设计

### PatchCoreNode

继承 `IProcessor`，类型字符串 `"patchcore"`。

```cpp
class PatchCoreNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    std::unique_ptr<IModelBackend> backend_;  // IModelBackend 方式
    cv::dnn::Net net_;                        // OpenCV dnn 方式
    MemoryBank memoryBank_;
    std::vector<std::string> outputLayerNames_;
    bool useOpenCVDnn_ = false;
    std::string name_;
};
```

`Init()` 参数：
| 参数 | 说明 |
|------|------|
| `backbone` | backbone 类型：`"opencv_dnn"` 或 `"model_backend"` |
| `model_path` | ONNX 模型路径（opencv_dnn 方式） |
| `backend_type` | IModelBackend 方式的后端类型 |
| `memory_bank_path` | MemoryBank 文件路径 |
| `input_size` | 输入尺寸，如 `"224"` |
| `backbone_layers` | 提取特征的层名，逗号分隔（如 `"layer2,layer3"`） |
| `anomaly_threshold` | 异常判定阈值；节点在 NodeResult 中标注 is_anomaly，热力图颜色映射基于此阈值 |

`Process()` 输出：
- `outputs[0]` 为异常得分热力图（cv::Mat，float32，与原图同尺寸），由 feature map 经双线性插值上采样得到
- outputs 中的 Frame 携带 `NodeResult` 元数据，包含图像级 anomaly_score（取热力图最大值）

### MemoryBank

```cpp
struct PatchFeature {
    std::vector<float> features;
    int layerIdx;
    int patchRow, patchCol;
};

class MemoryBank {
public:
    bool Load(const std::string& path);
    bool Save(const std::string& path) const;
    void Build(const std::vector<PatchFeature>& features);
    void CoresetReduce(size_t targetSize, CoresetSampler& sampler);
    // 返回最近邻在 bank_ 中的索引，dist 为 L2 距离（即 anomaly score）
    size_t NearestNeighbor(const std::vector<float>& query, float& distOut) const;
    // 对 queries 逐 patch 做 NN 搜索 → 重排为 (H_feat, W_feat) 得分图 → 上采样到 (imgH, imgW)
    std::vector<float> ComputeAnomalyMap(const std::vector<PatchFeature>& queries,
                                          int imgH, int imgW) const;

private:
    std::vector<PatchFeature> bank_;
};
```

### CoresetSampler

贪心 coreset 采样：迭代选择与已选集合最远距离最大的样本。

```cpp
class CoresetSampler {
public:
    std::vector<size_t> Sample(const std::vector<PatchFeature>& pool,
                                size_t targetSize);
};
```

### PatchCoreTrainer

```cpp
class PatchCoreTrainer {
public:
    // 通过 IDataset 训练
    Status Train(IDataset& dataset, const std::string& modelPath,
                 const std::string& outputPath,
                 const TrainConfig& cfg);
    // 通过文件夹训练
    Status TrainFromFolder(const std::string& folderPath,
                           const std::string& modelPath,
                           const std::string& outputPath,
                           const TrainConfig& cfg);
};

struct TrainConfig {
    int inputSize = 224;
    std::vector<std::string> backboneLayers = {"layer2", "layer3"};
    double coresetFraction = 0.1;      // coreset 保留比例（相对于 maxFeatures）
    size_t maxFeatures = 100000;        // 从全部特征中随机采样 maxFeatures 个，再进行 coreset
    std::string backboneType = "opencv_dnn";  // opencv_dnn / model_backend
};
```

### FolderDataset

内部工具类，实现 `IDataset`，扫描文件夹下图片文件。

```cpp
class FolderDataset : public IDataset {
public:
    Status Load(const std::string& path) override;  // path = 文件夹路径
    size_t Size() const override;
    Sample Get(size_t index) override;
    int NumClasses() const override { return 1; }
private:
    std::vector<Sample> samples_;
};
```

## CMake 集成

在 `CMakeLists.txt` 中添加新源文件到 `aicore.dll`（主推理 DLL）：

```cmake
set(AICORE_SOURCES
    ...
    src/patchcore/patchcore_node.cpp
    src/patchcore/memory_bank.cpp
    src/patchcore/coreset_sampler.cpp
    src/patchcore/patchcore_trainer.cpp
)
```

OpenCV::dnn 已经包含在现有 `OpenCV_LIBS` 中，无需额外依赖。

## Pipeline Builder 扩展

在 `pipeline_builder.cpp` 中添加分支：

```cpp
if (type == "patchcore") {
    auto node = std::make_shared<PatchCoreNode>();
    // 从 pc.params 初始化
    processor = node;
}
```

## JSON 配置示例

### 推理

```json
{
    "pipeline": {
        "name": "patchcore_infer",
        "nodes": [
            {
                "id": "input",
                "type": "input",
                "params": {}
            },
            {
                "id": "patchcore",
                "type": "patchcore",
                "params": {
                    "backbone": "opencv_dnn",
                    "model_path": "models/wideresnet.onnx",
                    "memory_bank_path": "models/memory_bank.bin",
                    "input_size": "224",
                    "backbone_layers": "layer2,layer3",
                    "anomaly_threshold": "0.5"
                }
            }
        ],
        "edges": [
            {"from": "input", "to": "patchcore"}
        ]
    }
}
```

### 训练

```json
{
    "patchcore_train": {
        "data_source": "folder",
        "folder_path": "data/normal/",
        "model_path": "models/wideresnet.onnx",
        "output_path": "models/memory_bank.bin",
        "input_size": 224,
        "backbone_layers": "layer2,layer3",
        "coreset_fraction": 0.1
    }
}
```

## 训练调用入口

训练通过独立 CLI 可执行文件 `PatchCoreTrain.exe` 调用，链接 `aicore.dll`，直接使用 `PatchCoreTrainer` C++ 类：

```bash
PatchCoreTrain --data ./normal_images/ --model wideresnet.onnx --output memory_bank.bin
```

CLI 入口文件：`cli/patchcore_train_main.cpp`。JSON 训练配置由 CLI 内部解析后填入 `TrainConfig`，不经过 C API。

## AICoreUI 扩展

AICoreUI 需要支持显示异常热力图：
- 检测到 output 中有 `anomaly_map` 时，用颜色映射（jet colormap）叠加到原图
- 显示 anomaly_score 数值

## 测试

- `MemoryBank`：构建 → 保存 → 加载 → 查询的往返测试
- `CoresetSampler`：输入 N 个随机特征 → 降采样到 M 个 → 验证 M <= N
- `PatchCoreNode`：Init → Process → 验证输出包含 anomaly_map
- `PatchCoreTrainer`：用随机图像模拟训练 → 验证输出文件存在
- `FolderDataset`：从测试图片目录加载 → 验证数量正确
