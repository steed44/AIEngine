# Phase 3: 训练模块 设计文档

## 概述

训练模块提供基于 LibTorch 的 C++ 训练能力，支持在工控机上直接完成模型的全量训练或增量微调。核心能力包括：多格式数据集加载、可组合数据增强流水线、多模型串行训练流水线、运行时自适应调度，以及训练完成后的 ONNX/TensorRT 一键导出。

### YOLOv8 训练策略

YOLOv8 没有官方的 LibTorch C++ 训练实现。本文档采用混合方案：

- **全量训练 / 完整微调**：通过 Phase 2 的 Python 嵌入层调用 PyTorch 完成（复用 `PythonEmbedding`），C++ 端负责数据集准备和训练调度
- **简单模型 / 自定义轻量模型**：直接用 LibTorch C++ API 训练（Faster R-CNN、自定义 CNN 分类器等）

### 多 GPU 策略

LibTorch 2.1.0 中 `DataParallel` 已废弃，Windows 不支持 `DistributedDataParallel`。替代方案：

- **手动梯度同步**：每个 GPU 独立前向+反向，收集梯度后 `all_reduce` 同步，手动更新权重
- **单 GPU 多卡时间片**：轮转使用各 GPU 完成不同 batch
- **单 GPU 单卡（推荐）**：工控机场景以单卡为主

## 总体架构

```
CLI 层 (ModelTrainer.exe)
    │
Trainer DLL (aicore_trainer.dll)           ← 链接 aicore_optimizer.lib 复用导出能力
    ├── TrainingPipeline 编排器
    ├── 数据层
    │   ├── IDataset, DataLoader, AugmentationPipeline
    ├── 训练层
    │   ├── TrainingLoop (LibTorch)
    │   ├── ModelFactory (Faster R-CNN / 自定义)
    │   ├── PythonTrainer (YOLOv8 通过 Phase 2 嵌入)
    │   └── TrainingScheduler (自适应)
    ├── 验证层
    │   ├── Validator (mAP / Precision / Recall)
    │   └── EarlyStopping
    └── 导出层
        └── ModelExporter → 调用 aicore_optimizer.dll
```

## 核心接口

### IDataset — 数据集抽象

```cpp
class IDataset {
public:
    virtual ~IDataset() = default;
    virtual size_t size() const = 0;
    virtual Status getItem(size_t index, TrainSample& sample) = 0;
    virtual std::vector<std::string> classNames() const = 0;
};

struct TrainSample {
    cv::Mat image;
    std::vector<BBox> boxes;
    std::vector<int> labels;
    std::string imagePath;
    int width, height;
};
```

### DataLoader — 多线程数据加载

```cpp
class DataLoader {
public:
    DataLoader(std::unique_ptr<IDataset> dataset,
               int batchSize, int numWorkers, bool shuffle);

    void start();              // 启动预加载线程
    bool nextBatch(std::vector<TrainSample>& batch);  // 取一个 batch
    void stop();
    size_t epochSize() const;  // 一个 epoch 的 batch 数

private:
    ThreadSafeQueue<std::vector<TrainSample>> queue_;
    std::vector<std::thread> workers_;
};
```

### IAugmentation — 数据增强接口

```cpp
class IAugmentation {
public:
    virtual ~IAugmentation() = default;
    virtual Status process(TrainSample& sample) = 0;
};

// 内置增强
class RandomFlip : public IAugmentation;
class RandomCrop : public IAugmentation;
class ColorJitter : public IAugmentation;
class Mosaic : public IAugmentation;     // YOLO 专用
class RandomPerspective : public IAugmentation;

class AugmentationPipeline {
    std::vector<std::unique_ptr<IAugmentation>> steps_;
public:
    void addStep(std::unique_ptr<IAugmentation> step);
    Status process(TrainSample& sample);
};
```

### TrainingLoop — LibTorch 训练循环

```cpp
class TrainingLoop {
public:
    TrainingLoop(const TrainConfig& cfg);

    Status run(DataLoader& trainLoader, DataLoader* valLoader,
               ITrainCallback* callback);

private:
    torch::nn::AnyModule model_;
    torch::optim::Optimizer* optimizer_ = nullptr;
    std::unique_ptr<torch::optim::lr_scheduler::LRScheduler> scheduler_;
    torch::Device device_{torch::kCPU};

    Status trainEpoch(int epoch, DataLoader& loader);
    Status validate(DataLoader& loader, float& map);
    Status saveCheckpoint(int epoch, float loss, float map);
    Status loadCheckpoint(const std::string& path);
};
```

### PythonTrainer — YOLO 等模型训练（通过 Python 嵌入）

```cpp
class PythonTrainer {
    PythonEmbedding* py_;
public:
    Status train(const TrainConfig& cfg, ITrainCallback* callback);

private:
    static Status callbackProxy(void* userData, int epoch,
                                 float loss, float map);
};
```

内部调用 Python 脚本进行训练，C++ 端通过回调接收进度。

### ModelFactory — 模型工厂

```cpp
class ModelFactory {
public:
    static std::unique_ptr<torch::nn::AnyModule> create(
        const std::string& type, int numClasses,
        const std::string& pretrainedPath);
    
    static std::vector<std::string> supportedModels();
};
```

内置支持：`faster_rcnn`（TorchVision C++）、`custom_cnn`、`resnet_classifier`。

YOLOv8 等复杂模型通过 `PythonTrainer` 训练。

### CheckpointManager

```cpp
class CheckpointManager {
public:
    CheckpointManager(const std::string& outputDir, int saveInterval,
                      bool saveBestOnly);

    Status save(const std::string& tag, int epoch,
                torch::nn::Module& model,
                torch::optim::Optimizer& opt,
                float metrics);

    Status loadLatest(torch::nn::Module& model,
                      torch::optim::Optimizer& opt,
                      int& epoch);

    Status loadBest(torch::nn::Module& model,
                    torch::optim::Optimizer& opt,
                    float& bestMetrics);

private:
    std::string outputDir_;
    int saveInterval_;
    bool saveBestOnly_;
    float bestMetrics_ = 0;
    std::string bestPath_;

    std::string makePath(const std::string& tag) const;
};
```

命名规则：`checkpoint_epoch_050_loss_0.123.pt` / `model_best.pt`

## 混合精度与梯度累积

```cpp
struct TrainConfig {
    // 原字段...
    // 新增
    bool enableAmp = true;           // 混合精度 (automatic mixed precision)
    int gradientAccumulation = 1;    // 梯度累积步数
    float gradClipNorm = 0.0f;       // 梯度裁剪（0=禁用）
};
```

- 混合精度：LibTorch 的 `torch::amp::autocast` + `torch::amp::GradScaler`
- 梯度累积：`optimizer->zero_grad()` 每 `accumulation` 步才执行，等效放大 batch_size
- 适用于工控机显存受限场景

## 验证与早停

```cpp
class Validator {
public:
    Status evaluate(DataLoader& valLoader,
                    torch::nn::AnyModule& model,
                    ValidationResult& result);

    bool shouldEarlyStop(const std::vector<float>& valHistory,
                         int patience);
};

struct ValidationResult {
    float mAP;          // 平均精度 (COCO-style)
    float mAP50;        // IoU=0.5 时的 mAP
    float precision;    // 精确率
    float recall;       // 召回率
    int totalImages;
};
```

早停策略：验证集 mAP 连续 `patience` 个 epoch 未提升则停止训练，恢复最佳权重。

## 训练调度器（自适应）

```cpp
class TrainingScheduler {
    Status run(TrainingPipelineConfig& cfg, ITrainCallback* callback);
};

struct TrainingPipelineConfig {
    std::string name;
    std::vector<TrainingStep> steps;
    std::string onError = "stop";
};

struct TrainingStep {
    std::string id;
    StepType type;       // kTrain / kExportOnnx / kBuildTensorRT
    Config config;
    bool required = true;
};
```

默认串行执行。单 GPU + 多任务的场景按步骤顺序串行执行，无需轮转并行——训练完成后模型从 GPU 显存卸载，再加载下一个模型。

## 导出层（跨 DLL 集成）

```cpp
class ModelExporter {
public:
    Status exportToOnnx(const ExportConfig& cfg, std::string& outOnnxPath);
    Status buildTensorRT(const BuildConfig& cfg, std::string& outEnginePath);
};
```

实现方式：`aicore_trainer.dll` 链接 `aicore_optimizer.lib`，直接调用 `AICore_ExportOnnx` / `AICore_BuildEngine` C 接口。训练完成后一步到位生成 `.engine` 文件。

## 训练回调

```cpp
class ITrainCallback {
public:
    virtual void onEpochBegin(int epoch, int totalEpochs) = 0;
    virtual void onEpochEnd(int epoch, float loss, float valMap) = 0;
    virtual void onBatchEnd(int batch, int totalBatches, float loss) = 0;
    virtual void onLog(LogLevel level, const std::string& msg) = 0;
    virtual Status onCheckpoint(const std::string& path) = 0;
    virtual bool shouldStop() = 0;  // 外部请求停止
};
```

Qt 实现此接口即可实时展示训练进度。

## DLL C 接口

```cpp
extern "C" {
AICORE_API int AICore_Train(const char* configPath);
AICORE_API int AICore_TrainAsync(const char* configPath, void* callback);
AICORE_API void AICore_TrainStop();
AICORE_API int AICore_TrainAndExport(const char* trainCfg, const char* optCfg);
}
```

`AICore_TrainAndExport` 执行流程：训练 → validate → export ONNX → build TensorRT engine，一步到位。

## CLI 使用方式

```
ModelTrainer.exe --config config_train.json
ModelTrainer.exe --train --data ./dataset --model faster_rcnn --epochs 100
ModelTrainer.exe --resume ./checkpoints/model_best.pt
ModelTrainer.exe --train-and-export --train-config train.json --optimize-config optimize.json
```

## 测试策略

- 单元测试：mock DataLoader 和训练回调，测试 TrainingLoop 的状态机
- 数据加载测试：用小型 COCO/VOC 测试集验证 Dataset 解析
- 模型测试：验证 ModelFactory 能正确创建各类型模型
- 集成测试：用 10 张图片微调 Faster R-CNN 2 个 epoch，验证损失下降
- YOLO 测试：通过 Python 嵌入执行最小化训练，验证流程通顺

## 目录结构

```
aicore_trainer/
├── include/
│   ├── core/             # 来自 Phase 1
│   ├── data/
│   │   ├── dataset.h, data_loader.h, augmentation.h
│   ├── model/
│   │   ├── model_factory.h, model_wrapper.h
│   ├── training/
│   │   ├── training_loop.h, scheduler.h, checkpoint.h
│   │   ├── multi_gpu_trainer.h, python_trainer.h
│   ├── validation/
│   │   ├── validator.h, early_stopping.h
│   ├── export/
│   │   └── model_exporter.h
│   ├── callback.h
│   └── trainer_api.h
├── src/
├── cli/
├── tests/
├── CMakeLists.txt
└── config_train.json
```

## 版本兼容

| 依赖 | 版本 |
|------|------|
| CUDA Toolkit | 11.8 |
| LibTorch | 2.1.0+cu118 |
| OpenCV | 4.7.0 |
| cuDNN | 8.7.0 |
| Python | 3.10+（仅 YOLO 训练需要） |
| PyTorch | 2.1.0+cu118（Python 端） |
