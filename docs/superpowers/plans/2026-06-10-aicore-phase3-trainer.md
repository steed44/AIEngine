# Phase 3: 训练模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 LibTorch 的 C++ 训练模块，支持数据集加载、数据增强、多模型串行训练流水线、自动调度退化、训练完成一键导出 TensorRT 引擎。

**Architecture:** LibTorch C++ 训练简单模型，Python 嵌入训练 YOLO 等复杂模型。TrainingPipeline 编排串行训练任务。训练完成后通过 aicore_optimizer.dll 导出 ONNX+TensorRT。

**Tech Stack:** VS2022, C++17, CUDA 11.8, LibTorch 2.1.0+cu118, OpenCV 4.7.0, cuDNN 8.7.0, Python 3.10+（YOLO 训练）

**Spec:** `docs/superpowers/specs/2026-06-10-aicore-phase3-trainer-design.md`
**Depends On:** Phase 1 (`aicore.dll` 公共类型), Phase 2 (`aicore_optimizer.dll` 导出功能)

---

## 文件结构

```
D:\w\AIEngine\aicore_trainer/
├── CMakeLists.txt
├── include/
│   ├── core/types.h                     # 引用 Phase 1 公共类型
│   ├── data/
│   │   ├── dataset.h                    # IDataset (COCO/VOC)
│   │   ├── data_loader.h               # DataLoader
│   │   └── augmentation.h              # AugmentationPipeline
│   ├── model/
│   │   ├── model_factory.h             # ModelFactory
│   │   └── python_trainer.h            # PythonTrainer (YOLO)
│   ├── training/
│   │   ├── training_loop.h             # TrainingLoop
│   │   ├── training_scheduler.h        # TrainingScheduler
│   │   ├── checkpoint.h               # CheckpointManager
│   │   └── early_stopping.h            # EarlyStopping
│   ├── validation/
│   │   └── validator.h                 # Validator (mAP)
│   ├── export/
│   │   └── model_exporter.h            # ModelExporter
│   ├── callback.h                      # ITrainCallback
│   └── trainer_api.h                   # DLL C API
├── scripts/
│   └── train_yolo.py                   # YOLO 训练脚本
├── src/
│   ├── data/ (dataset.cpp, data_loader.cpp, augmentation.cpp)
│   ├── model/ (model_factory.cpp, python_trainer.cpp)
│   ├── training/ (training_loop.cpp, training_scheduler.cpp, checkpoint.cpp, early_stopping.cpp)
│   ├── validation/ (validator.cpp)
│   ├── export/ (model_exporter.cpp)
│   └── trainer_api.cpp
├── cli/
│   └── main.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_dataset.cpp
│   ├── test_data_loader.cpp
│   ├── test_training_loop.cpp
│   └── test_checkpoint.cpp
├── CMakeLists.txt
└── config_train.json
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore_trainer\tests\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore_trainer\config_train.json`

- [ ] **Step 1: 创建 CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(aicore_trainer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖路径
set(AICORE_COMMON_DIR ${CMAKE_SOURCE_DIR}/../aicore/include)
set(AICORE_OPTIMIZER_DIR ${CMAKE_SOURCE_DIR}/../aicore_optimizer)

# LibTorch
set(LibTorch_DIR "C:/path/to/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# OpenCV
find_package(OpenCV 4.7.0 REQUIRED)

# Python (for YOLO training)
find_package(Python 3.10 REQUIRED COMPONENTS Development)

set(TRAINER_SOURCES
    src/data/dataset.cpp
    src/data/data_loader.cpp
    src/data/augmentation.cpp
    src/model/model_factory.cpp
    src/model/python_trainer.cpp
    src/training/training_loop.cpp
    src/training/training_scheduler.cpp
    src/training/checkpoint.cpp
    src/training/early_stopping.cpp
    src/validation/validator.cpp
    src/export/model_exporter.cpp
    src/trainer_api.cpp
)

add_library(aicore_trainer SHARED ${TRAINER_SOURCES})
target_include_directories(aicore_trainer PUBLIC include ${AICORE_COMMON_DIR})
target_include_directories(aicore_trainer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(aicore_trainer PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(aicore_trainer PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${Python_LIBRARIES}
)
target_compile_definitions(aicore_trainer PRIVATE AICORE_EXPORTS)

# CLI
add_executable(ModelTrainer cli/main.cpp)
target_link_libraries(ModelTrainer PRIVATE aicore_trainer)

enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 2: 创建 config_train.json**

```json
{
  "training_pipeline": {
    "name": "缺陷检测模型训练",
    "steps": [
      {
        "id": "train_detector",
        "type": "train",
        "config": {
          "model_type": "faster_rcnn",
          "train_image_dir": "./dataset/train",
          "train_label_dir": "./dataset/train/labels",
          "val_image_dir": "./dataset/val",
          "val_label_dir": "./dataset/val/labels",
          "annotation_format": "coco",
          "class_names": ["background", "scratch", "dent", "stain"],
          "num_classes": 4,
          "input_width": 640,
          "input_height": 640,
          "batch_size": 4,
          "epochs": 100,
          "learning_rate": 0.001,
          "optimizer": "adam",
          "enable_amp": true,
          "gradient_accumulation": 2,
          "output_dir": "./models/checkpoints",
          "save_interval": 10,
          "save_best_only": true
        }
      },
      {
        "id": "export_engine",
        "type": "export_onnx",
        "depends_on": "train_detector",
        "config": {
          "precision": "fp16",
          "input_width": 640,
          "input_height": 640
        }
      }
    ],
    "on_error": "stop"
  }
}
```

---

### Task 2: IDataset — 数据集加载

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\data\dataset.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\data\dataset.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\tests\test_dataset.cpp`

- [ ] **Step 1: 创建 dataset.h**

```cpp
#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace aicore {

struct BBox { float x, y, w, h; };
struct Annotation {
    BBox bbox;
    int label;
};

struct TrainSample {
    cv::Mat image;
    std::vector<Annotation> annotations;
    std::string imagePath;
    int width = 0, height = 0;
};

class IDataset {
public:
    virtual ~IDataset() = default;
    virtual size_t size() const = 0;
    virtual Status getItem(size_t index, TrainSample& sample) = 0;
    virtual std::vector<std::string> classNames() const = 0;
};

// COCO 格式数据集
class CocoDataset : public IDataset {
public:
    CocoDataset(const std::string& imageDir,
                const std::string& annotationFile);
    size_t size() const override;
    Status getItem(size_t index, TrainSample& sample) override;
    std::vector<std::string> classNames() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 dataset.cpp 骨架**

```cpp
#include "data/dataset.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace aicore {

struct CocoDataset::Impl {
    std::vector<std::string> images;
    std::vector<std::vector<Annotation>> annotations;
    std::vector<std::string> classNames_;
    std::string imageDir_;
};

CocoDataset::CocoDataset(const std::string& imageDir,
                          const std::string& annotationFile)
    : impl_(std::make_unique<Impl>()) {
    impl_->imageDir_ = imageDir;

    // 解析 COCO JSON
    std::ifstream f(annotationFile);
    json j; f >> j;

    // 读取类别
    for (auto& cat : j["categories"]) {
        impl_->classNames_.push_back(cat["name"]);
    }

    // 读取图片和标注（简化版）
    for (auto& img : j["images"]) {
        impl_->images.push_back(img["file_name"]);
        impl_->annotations.emplace_back();
    }
    // 完整版需建立 image_id → annotations 映射
}

size_t CocoDataset::size() const { return impl_->images.size(); }

Status CocoDataset::getItem(size_t index, TrainSample& sample) {
    if (index >= size()) return {StatusCode::ErrorInvalidInput, "Index out of range"};
    sample.imagePath = impl_->imageDir_ + "/" + impl_->images[index];
    sample.image = cv::imread(sample.imagePath);
    sample.annotations = impl_->annotations[index];
    sample.width = sample.image.cols;
    sample.height = sample.image.rows;
    return Status{};
}

std::vector<std::string> CocoDataset::classNames() const {
    return impl_->classNames_;
}

} // namespace aicore
```

- [ ] **Step 3: 测试**

```cpp
#include <gtest/gtest.h>
#include "data/dataset.h"

using namespace aicore;

TEST(CocoDatasetTest, LoadAnnotation) {
    // 需要真实测试数据，此处仅验证接口
    // 实际测试用小型 COCO 格式样本
}
```

---

### Task 3: DataLoader — 多线程数据加载

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\data\data_loader.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\data\data_loader.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\tests\test_data_loader.cpp`

- [ ] **Step 1: 创建 data_loader.h**

```cpp
#pragma once
#include "data/dataset.h"
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace aicore {

class DataLoader {
public:
    DataLoader(std::unique_ptr<IDataset> dataset,
               int batchSize, int numWorkers, bool shuffle = true);
    ~DataLoader();

    void start();
    bool nextBatch(std::vector<TrainSample>& batch);
    void stop();
    size_t epochSize() const;
    size_t datasetSize() const { return dataset_->size(); }

private:
    std::unique_ptr<IDataset> dataset_;
    int batchSize_, numWorkers_;
    bool shuffle_;
    std::queue<std::vector<TrainSample>> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;
    std::vector<size_t> indices_;
    size_t currentIdx_ = 0;

    void workerFunc();
};

} // namespace aicore
```

- [ ] **Step 2: 创建 data_loader.cpp**

```cpp
#include "data/data_loader.h"
#include <algorithm>
#include <random>

namespace aicore {

DataLoader::DataLoader(std::unique_ptr<IDataset> dataset,
                        int batchSize, int numWorkers, bool shuffle)
    : dataset_(std::move(dataset)), batchSize_(batchSize),
      numWorkers_(numWorkers), shuffle_(shuffle) {
    indices_.resize(dataset_->size());
    for (size_t i = 0; i < indices_.size(); ++i) indices_[i] = i;
}

DataLoader::~DataLoader() { stop(); }

void DataLoader::start() {
    if (running_) return;
    running_ = true;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(),
                      std::mt19937(std::random_device{}()));
    }
    currentIdx_ = 0;
    for (int i = 0; i < numWorkers_; ++i) {
        workers_.emplace_back(&DataLoader::workerFunc, this);
    }
}

void DataLoader::workerFunc() {
    while (running_) {
        std::vector<TrainSample> batch;
        for (int i = 0; i < batchSize_; ++i) {
            size_t idx;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (currentIdx_ >= indices_.size()) break;
                idx = indices_[currentIdx_++];
            }
            TrainSample sample;
            if (dataset_->getItem(idx, sample)) {
                batch.push_back(std::move(sample));
            }
        }
        if (batch.empty()) break;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(batch));
        }
        cond_.notify_one();
    }
}

bool DataLoader::nextBatch(std::vector<TrainSample>& batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return !queue_.empty() || !running_; });
    if (queue_.empty()) return false;
    batch = std::move(queue_.front());
    queue_.pop();
    return true;
}

void DataLoader::stop() {
    running_ = false;
    cond_.notify_all();
    for (auto& w : workers_) if (w.joinable()) w.join();
}

size_t DataLoader::epochSize() const {
    return (dataset_->size() + batchSize_ - 1) / batchSize_;
}

} // namespace aicore
```

---

### Task 4: AugmentationPipeline — 数据增强

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\data\augmentation.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\data\augmentation.cpp`

- [ ] **Step 1: 创建 augmentation.h**

```cpp
#pragma once
#include "data/dataset.h"
#include <memory>
#include <vector>

namespace aicore {

class IAugmentation {
public:
    virtual ~IAugmentation() = default;
    virtual Status process(TrainSample& sample) = 0;
};

class AugmentationPipeline {
    std::vector<std::unique_ptr<IAugmentation>> steps_;
public:
    void addStep(std::unique_ptr<IAugmentation> step);
    Status process(TrainSample& sample);
};

class RandomFlip : public IAugmentation {
    float prob_;
public:
    explicit RandomFlip(float prob = 0.5);
    Status process(TrainSample& sample) override;
};

class ColorJitter : public IAugmentation {
    float brightness_, contrast_, saturation_;
public:
    ColorJitter(float b = 0.2, float c = 0.2, float s = 0.2);
    Status process(TrainSample& sample) override;
};

} // namespace aicore
```

---

### Task 5: TrainingLoop + CheckpointManager

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\training\training_loop.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\training\training_loop.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\include\training\checkpoint.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\training\checkpoint.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\tests\test_training_loop.cpp`

- [ ] **Step 1: 创建 training_loop.h**

```cpp
#pragma once
#include "core/types.h"
#include "data/data_loader.h"
#include "callback.h"
#include <torch/torch.h>
#include <memory>

namespace aicore {

struct TrainConfig {
    std::string modelType;
    int numClasses;
    int inputWidth = 640, inputHeight = 640;
    int batchSize = 8, epochs = 100;
    float learningRate = 0.001f;
    std::string optimizer = "adam";
    std::string scheduler = "cosine";
    bool enableAmp = true;
    int gradientAccumulation = 1;
    float gradClipNorm = 0.0f;
    std::string device = "cuda:0";
    std::string outputDir;
    int saveInterval = 5;
    bool saveBestOnly = true;
    std::string resumeFrom;
    int earlyStopPatience = 10;
};

class TrainingLoop {
public:
    explicit TrainingLoop(const TrainConfig& cfg);
    ~TrainingLoop();

    Status run(DataLoader& trainLoader, DataLoader* valLoader,
               ITrainCallback* callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 checkpoint.h**

```cpp
#pragma once
#include "core/types.h"
#include <torch/torch.h>
#include <string>

namespace aicore {

class CheckpointManager {
public:
    CheckpointManager(const std::string& outputDir,
                      int saveInterval, bool saveBestOnly);

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
    float bestMetrics_ = 0.0f;
    std::string bestPath_;
};

} // namespace aicore
```

- [ ] **Step 3: 创建 checkpoint.cpp**

```cpp
#include "training/checkpoint.h"
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace aicore {

CheckpointManager::CheckpointManager(const std::string& outputDir,
                                      int saveInterval, bool saveBestOnly)
    : outputDir_(outputDir), saveInterval_(saveInterval),
      saveBestOnly_(saveBestOnly) {
    std::filesystem::create_directories(outputDir);
    bestPath_ = outputDir + "/model_best.pt";
}

Status CheckpointManager::save(const std::string&, int epoch,
                                torch::nn::Module& model,
                                torch::optim::Optimizer& opt,
                                float metrics) {
    std::ostringstream ss;
    ss << outputDir_ << "/checkpoint_epoch_"
       << std::setw(3) << std::setfill('0') << epoch
       << "_loss_" << std::fixed << std::setprecision(4) << metrics
       << ".pt";

    torch::save(model, ss.str());
    torch::save(opt, outputDir_ + "/optimizer.pt");

    if (saveBestOnly_ && metrics > bestMetrics_) {
        bestMetrics_ = metrics;
        torch::save(model, bestPath_);
    }

    return Status{};
}

Status CheckpointManager::loadBest(torch::nn::Module& model,
                                    torch::optim::Optimizer& opt,
                                    float& bestMetrics) {
    if (!std::filesystem::exists(bestPath_))
        return {StatusCode::ErrorConfigParse, "No best checkpoint found"};
    torch::load(model, bestPath_);
    bestMetrics = bestMetrics_;
    return Status{};
}

} // namespace aicore
```

---

### Task 6: TrainingScheduler — 多模型串行训练编排

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\training\training_scheduler.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\training\training_scheduler.cpp`

- [ ] **Step 1: 创建 training_scheduler.h**

```cpp
#pragma once
#include "core/types.h"
#include "training/training_loop.h"
#include "callback.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using Config = nlohmann::json;

namespace aicore {

enum class StepType { kTrain, kExportOnnx, kBuildTensorRT };

struct TrainingStep {
    std::string id;
    StepType type;
    Config config;
    bool required = true;
};

struct TrainingPipelineConfig {
    std::string name;
    std::vector<TrainingStep> steps;
    std::string onError = "stop";
};

class TrainingScheduler {
public:
    Status run(const TrainingPipelineConfig& cfg,
               ITrainCallback* callback);

private:
    Status executeStep(const TrainingStep& step,
                       ITrainCallback* callback);
    Status executeTrain(const Config& cfg, ITrainCallback* cb);
    Status executeExport(const Config& cfg);
    Status executeBuildEngine(const Config& cfg);
    TrainingConfig parseTrainConfig(const Config& cfg);
};

} // namespace aicore
```

- [ ] **Step 2: 创建 training_scheduler.cpp**

```cpp
#include "training/training_scheduler.h"

namespace aicore {

Status TrainingScheduler::run(const TrainingPipelineConfig& cfg,
                               ITrainCallback* callback) {
    for (auto& step : cfg.steps) {
        if (callback) callback->onLog(LogLevel::kInfo,
            "Starting step: " + step.id);

        auto status = executeStep(step, callback);

        if (!status) {
            std::string msg = "Step " + step.id + " failed: " + status.message;
            if (callback) callback->onLog(LogLevel::kError, msg);

            if (cfg.onError == "stop" || step.required) {
                return status;
            }
            // cfg.onError == "skip": 继续下一步
        }
    }
    return Status{};
}

Status TrainingScheduler::executeStep(const TrainingStep& step,
                                       ITrainCallback* callback) {
    switch (step.type) {
        case StepType::kTrain:
            return executeTrain(step.config, callback);
        case StepType::kExportOnnx:
            return executeExport(step.config);
        case StepType::kBuildTensorRT:
            return executeBuildEngine(step.config);
    }
    return {StatusCode::ErrorInternal, "Unknown step type"};
}

Status TrainingScheduler::executeTrain(const Config& cfg,
                                        ITrainCallback* callback) {
    auto trainCfg = parseTrainConfig(cfg);

    // 创建数据集和数据加载器
    auto dataset = std::make_unique<CocoDataset>(
        cfg["train_image_dir"], cfg["train_label_dir"]);
    auto loader = std::make_unique<DataLoader>(
        std::move(dataset), trainCfg.batchSize, 4, true);
    loader->start();

    DataLoader* valLoader = nullptr;
    std::unique_ptr<DataLoader> valLoaderOwner;
    if (cfg.contains("val_image_dir")) {
        auto valDataset = std::make_unique<CocoDataset>(
            cfg["val_image_dir"], cfg["val_label_dir"]);
        valLoaderOwner = std::make_unique<DataLoader>(
            std::move(valDataset), trainCfg.batchSize, 2, false);
        valLoaderOwner->start();
        valLoader = valLoaderOwner.get();
    }

    // 运行训练
    TrainingLoop loop(trainCfg);
    auto status = loop.run(*loader, valLoader, callback);

    loader->stop();
    if (valLoaderOwner) valLoaderOwner->stop();

    return status;
}

TrainingConfig TrainingScheduler::parseTrainConfig(const Config& cfg) {
    TrainingConfig tc;
    tc.modelType = cfg.value("model_type", "faster_rcnn");
    tc.numClasses = cfg["num_classes"];
    tc.inputWidth = cfg.value("input_width", 640);
    tc.inputHeight = cfg.value("input_height", 640);
    tc.batchSize = cfg.value("batch_size", 4);
    tc.epochs = cfg.value("epochs", 100);
    tc.learningRate = cfg.value("learning_rate", 0.001f);
    tc.optimizer = cfg.value("optimizer", "adam");
    tc.enableAmp = cfg.value("enable_amp", true);
    tc.gradientAccumulation = cfg.value("gradient_accumulation", 1);
    tc.outputDir = cfg.value("output_dir", "models/checkpoints");
    tc.saveInterval = cfg.value("save_interval", 5);
    tc.saveBestOnly = cfg.value("save_best_only", true);
    return tc;
}

Status TrainingScheduler::executeExport(const Config& cfg) {
    // 调用 Phase 2 的 AICore_ExportOnnx
    // 通过动态加载 aicore_optimizer.dll 实现
    // （实际实现体在 model_exporter.cpp 中）
    return Status{};
}

Status TrainingScheduler::executeBuildEngine(const Config& cfg) {
    // 调用 Phase 2 的 AICore_BuildEngine
    return Status{};
}

} // namespace aicore
```

---

### Task 7: EarlyStopping + Validator

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\training\early_stopping.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\training\early_stopping.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\include\validation\validator.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\validation\validator.cpp`

- [ ] **Step 1: 创建 early_stopping.h**

```cpp
#pragma once
#include <vector>

namespace aicore {

class EarlyStopping {
public:
    explicit EarlyStopping(int patience = 10, float minDelta = 0.001f);

    bool shouldStop(float currentMetric);
    float bestMetric() const { return bestMetric_; }
    int bestEpoch() const { return bestEpoch_; }

private:
    int patience_;
    float minDelta_;
    float bestMetric_ = 0;
    int bestEpoch_ = 0;
    int counter_ = 0;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 validator.h**

```cpp
#pragma once
#include "core/types.h"
#include "data/data_loader.h"
#include <torch/torch.h>

namespace aicore {

struct ValidationResult {
    float mAP = 0;
    float mAP50 = 0;
    float precision = 0;
    float recall = 0;
    int totalImages = 0;
    int totalDetections = 0;
};

class Validator {
public:
    Status evaluate(DataLoader& valLoader,
                    torch::nn::AnyModule& model,
                    torch::Device& device,
                    int numClasses,
                    ValidationResult& result);
};

} // namespace aicore
```

---

### Task 8: ModelExporter — 导出层（跨 DLL 集成）

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\export\model_exporter.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\export\model_exporter.cpp`

- [ ] **Step 1: 创建 model_exporter.h**

```cpp
#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

class ModelExporter {
public:
    // 将训练好的模型导出为 ONNX（调用 aicore_optimizer.dll）
    Status exportToOnnx(const std::string& modelPath,
                        const std::string& outputPath,
                        int inputWidth, int inputHeight,
                        const std::string& precision);

    // 构建 TensorRT 引擎（调用 aicore_optimizer.dll）
    Status buildEngine(const std::string& onnxPath,
                       const std::string& outputPath,
                       const std::string& precision);
};

} // namespace aicore
```

- [ ] **Step 2: 创建 model_exporter.cpp（通过 LoadLibrary 调用 aicore_optimizer.dll）**

```cpp
#include "export/model_exporter.h"
#include <Windows.h>

namespace aicore {

Status ModelExporter::exportToOnnx(const std::string& modelPath,
                                    const std::string& outputPath,
                                    int inputWidth, int inputHeight,
                                    const std::string& precision) {
    // 动态加载 aicore_optimizer.dll
    HMODULE hLib = LoadLibrary(L"aicore_optimizer.dll");
    if (!hLib) {
        return {StatusCode::ErrorInternal, "Cannot load aicore_optimizer.dll"};
    }

    using ExportFn = int(*)(const char*);
    auto fn = (ExportFn)GetProcAddress(hLib, "AICore_ExportOnnx");
    if (!fn) {
        FreeLibrary(hLib);
        return {StatusCode::ErrorInternal, "Cannot find AICore_ExportOnnx"};
    }

    // 构造临时配置 JSON
    nlohmann::json cfg;
    cfg["model_path"] = modelPath;
    cfg["output_dir"] = outputPath.substr(0, outputPath.find_last_of("/\\"));
    cfg["input"]["width"] = inputWidth;
    cfg["input"]["height"] = inputHeight;
    cfg["precision"] = precision;

    std::string cfgStr = cfg.dump();
    // 写入临时文件
    std::string tmpPath = outputPath + ".tmp_config.json";
    std::ofstream(tmpPath) << cfgStr;

    int ret = fn(tmpPath.c_str());
    std::remove(tmpPath.c_str());

    FreeLibrary(hLib);
    return ret == 0 ? Status{} :
        Status{StatusCode::ErrorInternal, "Export failed"};
}

Status ModelExporter::buildEngine(const std::string& onnxPath,
                                   const std::string& outputPath,
                                   const std::string& precision) {
    HMODULE hLib = LoadLibrary(L"aicore_optimizer.dll");
    if (!hLib) return {StatusCode::ErrorInternal, "Cannot load aicore_optimizer.dll"};

    using BuildFn = int(*)(const char*);
    auto fn = (BuildFn)GetProcAddress(hLib, "AICore_BuildEngine");
    if (!fn) { FreeLibrary(hLib); return {StatusCode::ErrorInternal, "Cannot find AICore_BuildEngine"}; }

    nlohmann::json cfg;
    cfg["onnx_path"] = onnxPath;
    cfg["output_dir"] = outputPath.substr(0, outputPath.find_last_of("/\\"));
    cfg["precision"] = precision;

    std::string tmpPath = outputPath + ".tmp_build_config.json";
    std::ofstream(tmpPath) << cfg.dump();

    int ret = fn(tmpPath.c_str());
    std::remove(tmpPath.c_str());

    FreeLibrary(hLib);
    return ret == 0 ? Status{} : Status{StatusCode::ErrorInternal, "Build failed"};
}

} // namespace aicore
```

---

### Task 9: ITrainCallback + DLL C API + CLI

**Files:**
- Create: `D:\w\AIEngine\aicore_trainer\include\callback.h`
- Create: `D:\w\AIEngine\aicore_trainer\include\trainer_api.h`
- Create: `D:\w\AIEngine\aicore_trainer\src\trainer_api.cpp`
- Create: `D:\w\AIEngine\aicore_trainer\cli\main.cpp`

- [ ] **Step 1: 创建 callback.h**

```cpp
#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

enum class LogLevel { kDebug, kInfo, kWarn, kError };

class ITrainCallback {
public:
    virtual ~ITrainCallback() = default;
    virtual void onEpochBegin(int epoch, int totalEpochs) = 0;
    virtual void onEpochEnd(int epoch, float loss, float valMap) = 0;
    virtual void onBatchEnd(int batch, int totalBatches, float loss) = 0;
    virtual void onLog(LogLevel level, const std::string& msg) = 0;
    virtual Status onCheckpoint(const std::string& path) = 0;
    virtual bool shouldStop() = 0;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 trainer_api.h**

```cpp
#pragma once
#include "core/types.h"

extern "C" {

AICORE_API int AICore_Train(const char* configPath);
AICORE_API int AICore_TrainAsync(const char* configPath, void* callback);
AICORE_API void AICore_TrainStop();
AICORE_API int AICore_TrainAndExport(const char* trainCfg, const char* optCfg);

}
```

- [ ] **Step 3: 创建 CLI main.cpp**

```cpp
#include <iostream>
#include "trainer_api.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "  ModelTrainer.exe --config config.json" << std::endl;
        std::cerr << "  ModelTrainer.exe --train-and-export train.json optimize.json" << std::endl;
        return 1;
    }

    std::string cmd = argv[1];
    int ret = -1;

    if (cmd == "--config") {
        ret = AICore_Train(argv[2]);
    } else if (cmd == "--train-and-export") {
        ret = AICore_TrainAndExport(argv[2], argv[3]);
    }

    if (ret != 0) {
        std::cerr << "Training failed with code: " << ret << std::endl;
        return ret;
    }
    std::cout << "Training completed successfully." << std::endl;
    return 0;
}
```

---

### Task 10: 编译验证

**Files:**
- Modify: `D:\w\AIEngine\aicore_trainer\CMakeLists.txt`
- Modify: `D:\w\AIEngine\aicore_trainer\tests\CMakeLists.txt`

- [ ] **Step 1: 完整编译**

```bash
cd D:\w\AIEngine\aicore_trainer && mkdir build; cd build
cmake .. -DCMAKE_PREFIX_PATH="path/to/libtorch;path/to/opencv"
cmake --build . --config Release
```

Expected: aicore_trainer.dll + ModelTrainer.exe 生成成功

- [ ] **Step 2: 运行测试**

```bash
ctest --output-on-failure
```

Expected: 所有测试通过
