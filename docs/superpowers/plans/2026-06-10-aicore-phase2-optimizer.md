# Phase 2: 模型优化工具 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 PyTorch→ONNX→TensorRT 全流程模型优化工具，提供 CLI (ModelOptimizer.exe) + DLL (aicore_optimizer.dll) 双形态，C++ 内嵌 Python 完成 ONNX 导出。

**Architecture:** Python 嵌入层管理 PyTorch 导出流程，TensorRT C++ API 构建引擎引擎。优化器 DLL 独立于推理引擎，仅训练阶段调用。

**Tech Stack:** VS2022, C++17, CUDA 11.8, TensorRT 8.5.3, Python 3.10+, PyTorch 2.1.0, ONNX 1.14, onnxsim 0.4

**Spec:** `docs/superpowers/specs/2026-06-10-aicore-phase2-optimizer-design.md`

---

## 文件结构

```
D:\w\AIEngine\aicore_optimizer/
├── CMakeLists.txt
├── include/
│   ├── core/types.h                    # 引用 Phase 1 的公共类型（Status, StatusCode, AICORE_API）
│   ├── python_embedding.h
│   ├── onnx_exporter.h
│   ├── tensorrt_builder.h
│   ├── int8_calibrator.h
│   ├── model_optimizer.h
│   └── optimizer_api.h
├── scripts/
│   └── export_onnx.py
├── src/
│   ├── python_embedding.cpp
│   ├── onnx_exporter.cpp
│   ├── tensorrt_builder.cpp
│   ├── int8_calibrator.cpp
│   ├── model_optimizer.cpp
│   └── optimizer_api.cpp
├── cli/
│   └── main.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_python_embedding.cpp
│   ├── test_tensorrt_builder.cpp
│   └── test_int8_calibrator.cpp
└── config_optimize.json
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore_optimizer\tests\CMakeLists.txt`
- Create: `D:\w\AIEngine\aicore_optimizer\config_optimize.json`
- Create: `D:\w\AIEngine\aicore_optimizer\include\core\types.h`（复制 Phase 1 的 types.h，或 CMake 引用路径）

- [ ] **Step 1: 创建 CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(aicore_optimizer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 引用 Phase 1 公共类型
set(AICORE_COMMON_DIR ${CMAKE_SOURCE_DIR}/../aicore/include)

# CUDA
find_package(CUDA 11.8 REQUIRED)

# TensorRT
set(TENSORRT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.5.3")
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

# Python
find_package(Python 3.10 REQUIRED COMPONENTS Development)

set(OPTIMIZER_SOURCES
    src/python_embedding.cpp
    src/onnx_exporter.cpp
    src/tensorrt_builder.cpp
    src/int8_calibrator.cpp
    src/model_optimizer.cpp
    src/optimizer_api.cpp
)

add_library(aicore_optimizer SHARED ${OPTIMIZER_SOURCES})
target_include_directories(aicore_optimizer PUBLIC include ${AICORE_COMMON_DIR})
target_include_directories(aicore_optimizer PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(aicore_optimizer PRIVATE
    ${CUDA_LIBRARIES}
    ${Python_LIBRARIES}
    nvinfer nvonnxparser nvinfer_plugin
)
target_compile_definitions(aicore_optimizer PRIVATE AICORE_EXPORTS)

# CLI
add_executable(ModelOptimizer cli/main.cpp)
target_link_libraries(ModelOptimizer PRIVATE aicore_optimizer)

enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 2: 创建 config_optimize.json**

```json
{
  "model_path": "models/trained_model.pt",
  "model_type": "torchscript",
  "output_dir": "models/optimized",
  "input_width": 640,
  "input_height": 640,
  "batch_size": 1,
  "dynamic_batch": true,
  "opset_version": 17,
  "precision": "fp16",
  "calib_dir": "data/calibration_samples",
  "python_home": "C:/Users/user/AppData/Local/Programs/Python/Python310",
  "python_path": "C:/Users/user/AppData/Local/Programs/Python/Python310/Lib/site-packages"
}
```

---

### Task 2: PythonEmbedding — Python 解释器嵌入层

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\include\python_embedding.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\python_embedding.cpp`
- Create: `D:\w\AIEngine\aicore_optimizer\tests\test_python_embedding.cpp`

- [ ] **Step 1: 创建 python_embedding.h**

```cpp
#pragma once
#include "core/types.h"
#include <nlohmann/json.hpp>
#include <string>
#include <mutex>

using Config = nlohmann::json;

namespace aicore {

class PythonEmbedding {
public:
    PythonEmbedding();
    ~PythonEmbedding();

    Status initialize(const std::string& pythonHome,
                      const std::string& pythonPath);
    Status execFile(const std::string& scriptPath);
    Status execString(const std::string& code);
    Status callFunction(const std::string& module,
                        const std::string& func,
                        const Config& args, Config& result);
    void finalize();

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_ = false;
    std::once_flag initFlag_;
    std::mutex gilMutex_;

    void ensureGil();

    static bool globalPythonInitialized_;
    static std::mutex globalInitMutex_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 python_embedding.cpp**

```cpp
#include "python_embedding.h"
#include <Python.h>
#include <filesystem>

namespace aicore {

bool PythonEmbedding::globalPythonInitialized_ = false;
std::mutex PythonEmbedding::globalInitMutex_;

PythonEmbedding::PythonEmbedding() = default;
PythonEmbedding::~PythonEmbedding() { finalize(); }

Status PythonEmbedding::initialize(const std::string& pythonHome,
                                    const std::string& pythonPath) {
    std::call_once(initFlag_, [this, &pythonHome, &pythonPath]() {
        std::lock_guard<std::mutex> lock(globalInitMutex_);
        if (!globalPythonInitialized_) {
            // 设置 Python Home
            std::wstring homeW(pythonHome.begin(), pythonHome.end());
            Py_SetPythonHome(homeW.c_str());

            // 添加 site-packages 路径
            wchar_t* path = Py_DecodeLocale(pythonPath.c_str(), nullptr);
            Py_SetPath(path);
            PyMem_RawFree(path);

            Py_Initialize();
            if (!Py_IsInitialized()) {
                initialized_ = false;
                return;
            }
            globalPythonInitialized_ = true;
        }
        initialized_ = true;
    });
    return initialized_ ? Status{} :
        Status{StatusCode::ErrorInternal, "Python init failed"};
}

Status PythonEmbedding::execString(const std::string& code) {
    std::lock_guard<std::mutex> lock(gilMutex_);
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyRun_SimpleString(code.c_str());
    PyGILState_Release(gstate);
    return Status{};
}

Status PythonEmbedding::callFunction(const std::string& module,
                                      const std::string& func,
                                      const Config& args, Config& result) {
    std::lock_guard<std::mutex> lock(gilMutex_);
    PyGILState_STATE gstate = PyGILState_Ensure();

    // 导入模块
    PyObject* pModule = PyImport_ImportModule(module.c_str());
    if (!pModule) {
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal, "Failed to import: " + module};
    }

    // 获取函数
    PyObject* pFunc = PyObject_GetAttrString(pModule, func.c_str());
    if (!pFunc || !PyCallable_Check(pFunc)) {
        Py_DECREF(pModule);
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal, "Function not found: " + func};
    }

    // 将 Config (nlohmann::json) 转为 Python dict
    std::string jsonStr = args.dump();
    PyObject* pArgs = PyDict_New();
    // 简化为调 eval 解析 JSON
    std::string evalCode = "import json; json.loads('" + jsonStr + "')";
    pArgs = PyRun_String(evalCode.c_str(), Py_eval_input, 
                         PyEval_GetGlobals(), PyEval_GetLocals());

    // 调用函数
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if (!pResult) {
        Py_DECREF(pFunc); Py_DECREF(pModule);
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal, "Function call failed"};
    }

    // 从 Python 结果转回 Config
    PyObject* pStr = PyObject_Str(pResult);
    const char* cStr = PyUnicode_AsUTF8(pStr);
    if (cStr) result = Config::parse(cStr);

    Py_XDECREF(pStr); Py_DECREF(pResult);
    Py_DECREF(pFunc); Py_DECREF(pModule);
    PyGILState_Release(gstate);
    return Status{};
}

Status PythonEmbedding::execFile(const std::string& scriptPath) {
    if (!std::filesystem::exists(scriptPath)) {
        return Status{StatusCode::ErrorConfigParse, "Script not found: " + scriptPath};
    }
    std::lock_guard<std::mutex> lock(gilMutex_);
    PyGILState_STATE gstate = PyGILState_Ensure();

    FILE* fp = fopen(scriptPath.c_str(), "r");
    if (fp) {
        PyRun_SimpleFile(fp, scriptPath.c_str());
        fclose(fp);
    } else {
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal, "Cannot open: " + scriptPath};
    }

    PyGILState_Release(gstate);
    return Status{};
}

void PythonEmbedding::finalize() {
    if (globalPythonInitialized_) {
        std::lock_guard<std::mutex> lock(globalInitMutex_);
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_Finalize();
        globalPythonInitialized_ = false;
        initialized_ = false;
    }
}

} // namespace aicore
```

- [ ] **Step 3: 测试**

```cpp
#include <gtest/gtest.h>
#include "python_embedding.h"

using namespace aicore;

TEST(PythonEmbeddingTest, InitAndFinalize) {
    PythonEmbedding py;
    auto status = py.initialize("", "");
    EXPECT_TRUE(status) << status.message;
    EXPECT_TRUE(py.isInitialized());
}

TEST(PythonEmbeddingTest, ExecString) {
    PythonEmbedding py;
    ASSERT_TRUE(py.initialize("", ""));
    auto status = py.execString("a = 1 + 2");
    EXPECT_TRUE(status);
}
```

---

### Task 3: export_onnx.py — ONNX 导出脚本

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\scripts\export_onnx.py`

- [ ] **Step 1: 创建 export_onnx.py**

```python
"""
被 C++ PythonEmbedding::callFunction 调用的 ONNX 导出脚本。
函数签名: export_model(model_path, output_path, input_width, input_height,
                        batch_size, dynamic_batch, opset_version, device)
         -> {"success": bool, "onnx_path": str, "error": str}
"""
import torch
import onnx
import onnxsim
import json
import sys


def export_model(model_path, output_path, input_width=640, input_height=640,
                  batch_size=1, dynamic_batch=True, opset_version=17,
                  device="cuda:0"):
    try:
        # 加载 TorchScript 模型
        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        # 创建 dummy input
        dummy = torch.randn(batch_size, 3, input_height, input_width).to(device)

        # ONNX 导出
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }

        torch.onnx.export(
            model, dummy, output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )

        # 精简模型
        model_simp, check = onnxsim.simplify(output_path)
        if check:
            onnx.save(model_simp, output_path)

        return json.dumps({
            "success": True,
            "onnx_path": output_path,
            "error": ""
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "onnx_path": "",
            "error": str(e)
        })


if __name__ == "__main__":
    # CLI 调用入口
    args = json.loads(sys.argv[1])
    result = export_model(**args)
    print(result)
```

---

### Task 4: OnnxExporter — ONNX 导出编排器

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\include\onnx_exporter.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\onnx_exporter.cpp`

- [ ] **Step 1: 创建 onnx_exporter.h**

```cpp
#pragma once
#include "core/types.h"
#include "python_embedding.h"
#include <nlohmann/json.hpp>
#include <string>

using Config = nlohmann::json;

namespace aicore {

struct ExportConfig {
    std::string modelPath;
    std::string modelType = "torchscript";
    std::string outputDir;
    int inputWidth = 640;
    int inputHeight = 640;
    int batchSize = 1;
    bool dynamicBatch = true;
    int opsetVersion = 17;
    std::string device = "cuda:0";
};

class OnnxExporter {
public:
    explicit OnnxExporter(PythonEmbedding* py);
    Status exportToOnnx(const ExportConfig& cfg, std::string& outOnnxPath);

private:
    PythonEmbedding* py_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 onnx_exporter.cpp**

```cpp
#include "onnx_exporter.h"
#include <filesystem>

namespace aicore {

OnnxExporter::OnnxExporter(PythonEmbedding* py) : py_(py) {}

Status OnnxExporter::exportToOnnx(const ExportConfig& cfg,
                                   std::string& outOnnxPath) {
    // 确认输出目录
    std::filesystem::create_directories(cfg.outputDir);

    // 生成输出路径
    std::string baseName = std::filesystem::path(cfg.modelPath).stem().string();
    outOnnxPath = cfg.outputDir + "/" + baseName + ".onnx";

    // 构造参数
    Config args;
    args["model_path"] = cfg.modelPath;
    args["output_path"] = outOnnxPath;
    args["input_width"] = cfg.inputWidth;
    args["input_height"] = cfg.inputHeight;
    args["batch_size"] = cfg.batchSize;
    args["dynamic_batch"] = cfg.dynamicBatch;
    args["opset_version"] = cfg.opsetVersion;
    args["device"] = cfg.device;

    // 调用 Python 脚本
    Config result;
    auto status = py_->callFunction("export_onnx", "export_model",
                                     args, result);
    if (!status) return status;

    if (!result.value("success", false)) {
        return Status{StatusCode::ErrorInternal,
                       result.value("error", "Unknown export error")};
    }

    return Status{};
}

} // namespace aicore
```

---

### Task 5: TensorRTBuilder — TensorRT 引擎构建

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\include\tensorrt_builder.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\tensorrt_builder.cpp`
- Create: `D:\w\AIEngine\aicore_optimizer\tests\test_tensorrt_builder.cpp`

- [ ] **Step 1: 创建 tensorrt_builder.h**

```cpp
#pragma once
#include "core/types.h"
#include <string>
#include <memory>

namespace aicore {

struct BuildConfig {
    std::string onnxPath;
    std::string outputDir;
    int inputWidth = 640;
    int inputHeight = 640;
    int batchSize = 1;
    int maxBatchSize = 8;
    std::string precision = "fp16";  // "fp32" / "fp16" / "int8"
    std::string calibDir;            // INT8 校准目录
    size_t enginePoolSize = 3;
};

class TensorRTBuilder {
public:
    TensorRTBuilder();
    ~TensorRTBuilder();

    Status buildEngine(const BuildConfig& cfg, std::string& outEnginePath);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 tensorrt_builder.cpp**

```cpp
#include "tensorrt_builder.h"
#include "int8_calibrator.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <filesystem>

namespace aicore {

// 简单日志器
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TensorRTBuilder::Impl {
public:
    TRTLogger logger_;
    std::unique_ptr<Int8Calibrator> calibrator_;
};

TensorRTBuilder::TensorRTBuilder() : impl_(std::make_unique<Impl>()) {}
TensorRTBuilder::~TensorRTBuilder() = default;

Status TensorRTBuilder::buildEngine(const BuildConfig& cfg,
                                     std::string& outEnginePath) {
    std::filesystem::create_directories(cfg.outputDir);
    std::string baseName = std::filesystem::path(cfg.onnxPath).stem().string();
    outEnginePath = cfg.outputDir + "/" + baseName + ".engine";

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(impl_->logger_));
    if (!builder) return {StatusCode::ErrorInternal, "Create builder failed"};

    auto flags = 1U << static_cast<int>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(flags));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, impl_->logger_));

    // 解析 ONNX
    if (!parser->parseFromFile(cfg.onnxPath.c_str(),
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return {StatusCode::ErrorConfigParse, "ONNX parse failed"};
    }

    // 设置动态 batch
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           {1, 3, cfg.inputHeight, cfg.inputWidth});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           {cfg.batchSize, 3, cfg.inputHeight, cfg.inputWidth});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           {cfg.maxBatchSize, 3, cfg.inputHeight, cfg.inputWidth});
    config->addOptimizationProfile(profile);

    // 设置精度
    if (cfg.precision == "fp16") {
        if (!builder->platformHasFastFp16()) {
            std::cerr << "[WARN] GPU does not support fast FP16" << std::endl;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (cfg.precision == "int8") {
        if (!builder->platformHasFastInt8()) {
            return {StatusCode::ErrorInternal, "GPU does not support INT8"};
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        impl_->calibrator_ = std::make_unique<Int8Calibrator>(
            cfg.calibDir, cfg.batchSize, cfg.inputWidth, cfg.inputHeight);
        config->setInt8Calibrator(impl_->calibrator_.get());
    }

    // 设置最大工作空间
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // 构建引擎
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        return {StatusCode::ErrorInternal, "Build engine failed"};
    }

    // 保存
    std::ofstream f(outEnginePath, std::ios::binary);
    f.write(static_cast<const char*>(serialized->data()), serialized->size());

    return Status{};
}

} // namespace aicore
```

- [ ] **Step 3: 测试（mock TensorRT）**

```cpp
#include <gtest/gtest.h>
#include "tensorrt_builder.h"

using namespace aicore;

TEST(TensorRTBuilderTest, ConfigValidation) {
    TensorRTBuilder builder;
    BuildConfig cfg;
    cfg.onnxPath = "nonexistent.onnx";
    cfg.precision = "fp16";
    std::string outPath;
    auto status = builder.buildEngine(cfg, outPath);
    // 预期失败（文件不存在），但不崩溃
    EXPECT_FALSE(status);
}
```

---

### Task 6: Int8Calibrator

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\include\int8_calibrator.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\int8_calibrator.cpp`

- [ ] **Step 1: 创建 int8_calibrator.h**

```cpp
#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

namespace aicore {

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(const std::string& calibDir, int batchSize,
                   int inputWidth, int inputHeight);
    ~Int8Calibrator() override;

    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache,
                               size_t length) noexcept override;

private:
    std::vector<std::string> calibImages_;
    int batchSize_;
    int inputWidth_, inputHeight_;
    int currentBatch_ = 0;
    std::vector<float> batchData_;
    std::string cachePath_;
    void* deviceInput_ = nullptr;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 int8_calibrator.cpp**

```cpp
#include "int8_calibrator.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace aicore {

Int8Calibrator::Int8Calibrator(const std::string& calibDir, int batchSize,
                                int inputWidth, int inputHeight)
    : batchSize_(batchSize), inputWidth_(inputWidth), inputHeight_(inputHeight) {
    // 收集校准图片
    for (auto& entry : std::filesystem::directory_iterator(calibDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            calibImages_.push_back(entry.path().string());
        }
    }
    cachePath_ = calibDir + "/calibration.cache";
    size_t dataSize = batchSize * 3 * inputHeight * inputWidth * sizeof(float);
    batchData_.resize(dataSize / sizeof(float));
    cudaMalloc(&deviceInput_, dataSize);
}

Int8Calibrator::~Int8Calibrator() {
    if (deviceInput_) cudaFree(deviceInput_);
}

int Int8Calibrator::getBatchSize() const noexcept { return batchSize_; }

bool Int8Calibrator::getBatch(void* bindings[], const char*[],
                               int) noexcept {
    if (currentBatch_ >= (int)calibImages_.size()) return false;

    int imgCount = 0;
    for (int i = 0; i < batchSize_ && currentBatch_ < (int)calibImages_.size();
         ++i, ++currentBatch_) {
        cv::Mat img = cv::imread(calibImages_[currentBatch_]);
        cv::resize(img, img, cv::Size(inputWidth_, inputHeight_));
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);

        // HWC → CHW
        std::vector<cv::Mat> ch(3);
        cv::split(img, ch);
        for (int c = 0; c < 3; ++c) {
            memcpy(batchData_.data() + i * 3 * inputWidth_ * inputHeight_
                   + c * inputWidth_ * inputHeight_,
                   ch[c].data, inputWidth_ * inputHeight_ * sizeof(float));
        }
        imgCount++;
    }

    size_t copySize = imgCount * 3 * inputWidth_ * inputHeight_ * sizeof(float);
    cudaMemcpy(deviceInput_, batchData_.data(), copySize, cudaMemcpyHostToDevice);
    bindings[0] = deviceInput_;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    std::ifstream f(cachePath_, std::ios::binary);
    if (!f) return nullptr;
    f.seekg(0, std::ios::end);
    length = f.tellg();
    f.seekg(0, std::ios::beg);
    static std::vector<char> cache;
    cache.resize(length);
    f.read(cache.data(), length);
    return cache.data();
}

void Int8Calibrator::writeCalibrationCache(const void* cache,
                                            size_t length) noexcept {
    std::ofstream f(cachePath_, std::ios::binary);
    f.write(static_cast<const char*>(cache), length);
}

} // namespace aicore
```

---

### Task 7: ModelOptimizer + DLL C API

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\include\model_optimizer.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\model_optimizer.cpp`
- Create: `D:\w\AIEngine\aicore_optimizer\include\optimizer_api.h`
- Create: `D:\w\AIEngine\aicore_optimizer\src\optimizer_api.cpp`

- [ ] **Step 1: 创建 model_optimizer.h**

```cpp
#pragma once
#include "core/types.h"
#include "python_embedding.h"
#include "onnx_exporter.h"
#include "tensorrt_builder.h"

namespace aicore {

class ModelOptimizer {
public:
    ModelOptimizer();
    ~ModelOptimizer();

    Status optimize(const Config& config);

private:
    PythonEmbedding py_;
    OnnxExporter exporter_{&py_};
    TensorRTBuilder builder_;
};

} // namespace aicore
```

- [ ] **Step 2: 创建 model_optimizer.cpp**

```cpp
#include "model_optimizer.h"

namespace aicore {

ModelOptimizer::ModelOptimizer() = default;
ModelOptimizer::~ModelOptimizer() = default;

Status ModelOptimizer::optimize(const Config& config) {
    // 1. 初始化 Python
    auto status = py_.initialize(
        config.value("python_home", ""),
        config.value("python_path", ""));
    if (!status) return status;

    // 2. 导出 ONNX
    ExportConfig expCfg;
    expCfg.modelPath = config["model_path"];
    expCfg.modelType = config.value("model_type", "torchscript");
    expCfg.outputDir = config.value("output_dir", "models/optimized");
    expCfg.inputWidth = config["input"].value("width", config.value("input_width", 640));
    expCfg.inputHeight = config["input"].value("height", config.value("input_height", 640));
    expCfg.batchSize = config.value("batch_size", 1);
    expCfg.dynamicBatch = config.value("dynamic_batch", true);
    expCfg.opsetVersion = config.value("opset_version", 17);
    expCfg.device = config.value("device", "cuda:0");

    std::string onnxPath;
    status = exporter_.exportToOnnx(expCfg, onnxPath);
    if (!status) return status;

    // 3. 构建 TensorRT 引擎
    BuildConfig buildCfg;
    buildCfg.onnxPath = onnxPath;
    buildCfg.outputDir = expCfg.outputDir;
    buildCfg.inputWidth = expCfg.inputWidth;
    buildCfg.inputHeight = expCfg.inputHeight;
    buildCfg.precision = config.value("precision", "fp16");
    buildCfg.calibDir = config.value("calib_dir", "");
    buildCfg.enginePoolSize = config.value("engine_pool_size", 3);

    std::string enginePath;
    status = builder_.buildEngine(buildCfg, enginePath);
    if (!status) return status;

    return Status{};
}

} // namespace aicore
```

- [ ] **Step 3: 创建 optimizer_api.h**

```cpp
#pragma once
#include "core/types.h"

extern "C" {

AICORE_API int AICore_Optimize(const char* configPath);
AICORE_API int AICore_ExportOnnx(const char* configPath);
AICORE_API int AICore_BuildEngine(const char* configPath);
AICORE_API void AICore_DestroyOptimizer();
AICORE_API const char* AICore_GetLastError();

}
```

- [ ] **Step 4: 创建 optimizer_api.cpp**

```cpp
#include "optimizer_api.h"
#include "model_optimizer.h"
#include <mutex>
#include <string>

namespace {
    std::unique_ptr<aicore::ModelOptimizer> gOptimizer;
    std::string gLastError;
    std::mutex gMutex;
}

int AICore_Optimize(const char* configPath) {
    std::lock_guard<std::mutex> lock(gMutex);
    try {
        std::ifstream f(configPath);
        if (!f.is_open()) {
            gLastError = "Cannot open config: " + std::string(configPath);
            return -1;
        }
        Config config;
        f >> config;

        gOptimizer = std::make_unique<aicore::ModelOptimizer>();
        auto status = gOptimizer->optimize(config);
        if (!status) {
            gLastError = status.message;
            return static_cast<int>(status.code);
        }
        return 0;
    } catch (const std::exception& e) {
        gLastError = e.what();
        return -2;
    }
}

void AICore_DestroyOptimizer() {
    std::lock_guard<std::mutex> lock(gMutex);
    gOptimizer.reset();
}

const char* AICore_GetLastError() {
    return gLastError.c_str();
}
```

---

### Task 8: CLI + 编译验证

**Files:**
- Create: `D:\w\AIEngine\aicore_optimizer\cli\main.cpp`
- Modify: `D:\w\AIEngine\aicore_optimizer\CMakeLists.txt`

- [ ] **Step 1: 创建 CLI main.cpp**

```cpp
#include "optimizer_api.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "  ModelOptimizer.exe --config config.json" << std::endl;
        std::cerr << "  ModelOptimizer.exe --export --input model.pt --output model.onnx" << std::endl;
        std::cerr << "  ModelOptimizer.exe --build --input model.onnx --output model.engine --precision fp16" << std::endl;
        return 1;
    }

    int ret = AICore_Optimize(argv[2]);
    if (ret != 0) {
        std::cerr << "Error: " << AICore_GetLastError() << std::endl;
        return ret;
    }
    std::cout << "Optimize completed successfully." << std::endl;
    AICore_DestroyOptimizer();
    return 0;
}
```

- [ ] **Step 2: 编译验证**

Run: `cd D:\w\AIEngine\aicore_optimizer && mkdir build; cd build; cmake ..`
Run: `cmake --build . --config Release`
Expected: aicore_optimizer.dll + ModelOptimizer.exe 生成成功
