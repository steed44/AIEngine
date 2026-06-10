# Phase 2: 模型优化工具 设计文档

## 概述

模型优化工具提供 PyTorch→ONNX→TensorRT 的全流程模型转换与优化能力。采用 C++ 内嵌 Python 解释器完成 ONNX 导出，C++ 原生调用 TensorRT API 构建推理引擎。同时提供独立 CLI 和 DLL 两种使用形态。

优化器 DLL（`aicore_optimizer.dll`）**独立于推理引擎 DLL（`aicore.dll`）**，两者无链接依赖。优化器仅在模型转换阶段被调用，推理时不需要加载。

## 总体架构

```
CLI 层 (ModelOptimizer.exe)
    │
Optimizer DLL (aicore_optimizer.dll)    ← 独立 DLL，不依赖 aicore.dll
    ├── Python 嵌入层 (PythonEmbedding)
    ├── ONNX 导出器 (OnnxExporter)
    ├── TensorRT 构建器 (TensorRTBuilder)
    ├── INT8 校准器 (Int8Calibrator)
    └── ModelOptimizer 编排器
```

## 公共类型定义

与 Phase 1 一致，使用 `aicore` 命名空间：

```cpp
#include "core/types.h"        // Status, StatusCode, AICORE_API
using Config = nlohmann::json;  // 与 Phase 1 一致
```

错误码映射：C API 返回 `int`，`0=OK`，负数=系统错误（`-1` 参数错误，`-2` 资源不足），正数=StatusCode 枚举值。

## Python 嵌入层

```cpp
class PythonEmbedding {
public:
    Status initialize(const std::string& pythonHome,
                      const std::string& pythonPath);
    Status execFile(const std::string& scriptPath);
    Status execString(const std::string& code);
    Status callFunction(const std::string& module,
                        const std::string& func,
                        const Config& args, Config& result);
    void finalize();
};
```

### GIL 管理

所有调用 Python 的 C++ 函数（`execFile`、`execString`、`callFunction`）内部必须：
1. `PyGILState_STATE gstate = PyGILState_Ensure();`
2. 执行 Python 代码
3. `PyGILState_Release(gstate);`

### 线程安全

- 使用 `std::once_flag` + `std::call_once` 确保解释器只初始化一次
- `callFunction` / `execFile` 内部加 mutex 保护，防止多线程同时调用 Python
- `finalize()` 只能在拥有解释器锁的线程调用

### 生命周期

- `DLL_PROCESS_ATTACH` 时不初始化（Windows loader lock 下不能调 Python）
- 第一次调用 `AICore_Optimize()` 时通过 `std::call_once` 延迟初始化
- `finalize()` 在 `AICore_DestroyOptimizer()` 中调用

## ONNX 导出流程

Python 端脚本（`scripts/export_onnx.py`）被 C++ 调用执行：

1. `import torch, onnx, onnxsim`
2. 加载模型（TorchScript / 自定义等）
3. 创建 dummy input
4. `torch.onnx.export()` 导出，开启动态 batch
5. `onnxsim.simplify()` 精简计算图
6. 返回 ONNX 文件路径 + 校验结果

## TensorRT 引擎构建

使用 NVIDIA TensorRT C++ API：

1. 创建 `IBuilder` + `INetworkDefinition` + `IBuilderConfig`
2. `IONNXParser` 解析 ONNX 文件
3. 设置 `OptimizationProfile`（动态 batch）
4. 设置精度模式（FP32 / FP16 / INT8）
5. INT8 模式时设置 `IInt8Calibrator`
6. `buildSerializedNetwork()` 构建引擎
7. 序列化保存为 `.engine` 文件

### INT8 校准器

```cpp
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(const std::string& calibDir, int batchSize,
                   int inputWidth, int inputHeight);

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
};
```

校准数据集通过 `config.json` 的 `calib_dir` 字段传入，从目录中读取典型工业检测样本。校准缓存保存到磁盘避免重复校准。

## DLL C 接口

```cpp
extern "C" {
AICORE_API int AICore_Optimize(const char* configPath);
AICORE_API int AICore_ExportOnnx(const char* configPath);
AICORE_API int AICore_BuildEngine(const char* configPath);
AICORE_API void AICore_DestroyOptimizer();     // 释放 Python 解释器 + CUDA 资源
AICORE_API const char* AICore_GetLastError();
}
```

错误码：`0=OK`，负数=系统错误（`-1` 参数错误，`-2` 资源不足），正数=StatusCode 枚举值。

## CLI 使用方式

```
ModelOptimizer.exe --config config_optimize.json
ModelOptimizer.exe --export --input model.pt --output model.onnx
ModelOptimizer.exe --build --input model.onnx --output model.engine --precision fp16
```

## 测试策略

- 单元测试：mock Python 嵌入层，测试 `TensorRTBuilder` 的配置逻辑
- Python 脚本测试：单独用 pytest 测试 `export_onnx.py` 的导出逻辑
- 集成测试：用预训练小模型（如 YOLOv8n）执行完整导出+构建流程
- Windows CI：在装有 CUDA 11.8 + TensorRT 8.5.3 的机器上执行

## 目录结构

```
aicore_optimizer/
├── include/
│   ├── core/         # 来自 Phase 1 的 types.h（CMake 引用路径）
│   ├── python_embedding.h
│   ├── onnx_exporter.h
│   ├── tensorrt_builder.h
│   ├── int8_calibrator.h
│   ├── model_optimizer.h
│   └── optimizer_api.h
├── scripts/export_onnx.py
├── src/
├── cli/
├── tests/
├── CMakeLists.txt
└── config_optimize.json
```

## 版本兼容

CUDA 11.8 版本配套：
| 依赖 | 版本 |
|------|------|
| CUDA Toolkit | 11.8 |
| TensorRT | 8.5.3 |
| cuDNN | 8.7.0 |
| Python | 3.10+ |
| PyTorch | 2.1.0+cu118 |
| ONNX | 1.14+ |
| onnxsim | 0.4+ |
