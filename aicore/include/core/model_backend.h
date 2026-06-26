// 模型后端抽象接口
// 统一不同推理框架（TensorRT、ONNX Runtime、LibTorch）的加载和推理操作
//
// 设计模式：适配器模式（Adapter）
//   IModelBackend 将各推理框架（TRT/ORT/LibTorch）的 API 差异适配为统一接口，
//   上游代码仅依赖此接口，无需关心具体后端实现
//
// 线程安全：
//   Load()      — 非线程安全，在 pipeline 构建阶段单线程调用
//   Infer()     — 线程安全，需子类实现内部互斥或引擎池
//   只读方法    — const 方法，线程安全
//
// 所有权约定：
//   输入 Tensor：data 指针指向调用方内存，后端只读不写不管理
//   输出 Tensor：后端分配 data 缓冲区（allocId=1），调用方通过 delete[] 释放
#pragma once
#include "core/types.h"
#include <string>
#include <vector>
#include <map>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 支持的推理后端类型枚举
//   kTensorRT:    NVIDIA TensorRT 推理引擎
//   kONNXRuntime: 微软 ONNX Runtime 推理引擎
//   kLibTorch:    PyTorch LibTorch 推理引擎
//   kUnknown:     未知/未指定的后端类型
//
// JSON 配置映射：
//   "tensorrt"    → kTensorRT
//   "onnxruntime" → kONNXRuntime
//   "libtorch"    → kLibTorch
enum class BackendType { kTensorRT, kONNXRuntime, kLibTorch, kUnknown };

// 模型元信息结构体
// 描述一个推理模型的所有加载参数和输入输出规格
struct ModelInfo {
    std::string modelPath;                      // 模型文件路径（如 .onnx / .engine / .pt）
    std::string enginePath;                     // TensorRT 引擎文件缓存路径（可选）
    std::vector<int64_t> inputShape;            // 模型输入张量形状
    std::vector<int64_t> outputShape;           // 模型输出张量形状
    DataType inputDtype = DataType::kFloat32;   // 输入张量数据类型，默认 float32
    DataType outputDtype = DataType::kFloat32;  // 输出张量数据类型，默认 float32
    int batchSize = 1;                          // 批处理大小，默认 1
    int deviceId = 0;                           // GPU 设备 ID，默认 0
    BackendType backend = BackendType::kUnknown;// 推理后端类型
    int numInputs = 1;                          // 模型输入数量，默认 1
    int numOutputs = 1;                         // 模型输出数量，默认 1
};

// 模型后端的抽象接口
// 封装模型加载、推理执行和元数据查询，屏蔽不同推理框架的差异
class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    // 加载模型到指定设备
    // @param info 模型元信息，包含路径、形状、后端类型等
    // @return 成功返回 Status::kOk
    // 前置条件：ModelInfo 中 modelPath 非空，文件可访问
    // 后置条件：成功时 IsLoaded() 返回 true；失败时返回错误码，IsLoaded() 为 false
    virtual Status Load(const ModelInfo& info) = 0;
    // 执行推理
    // @param inputs  输入张量列表
    // @param outputs 输出张量列表（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：Load() 已成功调用
    // 前置条件：inputs 中每个 Tensor 的 data 非空，shape 匹配模型输入
    // 后置条件：outputs 中每个 Tensor 的 data 指向新分配的独立缓冲区（调用方 delete[] 释放）
    // 后置条件：outputs 中每个 Tensor 的 allocId == 1（标记后端分配）
    // 线程安全：多线程并发 Infer() 时后端需保证内部状态互斥
    virtual Status Infer(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) = 0;
    // 获取模型输入张量的形状列表
    // @param shapes 形状信息（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：Load() 已成功调用
    // 后置条件：shapes 被替换为模型输入形状列表
    virtual Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const = 0;
    // 获取模型输出张量的形状列表
    // @param shapes 形状信息（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：Load() 已成功调用
    // 后置条件：shapes 被替换为模型输出形状列表
    virtual Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const = 0;
    // 获取当前后端类型
    // @return BackendType 枚举值
    // 线程安全：const 方法，线程安全
    virtual BackendType GetBackendType() const = 0;
    // 设置推理设备 ID
    // @param deviceId GPU 设备编号（-1 表示 CPU）
    // 设计说明：在 Load() 之前调用可指定目标设备，Load() 之后调用仅记录值
    // 线程安全：非线程安全，建议在 Initialize 阶段调用
    virtual void SetDeviceId(int deviceId) = 0;
    // 获取当前推理设备 ID
    // @return 设备编号
    // 线程安全：const 方法，线程安全
    virtual int GetDeviceId() const = 0;
    // 判断模型是否已成功加载
    // @return true 表示已加载就绪
    // 线程安全：const 方法，线程安全
    virtual bool IsLoaded() const = 0;
};

} // namespace aicore
