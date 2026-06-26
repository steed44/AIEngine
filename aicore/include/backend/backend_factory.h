// 推理后端工厂类
// 根据后端类型创建对应的 IModelBackend 实例
//
// 设计模式：简单工厂（Simple Factory）
//   BackendFactory::Create() 根据 BackendType 返回对应的后端实现
//   （TensorRT/ONNX Runtime/LibTorch），调用方无需关心具体子类名
//
// 条件编译策略：
//   - TensorRT 后端当前为 Stub（待完整实现）
//   - ONNX Runtime 后端仅在 AICORE_USE_ONNXRUNTIME 定义时编译真实实现
//   - LibTorch 后端仅在 AICORE_HAS_LIBTORCH 定义时编译真实实现
//   未启用的后端返回 Stub，调用时返回明确的"未实现"错误
#pragma once
#include "core/model_backend.h"
#include <memory>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 后端工厂类（静态工具类）
// 封装了 IModelBackend 的创建逻辑，根据 BackendType 选择具体的后端实现
class BackendFactory {
public:
    // 创建指定类型的模型后端实例
    // @param type 后端类型枚举值
    // @return 模型后端的唯一指针，不支持的类型返回 nullptr
    static std::unique_ptr<IModelBackend> Create(BackendType type);
};

// ONNX Runtime 后端创建函数（条件编译）
std::unique_ptr<IModelBackend> CreateONNXRuntimeBackend();

// LibTorch 后端创建函数（条件编译）
std::unique_ptr<IModelBackend> CreateLibTorchBackend();

} // namespace aicore
