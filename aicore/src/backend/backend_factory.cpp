// ============================================================
// backend_factory.cpp — 后端引擎工厂与桩实现
// 提供 BackendFactory::Create 工厂方法，根据后端类型创建对应的桩实例
// 目前返回空桩（stub），待各后端实际实现后替换
// ============================================================

#include "backend/backend_factory.h"

namespace aicore {

/**
 * StubBase — 后端桩基类
 * 所有桩实现的基类，提供默认的空实现。
 * Load 记录加载标记，Infer 返回"不支持推理"错误，
 * 形状查询返回固定错误，供外部识别桩状态。
 */
class StubBase : public IModelBackend {
public:
    Status Load(const ModelInfo& info) override { loaded_ = true; return Status{}; }
    Status Infer(const std::vector<Tensor>&, std::vector<Tensor>&) override { return Status{StatusCode::ErrorInternal, "stub cannot infer"}; }
    Status GetInputShapes(std::vector<std::vector<int64_t>>&) const override { return Status{StatusCode::ErrorInternal, "stub no shapes"}; }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>&) const override { return Status{StatusCode::ErrorInternal, "stub no shapes"}; }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

/**
 * TensorRT 后端桩 — 标识类型为 TensorRT
 */
class TensorRTBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kTensorRT; }
};

/**
 * ONNX Runtime 后端桩 — 标识类型为 ONNXRuntime
 */
class ONNXRuntimeBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }
};

/**
 * LibTorch 后端桩 — 标识类型为 LibTorch
 */
class LibTorchBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }
};

/**
 * 创建后端引擎实例
 * 根据 BackendType 枚举值创建对应的后端桩实例。
 * 待各后端实际实现后，此处应返回真实的 TensorRT/ONNX/LibTorch 引擎对象。
 * @param type 后端类型枚举
 * @return 后端引擎唯一指针，未知类型返回 nullptr
 */
std::unique_ptr<IModelBackend> BackendFactory::Create(BackendType type) {
    switch (type) {
    case BackendType::kTensorRT:
        return std::make_unique<TensorRTBackendStub>();
    case BackendType::kONNXRuntime:
#ifdef AICORE_USE_ONNXRUNTIME
        return CreateONNXRuntimeBackend();
#else
        return std::make_unique<ONNXRuntimeBackendStub>();
#endif
    case BackendType::kLibTorch:
        return std::make_unique<LibTorchBackendStub>();
    default:
        return nullptr;
    }
}

} // namespace aicore
