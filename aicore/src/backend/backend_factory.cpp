#include "backend/backend_factory.h"

namespace aicore {

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

class TensorRTBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kTensorRT; }
};

class ONNXRuntimeBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }
};

class LibTorchBackendStub : public StubBase {
public:
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }
};

std::unique_ptr<IModelBackend> BackendFactory::Create(BackendType type) {
    switch (type) {
    case BackendType::kTensorRT:
        return std::make_unique<TensorRTBackendStub>();
    case BackendType::kONNXRuntime:
        return std::make_unique<ONNXRuntimeBackendStub>();
    case BackendType::kLibTorch:
        return std::make_unique<LibTorchBackendStub>();
    default:
        return nullptr;
    }
}

} // namespace aicore
