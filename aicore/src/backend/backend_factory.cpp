// ============================================================
// backend_factory.cpp — 后端引擎工厂
// 根据 BackendType 创建对应的真实后端实现
// ============================================================

#include "backend/backend_factory.h"

#ifdef AICORE_HAS_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

#ifdef AICORE_USE_ONNXRUNTIME
#include "backend/onnxruntime_backend.cpp"
#endif

namespace aicore {

#ifdef AICORE_HAS_LIBTORCH
class LibTorchBackend : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }

    Status Load(const ModelInfo& info) override {
        modelPath_ = info.modelPath;
        deviceId_ = info.deviceId;
        try {
            torch::DeviceType deviceType = torch::kCPU;
            if (deviceId_ >= 0 && torch::cuda::is_available()) {
                deviceType = torch::kCUDA;
            }
            module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath_, deviceType));
            module_->to(deviceType);
            module_->eval();
            loaded_ = true;
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelLoad, std::string("LibTorch load failed: ") + e.what()};
        }
    }

    Status Infer(const std::vector<Tensor>& inputs,
                 std::vector<Tensor>& outputs) override {
        if (!loaded_ || !module_) {
            return Status{StatusCode::ErrorModelLoad, "LibTorch model not loaded"};
        }
        try {
            std::vector<torch::jit::IValue> ivInputs;
            for (auto& t : inputs) {
                auto tensor = torch::from_blob(t.data, torch::IntArrayRef(t.shape.data(), t.shape.size()),
                    torch::TensorOptions(torch::kFloat32));
                ivInputs.push_back(tensor);
            }
            auto results = module_->forward(ivInputs);
            // script::Module::forward returns IValue which may be Tensor or Tuple<IValue>
            if (results.isTensor()) {
                auto out = results.toTensor();
                auto cpuTensor = out.contiguous().cpu();
                auto count = cpuTensor.numel();
                Tensor t;
                t.dtype = DataType::kFloat32;
                t.shape.assign(cpuTensor.sizes().begin(), cpuTensor.sizes().end());
                t.bytes = count * sizeof(float);
                t.memory = MemoryType::kCPU;
                auto* buf = new float[count];
                std::memcpy(buf, cpuTensor.data_ptr<float>(), t.bytes);
                t.data = buf;
                t.allocId = 1;
                outputs.push_back(t);
            } else if (results.isTuple()) {
                for (auto& elem : results.toTuple()->elements()) {
                    if (elem.isTensor()) {
                        auto out = elem.toTensor();
                        auto cpuTensor = out.contiguous().cpu();
                        auto count = cpuTensor.numel();
                        Tensor t;
                        t.dtype = DataType::kFloat32;
                        t.shape.assign(cpuTensor.sizes().begin(), cpuTensor.sizes().end());
                        t.bytes = count * sizeof(float);
                        t.memory = MemoryType::kCPU;
                        auto* buf = new float[count];
                        std::memcpy(buf, cpuTensor.data_ptr<float>(), t.bytes);
                        t.data = buf;
                        t.allocId = 1;
                        outputs.push_back(t);
                    }
                }
            }
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelInfer, std::string("LibTorch infer failed: ") + e.what()};
        }
    }

    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "shape introspection not available"};
    }

    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "shape introspection not available"};
    }

    void SetDeviceId(int deviceId) override { deviceId_ = deviceId; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }

private:
    std::string modelPath_;
    int deviceId_ = 0;
    bool loaded_ = false;
    std::unique_ptr<torch::jit::script::Module> module_;
};
#endif

class TensorRTBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kTensorRT; }
    Status Load(const ModelInfo& info) override { loaded_ = true; return Status{}; }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorModelInfer, "TensorRT backend not yet implemented"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "TensorRT backend not yet implemented"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorModelLoad, "TensorRT backend not yet implemented"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

class ONNXRuntimeBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }
    Status Load(const ModelInfo& info) override { loaded_ = true; return Status{}; }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

class LibTorchBackendStub : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kLibTorch; }
    Status Load(const ModelInfo& info) override { loaded_ = true; return Status{}; }
    Status Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "LibTorch not available"};
    }
    void SetDeviceId(int id) override { deviceId_ = id; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }
protected:
    int deviceId_ = 0;
    bool loaded_ = false;
};

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
#ifdef AICORE_HAS_LIBTORCH
        return std::make_unique<LibTorchBackend>();
#else
        return std::make_unique<LibTorchBackendStub>();
#endif
    default:
        return nullptr;
    }
}

} // namespace aicore
