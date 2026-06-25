#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <vector>
#include <string>
#include <cstring>

#ifdef AICORE_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace aicore {

class ONNXRuntimeBackend : public IModelBackend {
public:
    BackendType GetBackendType() const override { return BackendType::kONNXRuntime; }

    Status Load(const ModelInfo& info) override {
        modelPath_ = info.modelPath;
        deviceId_ = info.deviceId;
#ifdef AICORE_USE_ONNXRUNTIME
        try {
            env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "aicore-ort");
            Ort::SessionOptions sessionOpts;
            sessionOpts.SetIntraOpNumThreads(4);
            sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            if (deviceId_ >= 0) {
                OrtCUDAProviderOptions cudaOpts;
                cudaOpts.device_id = deviceId_;
                sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
            }

            session_ = std::make_shared<Ort::Session>(*env_, modelPath_.c_str(), sessionOpts);

            Ort::AllocatorWithDefaultOptions allocator;
            inputNames_.clear();
            outputNames_.clear();

            size_t numInputs = session_->GetInputCount();
            size_t numOutputs = session_->GetOutputCount();

            for (size_t i = 0; i < numInputs; i++) {
                auto name = session_->GetInputNameAllocated(i, allocator);
                inputNames_.push_back(name.get());
                auto typeInfo = session_->GetInputTypeInfo(i);
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                inputShapes_.push_back(shapeInfo.GetShape());
            }

            for (size_t i = 0; i < numOutputs; i++) {
                auto name = session_->GetOutputNameAllocated(i, allocator);
                outputNames_.push_back(name.get());
                auto typeInfo = session_->GetOutputTypeInfo(i);
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                outputShapes_.push_back(shapeInfo.GetShape());
            }

            loaded_ = true;
            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelLoad,
                "ONNX Runtime load failed: " + std::string(e.what())};
        }
#else
        (void)info;
        return Status{StatusCode::ErrorModelLoad,
            "ONNX Runtime not available (recompile with AICORE_USE_ONNXRUNTIME)"};
#endif
    }

    Status Infer(const std::vector<Tensor>& inputs,
                 std::vector<Tensor>& outputs) override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_ || !session_) {
            return Status{StatusCode::ErrorModelLoad, "ONNX Runtime not loaded"};
        }
        try {
            std::vector<Ort::Value> ortInputs;
            std::vector<const char*> ortInputNames;
            std::vector<const char*> ortOutputNames;

            for (size_t i = 0; i < inputs.size() && i < inputNames_.size(); i++) {
                auto& t = inputs[i];
                std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
                Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
                    OrtArenaAllocator, OrtMemTypeDefault);
                ortInputs.push_back(Ort::Value::CreateTensor<float>(
                    memInfo, static_cast<float*>(t.data), t.bytes / sizeof(float),
                    shape.data(), shape.size()));
                ortInputNames.push_back(inputNames_[i].c_str());
            }

            for (auto& name : outputNames_) {
                ortOutputNames.push_back(name.c_str());
            }

            auto ortOutputs = session_->Run(Ort::RunOptions{nullptr},
                ortInputNames.data(), ortInputs.data(), ortInputs.size(),
                ortOutputNames.data(), ortOutputNames.size());

            outputs.clear();
            outputs.reserve(ortOutputs.size());
            for (auto& ortOut : ortOutputs) {
                auto typeInfo = ortOut.GetTensorTypeAndShapeInfo();
                auto shape = typeInfo.GetShape();
                auto count = typeInfo.GetElementCount();

                Tensor t;
                t.dtype = DataType::kFloat32;
                t.shape.assign(shape.begin(), shape.end());
                t.bytes = count * sizeof(float);
                t.memory = MemoryType::kCPU;
                t.data = new float[count];
                std::memcpy(t.data, ortOut.GetTensorData<float>(), t.bytes);
                outputs.push_back(t);
            }

            return Status{};
        } catch (const std::exception& e) {
            return Status{StatusCode::ErrorModelInfer,
                "ONNX Runtime infer failed: " + std::string(e.what())};
        }
#else
        (void)inputs; (void)outputs;
        return Status{StatusCode::ErrorInternal,
            "ONNX Runtime not available (recompile with AICORE_USE_ONNXRUNTIME)"};
#endif
    }

    Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_) return Status{StatusCode::ErrorModelLoad, "not loaded"};
        shapes = inputShapes_;
        return Status{};
#else
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
#endif
    }

    Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const override {
#ifdef AICORE_USE_ONNXRUNTIME
        if (!loaded_) return Status{StatusCode::ErrorModelLoad, "not loaded"};
        shapes = outputShapes_;
        return Status{};
#else
        (void)shapes;
        return Status{StatusCode::ErrorInternal, "ONNX Runtime not available"};
#endif
    }

    void SetDeviceId(int deviceId) override { deviceId_ = deviceId; }
    int GetDeviceId() const override { return deviceId_; }
    bool IsLoaded() const override { return loaded_; }

private:
    std::string modelPath_;
    int deviceId_ = 0;
    bool loaded_ = false;
#ifdef AICORE_USE_ONNXRUNTIME
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::vector<std::vector<int64_t>> outputShapes_;
#endif
};

std::unique_ptr<IModelBackend> CreateONNXRuntimeBackend() {
    return std::make_unique<ONNXRuntimeBackend>();
}

} // namespace aicore