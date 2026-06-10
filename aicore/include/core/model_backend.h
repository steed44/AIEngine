#pragma once
#include "core/types.h"
#include <string>
#include <vector>
#include <map>

namespace aicore {

enum class BackendType { kTensorRT, kONNXRuntime, kLibTorch, kUnknown };

struct ModelInfo {
    std::string modelPath;
    std::string enginePath;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    DataType inputDtype = DataType::kFloat32;
    DataType outputDtype = DataType::kFloat32;
    int batchSize = 1;
    int deviceId = 0;
    BackendType backend = BackendType::kUnknown;
    int numInputs = 1;
    int numOutputs = 1;
};

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual Status Load(const ModelInfo& info) = 0;
    virtual Status Infer(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs) = 0;
    virtual Status GetInputShapes(std::vector<std::vector<int64_t>>& shapes) const = 0;
    virtual Status GetOutputShapes(std::vector<std::vector<int64_t>>& shapes) const = 0;
    virtual BackendType GetBackendType() const = 0;
    virtual void SetDeviceId(int deviceId) = 0;
    virtual int GetDeviceId() const = 0;
    virtual bool IsLoaded() const = 0;
};

} // namespace aicore
