#pragma once
#include "core/types.h"
#include <memory>
#include <string>

namespace aicore {

enum class ModelArch { kSimpleCNN, kResNet18, kCustom };

struct ModelConfig {
    ModelArch arch = ModelArch::kSimpleCNN;
    int numClasses = 10;
    int inputChannels = 3;
    int inputSize = 224;
    std::string pretrainedPath;
};

class IModel {
public:
    virtual ~IModel() = default;
    virtual Status Build(const ModelConfig& config) = 0;
    virtual Status Save(const std::string& path) = 0;
    virtual Status Load(const std::string& path) = 0;
    virtual std::string GetArchName() const = 0;
};

class ModelFactory {
public:
    static std::unique_ptr<IModel> Create(ModelArch arch);
};

} // namespace aicore
