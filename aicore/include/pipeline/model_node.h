#pragma once
#include "core/processor.h"
#include "core/model_backend.h"

namespace aicore {

class ModelNode : public IProcessor {
public:
    explicit ModelNode(std::shared_ptr<IModelBackend> backend);
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;
    std::shared_ptr<IModelBackend> GetBackend() const { return backend_; }

private:
    std::shared_ptr<IModelBackend> backend_;
    float confidenceThreshold_ = 0.5f;
};

} // namespace aicore
