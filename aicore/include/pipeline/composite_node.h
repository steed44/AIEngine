#pragma once
#include "core/processor.h"
#include "core/pipeline.h"
#include <memory>

namespace aicore {

class CompositeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

    void SetInnerPipeline(std::unique_ptr<IPipeline> pipeline);

private:
    std::unique_ptr<IPipeline> innerPipeline_;
    std::string name_;
};

} // namespace aicore
