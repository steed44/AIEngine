#pragma once
#include "core/processor.h"

namespace aicore {

class NmsNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    float iouThreshold_ = 0.45f;
    float confidenceThreshold_ = 0.5f;
};

} // namespace aicore
