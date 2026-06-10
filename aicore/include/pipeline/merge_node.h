#pragma once
#include "core/processor.h"
#include <vector>
#include <string>

namespace aicore {

class MergeNode : public IProcessor {
public:
    MergeNode();
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    std::string mergeMode_ = "concat";
    int maxInputs_ = 0;
};

} // namespace aicore
