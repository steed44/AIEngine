#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

class NormalizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    float mean_[3] = {0.485f, 0.456f, 0.406f};
    float std_[3] = {0.229f, 0.224f, 0.225f};
    bool bgrToRgb_ = false;
};

} // namespace aicore
