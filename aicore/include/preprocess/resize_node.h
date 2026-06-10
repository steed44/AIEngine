#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

class ResizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    int targetWidth_ = 640;
    int targetHeight_ = 640;
    int interpolation_ = cv::INTER_LINEAR;
};

} // namespace aicore
