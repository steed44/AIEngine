// Letterbox 预处理节点 — YOLO 推理的等比例缩放 + 灰边填充
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

class LetterboxNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    int targetWidth_ = 640;
    int targetHeight_ = 640;
    cv::Scalar padColor_ = cv::Scalar(114, 114, 114);
};

} // namespace aicore
