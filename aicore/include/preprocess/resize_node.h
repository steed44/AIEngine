// 图像缩放节点 — 将输入帧缩放到模型所需的固定尺寸
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// 图像缩放处理器
// 使用 OpenCV 的 resize 将输入图像调整到目标宽高
class ResizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    int targetWidth_ = 640;           // 目标宽度（像素）
    int targetHeight_ = 640;          // 目标高度（像素）
    int interpolation_ = cv::INTER_LINEAR;  // 插值方式（默认双线性）
};

} // namespace aicore
