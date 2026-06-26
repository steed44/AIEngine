// ============================================================
// resize_node.h — 图像缩放节点声明
//
// 功能：将输入帧缩放到模型所需的固定尺寸。
// 与 LetterboxNode 不同，ResizeNode 直接拉伸图像至目标尺寸，
// 不保持原始长宽比，也不传递坐标映射参数。
// ============================================================
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// 图像缩放处理器：使用 OpenCV 的 cv::resize 将输入图像调整到目标宽高
// 支持多种插值方式（默认双线性 INTER_LINEAR）
class ResizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    int targetWidth_ = 640;                 // 目标宽度（像素）
    int targetHeight_ = 640;                // 目标高度（像素）
    int interpolation_ = cv::INTER_LINEAR;  // 插值方式：LINEAR（默认）、NEAREST、CUBIC、AREA
};

} // namespace aicore
