// 图像归一化节点 — 将像素值缩放到模型输入范围并转换颜色空间
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// 图像归一化处理器
// 执行: BGR→RGB 转换（可选）、像素值归一化 (x - mean) / std
class NormalizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    float mean_[3] = {0.485f, 0.456f, 0.406f};  // ImageNet 均值 (RGB)
    float std_[3] = {0.229f, 0.224f, 0.225f};    // ImageNet 标准差 (RGB)
    bool bgrToRgb_ = false;                       // 是否执行 BGR→RGB 转换
};

} // namespace aicore
