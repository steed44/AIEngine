// ============================================================
// normalize_node.h — 图像归一化节点声明
//
// 功能：执行标准的深度学习图像预处理流程：
//   1. BGR→RGB 通道重排（可选，OpenCV 默认 BGR，模型通常用 RGB）
//   2. uint8 [0,255] → float32 [0,1] 缩放
//   3. 逐通道标准化：(x - mean_c) / std_c
//
// 归一化参数使用 ImageNet 数据集统计量：
//   mean = [0.485, 0.456, 0.406]   // RGB 各通道均值
//   std  = [0.229, 0.224, 0.225]   // RGB 各通道标准差
// ============================================================
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// 图像归一化处理器
// 执行: BGR→RGB 转换（可选）、uint8→float32 缩放、逐通道 (x-mean)/std 标准化
// 在流水线中通常位于 LetterboxNode 或 ResizeNode 之后：
//   [输入图像] → [Letterbox/Resize] → [NormalizeNode] → [模型推理]
class NormalizeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    // ImageNet 数据集归一化参数（RGB 顺序）
    // 所有在 ImageNet 上预训练的模型都使用相同的 mean/std
    float mean_[3] = {0.485f, 0.456f, 0.406f};  // RGB 均值
    float std_[3] = {0.229f, 0.224f, 0.225f};    // RGB 标准差
    bool bgrToRgb_ = false;  // 是否执行 BGR→RGB 转换（OpenCV→模型约定）
};

} // namespace aicore
