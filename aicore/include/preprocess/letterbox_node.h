// ============================================================
// letterbox_node.h — Letterbox 预处理节点声明
//
// 功能：对输入图像执行等比例缩放 + 灰边填充，使其适配模型
// 要求的固定尺寸输入。同时保存缩放参数（scale、padX、padY）
// 到 roiMap 中，供下游 YoloDecodeNode 做坐标逆变换使用。
// ============================================================
#pragma once
#include "core/processor.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// Letterbox 预处理节点
// 在流水线中通常位于 Resize 和 Normalize 之间，或替代 Resize：
//   [输入图像] → [LetterboxNode] → [NormalizeNode] → [YOLO 模型]
//
// 相比直接 Resize 的优势：
//   保持长宽比不变，目标形状不失真，检测精度更高。
//   代价是引入灰边区域，模型需要学会忽略填充区域的特征。
class LetterboxNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    int targetWidth_ = 640;               // 目标宽度（像素），建议 32 的倍数
    int targetHeight_ = 640;              // 目标高度（像素），建议 32 的倍数
    cv::Scalar padColor_ = cv::Scalar(114, 114, 114);  // 填充色 (BGR)，YOLO 默认灰
};

} // namespace aicore
