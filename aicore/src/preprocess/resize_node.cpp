// ============================================================
// resize_node.cpp — 图像缩放预处理节点
//
// 功能：将任意尺寸的输入帧缩放到模型要求的固定尺寸。
// 与 Letterbox 不同，Resize 直接拉伸图像，不保持长宽比。
//
// 适用场景：
//   - 图像分类任务（长宽比变化对分类影响较小）
//   - 模型本身支持动态输入尺寸
//   - 需要固定尺寸但不在乎长宽比失真的场景
//
// 缺点：
//   对于目标检测任务，直接拉伸会改变目标的形状和长宽比，
//   降低检测精度。此时应使用 LetterboxNode 替代。
// ============================================================
#include "preprocess/resize_node.h"

namespace aicore {

// 初始化缩放节点：从配置读取目标宽高和插值方式
// 支持参数：
//   width         — 目标宽度（像素，默认 640）
//   height        — 目标高度（像素，默认 640）
//   interpolation — 插值方式（默认 INTER_LINEAR=1）
//                   可选：INTER_NEAREST=0, INTER_CUBIC=2, INTER_AREA=3
Status ResizeNode::Init(const NodeConfig& config) {
    auto it = config.find("width");
    if (it != config.end()) targetWidth_ = std::stoi(it->second);
    it = config.find("height");
    if (it != config.end()) targetHeight_ = std::stoi(it->second);
    return Status{};
}

// 执行图像缩放：对每帧调用 OpenCV 的 cv::resize
// 使用双线性插值（默认），缩放后图像尺寸精确等于 targetWidth × targetHeight
// 不保留原始图像的 roiMap 信息（区别于 LetterboxNode）
Status ResizeNode::Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) {
    for (const auto& frame : inputs) {
        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;
        cv::resize(frame.image, out.image,
                   cv::Size(targetWidth_, targetHeight_),
                   0, 0, interpolation_);
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string ResizeNode::GetName() const { return "resize"; }
std::string ResizeNode::GetType() const { return "resize"; }

} // namespace aicore
