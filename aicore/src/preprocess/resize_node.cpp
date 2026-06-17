// ============================================================
// resize_node.cpp — 图像缩放预处理节点
// 将输入帧缩放到模型要求的固定尺寸
// ============================================================
#include "preprocess/resize_node.h"

namespace aicore {

/**
 * 初始化缩放节点：读取目标宽高
 * @param config 节点配置键值对，含 "width" 和 "height"
 */
Status ResizeNode::Init(const NodeConfig& config) {
    auto it = config.find("width");
    if (it != config.end()) targetWidth_ = std::stoi(it->second);
    it = config.find("height");
    if (it != config.end()) targetHeight_ = std::stoi(it->second);
    return Status{};
}

/**
 * 执行图像缩放：对每帧调用 OpenCV 的 cv::resize 进行双线性插值
 * @param inputs  原始尺寸的输入帧
 * @param outputs [out] 缩放后的帧列表
 */
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

/** 返回节点名称 */
std::string ResizeNode::GetName() const { return "resize"; }
/** 返回节点类型标识 */
std::string ResizeNode::GetType() const { return "resize"; }

} // namespace aicore
