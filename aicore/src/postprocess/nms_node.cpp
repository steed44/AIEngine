// ============================================================
// nms_node.cpp — 非极大值抑制后处理节点
// 通过 IoU 阈值过滤重叠检测框，保留置信度最高的预测
// ============================================================
#include "postprocess/nms_node.h"
#include "postprocess/nms_common.h"
#include <vector>
#include <algorithm>

namespace aicore {

Status NmsNode::Init(const NodeConfig& config) {
    auto it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

/**
 * NMS 后处理：从输入帧的 detections 中提取候选框，
 * 按置信度降序排列后贪心抑制重叠框，输出过滤后的结果。
 */
Status NmsNode::Process(const std::vector<Frame>& inputs,
                        std::vector<Frame>& outputs) {
    std::vector<NodeResult> allDetections;
    for (auto& input : inputs) {
        for (auto& det : input.detections) {
            if (det.confidence >= confidenceThreshold_)
                allDetections.push_back(std::move(det));
        }
    }

    NMSCommon(allDetections, iouThreshold_);

    Frame out;
    out.frameId = inputs.empty() ? 0 : inputs[0].frameId;
    out.timestamp = inputs.empty() ? 0 : inputs[0].timestamp;
    out.detections = std::move(allDetections);
    outputs.push_back(std::move(out));
    return Status{};
}

std::string NmsNode::GetName() const { return "nms"; }
std::string NmsNode::GetType() const { return "nms"; }

} // namespace aicore
