// ============================================================
// nms_node.cpp — 非极大值抑制后处理节点
//
// 功能：对 YOLO 解码后的检测框执行 NMS，去除重复检测。
//
// 在流水线中的位置：
//   [YoloDecodeNode] → [NmsNode] → [输出最终检测结果]
//
// 处理流程：
//   1. 根据置信度阈值过滤低分框（减少 NMS 计算量）
//   2. 调用 NMSCommon 执行贪心 NMS（同类别间按 IoU 抑制）
//   3. 输出过滤后的检测框列表
//
// NMS 必要性：
//   原始 YOLO 输出在同一个目标周围会产生大量高重叠的候选框，
//   每个 grid cell 都会预测一个框，导致一个目标被重复检测数十次。
//   NMS 将这些冗余框压缩到每个目标 1-2 个框。
// ============================================================
#include "postprocess/nms_node.h"
#include "postprocess/nms_common.h"
#include <vector>
#include <algorithm>

namespace aicore {

// 初始化：从节点配置读取 NMS 参数
// 支持参数：
//   iou_threshold        — IoU 抑制阈值（默认 0.45）
//   confidence_threshold — 置信度过滤阈值（默认 0.5）
Status NmsNode::Init(const NodeConfig& config) {
    auto it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

// NMS 后处理主函数
// 步骤：
//   1. 置信度过滤：收集所有输入帧中置信度 >= threshold 的检测框
//   2. 调用 NMSCommon 执行同类别非极大值抑制（按 IoU 阈值判断）
//   3. 输出紧凑的检测框列表到输出帧
Status NmsNode::Process(const std::vector<Frame>& inputs,
                        std::vector<Frame>& outputs) {
    // 步骤 1：置信度过滤
    // 提前丢弃低分框，减少 NMS 中 O(n²) 的计算量
    std::vector<NodeResult> allDetections;
    for (auto& input : inputs) {
        for (auto& det : input.detections) {
            if (det.confidence >= confidenceThreshold_)
                allDetections.push_back(std::move(det));
        }
    }

    // 步骤 2：NMS 贪心抑制
    // 按置信度排序 → 同标签内 IoU 超阈值则丢弃低分框
    NMSCommon(allDetections, iouThreshold_);

    // 步骤 3：组装输出帧
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
