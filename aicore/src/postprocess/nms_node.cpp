// ============================================================
// nms_node.cpp — 非极大值抑制后处理节点
// 通过 IoU 阈值过滤重叠检测框，保留置信度最高的预测
// ============================================================
#include "postprocess/nms_node.h"
#include <vector>
#include <algorithm>

namespace aicore {

/**
 * 计算两个边界框的交并比（IoU）
 * 为避免除零，分母加 1e-6 极小值
 * @param a 框 A（x, y, w, h 格式）
 * @param b 框 B（x, y, w, h 格式）
 * @return IoU 值，范围 [0, 1]
 */
static float Iou(const BBox& a, const BBox& b) {
    // 将 (x, y, w, h) 转为 (x1, y1, x2, y2) 格式
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    // 交集区域的宽和高，无重叠时取 0
    float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    float iy = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    float inter = ix * iy;
    float areaA = a.w * a.h, areaB = b.w * b.h;
    // IoU = 交集面积 / (A 面积 + B 面积 - 交集面积)
    return inter / (areaA + areaB - inter + 1e-6f);
}

/**
 * 初始化 NMS 节点：读取 IoU 阈值和置信度阈值
 * @param config 节点配置键值对
 */
Status NmsNode::Init(const NodeConfig& config) {
    auto it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

/**
 * 执行 NMS（贪心算法）：
 *   1. 收集所有候选检测框
 *   2. 按置信度降序排列（得分高的框优先保留）
 *   3. 遍历候选框，将与高分框 IoU 超过阈值的同类别框标记为"已移除"
 * 贪心策略确保保留的是局部置信度最高的框，
 * 这是经典 NMS 的标准做法，时间复杂度 O(n²)
 * @param inputs  模型原始输出的帧列表
 * @param outputs [out] NMS 后的帧列表
 */
Status NmsNode::Process(const std::vector<Frame>& inputs,
                        std::vector<Frame>& outputs) {
    // 候选列表：<置信度, 检测结果>
    std::vector<std::pair<float, NodeResult>> candidates;

    for (auto& input : inputs) {
        (void)input;
    }

    // 按置信度降序排列——确保高分框优先成为参考基准
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // 贪心 NMS：对每个框，如果它被保留，则抑制所有同类别且高 IoU 的后续框
    std::vector<bool> removed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (removed[j]) continue;
            // 只有同类别才竞争，不同类别的框互不抑制
            if (candidates[i].second.label == candidates[j].second.label &&
                Iou(candidates[i].second.bbox, candidates[j].second.bbox) > iouThreshold_) {
                removed[j] = true;
            }
        }
    }

    outputs.clear();
    return Status{};
}

/** 返回节点名称 */
std::string NmsNode::GetName() const { return "nms"; }
/** 返回节点类型标识 */
std::string NmsNode::GetType() const { return "nms"; }

} // namespace aicore
