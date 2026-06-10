#include "postprocess/nms_node.h"
#include <vector>
#include <algorithm>

namespace aicore {

static float Iou(const BBox& a, const BBox& b) {
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    float iy = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    float inter = ix * iy;
    float areaA = a.w * a.h, areaB = b.w * b.h;
    return inter / (areaA + areaB - inter + 1e-6f);
}

Status NmsNode::Init(const NodeConfig& config) {
    auto it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    return Status{};
}

Status NmsNode::Process(const std::vector<Frame>& inputs,
                        std::vector<Frame>& outputs) {
    std::vector<std::pair<float, NodeResult>> candidates;

    for (auto& input : inputs) {
        (void)input;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<bool> removed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (removed[j]) continue;
            if (candidates[i].second.label == candidates[j].second.label &&
                Iou(candidates[i].second.bbox, candidates[j].second.bbox) > iouThreshold_) {
                removed[j] = true;
            }
        }
    }

    outputs.clear();
    return Status{};
}

std::string NmsNode::GetName() const { return "nms"; }
std::string NmsNode::GetType() const { return "nms"; }

} // namespace aicore
