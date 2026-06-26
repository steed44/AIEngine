// NMS 通用工具 — 供 YoloDecodeNode 和 NmsNode 共享
#pragma once
#include "core/types.h"
#include <vector>

namespace aicore {

// 计算两个 BBox 的 IoU（中心点坐标 → 左上/右下）
static inline float IouBox(const BBox& a, const BBox& b) {
    float ax1 = a.x - a.w / 2.0f, ay1 = a.y - a.h / 2.0f;
    float ax2 = a.x + a.w / 2.0f, ay2 = a.y + a.h / 2.0f;
    float bx1 = b.x - b.w / 2.0f, by1 = b.y - b.h / 2.0f;
    float bx2 = b.x + b.w / 2.0f, by2 = b.y + b.h / 2.0f;

    float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    float iy = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    float inter = ix * iy;
    float areaA = a.w * a.h, areaB = b.w * b.h;
    return inter / (areaA + areaB - inter + 1e-6f);
}

// 通用 NMS：按置信度降序排序，贪心抑制同标签重叠框
// [in/out] candidates — 输入候选框列表，输出保留框列表
void NMSCommon(std::vector<NodeResult>& candidates, float iouThreshold);

} // namespace aicore
