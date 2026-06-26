// ============================================================
// nms_common.h — NMS 通用工具：IouBox + NMSCommon
// 供 YoloDecodeNode 和 NmsNode 共享，避免重复实现。
// 默认使用中心点坐标 (cx, cy, w, h) 格式的 BBox。
// ============================================================
#pragma once
#include "core/types.h"
#include <vector>

namespace aicore {

// 计算两个 BBox 的 IoU（中心点坐标格式 → 左上角/右下角 转换）
//
// 输入格式：BBox 使用中心点坐标 (cx, cy, w, h)
//   其中 cx, cy 是框中心坐标，w, h 是框的宽和高
//
// 计算步骤：
//   1. 将中心点格式 (cx, cy, w, h) 转换为左上/右下角格式 (x1,y1,x2,y2)
//      x1 = cx - w/2, y1 = cy - h/2
//      x2 = cx + w/2, y2 = cy + h/2
//   2. 计算交集区域的宽和高
//      ix = max(0, min(ax2,bx2) - max(ax1,bx1))
//      iy = max(0, min(ay2,by2) - max(ay1,by1))
//      若两个框不重叠，ix 或 iy 为 0（通过 max(0, ...) 保证非负）
//   3. IoU = 交集面积 / 并集面积
//      inter = ix * iy
//      union = areaA + areaB - inter
//      IoU = inter / (union + 1e-6f)
//
// 为什么用并集面积 = A + B - inter？
//   两个框的并集 = 框 A 面积 + 框 B 面积 - 重叠部分面积（交集被重复计算了两次）
//
// +1e-6f 防止除零（当 areaA = areaB = inter = 0 时分母为 0）
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
