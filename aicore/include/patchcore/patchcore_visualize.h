#pragma once
#include <opencv2/core.hpp>

// ============================================================
// patchcore_visualize.h — PatchCore 可视化接口
// 功能：异常热力图生成、叠加、掩膜提取
// ============================================================

namespace aicore {

// 异常得分 → 伪彩色热力图（COLORMAP_JET）
cv::Mat ColorizeAnomalyMap(const cv::Mat& anomalyMap, float maxVal = 0.0f);

// 热力图叠加到原图（支持阈值模式）
cv::Mat DrawAnomalyOverlay(const cv::Mat& anomalyMap, const cv::Mat& image,
                           float alpha = 0.6f, float threshold = 0.0f);

// 从异常得分图生成二值掩膜
cv::Mat MaskFromAnomalyMap(const cv::Mat& anomalyMap, float threshold);

} // namespace aicore
