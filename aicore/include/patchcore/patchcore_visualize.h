#pragma once
#include <opencv2/core.hpp>

namespace aicore {

// 将浮点得分图转换为伪彩色图 (jet colormap)
// @param anomalyMap  CV_32F 单通道, 任意尺寸
// @param maxVal      归一化上限 (0=自动取实际最大值)
// @return CV_8UC3 伪彩色图 (BGR)
cv::Mat ColorizeAnomalyMap(const cv::Mat& anomalyMap, float maxVal = 0.0f);

// 绘制异常热力图半透明叠加层
// @param anomalyMap  CV_32F 得分图 (需与原图同尺寸)
// @param image       原始 BGR 图像 (CV_8UC3)
// @param alpha       热力图透明度 0.0-1.0 (默认 0.6)
// @param threshold   透明阈值: 低于此值的区域完全透明 (默认 0)
// @return CV_8UC3 叠加结果
cv::Mat DrawAnomalyOverlay(const cv::Mat& anomalyMap, const cv::Mat& image,
                           float alpha = 0.6f, float threshold = 0.0f);

} // namespace aicore
