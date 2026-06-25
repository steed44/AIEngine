#pragma once
#include <opencv2/core.hpp>

namespace aicore {

cv::Mat ColorizeAnomalyMap(const cv::Mat& anomalyMap, float maxVal = 0.0f);
cv::Mat DrawAnomalyOverlay(const cv::Mat& anomalyMap, const cv::Mat& image,
                           float alpha = 0.6f, float threshold = 0.0f);
cv::Mat MaskFromAnomalyMap(const cv::Mat& anomalyMap, float threshold);

} // namespace aicore
