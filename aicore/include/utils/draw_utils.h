#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace aicore {

struct DrawOptions {
    float textScale = 0.5f;
    int thickness = 2;
    bool drawAnomalyScore = true;
    cv::Scalar normalColor = cv::Scalar(0, 255, 0);
    cv::Scalar anomalyColor = cv::Scalar(0, 0, 255);
    cv::Scalar defaultColor = cv::Scalar(255, 255, 0);
};

void DrawDetections(cv::Mat& image,
                    const std::vector<NodeResult>& detections,
                    const DrawOptions& options = {});

void DrawRoiAnomaly(cv::Mat& image, const cv::Rect& rect,
                    const std::string& label, float score,
                    bool isAnomaly, const DrawOptions& options = {});

} // namespace aicore
