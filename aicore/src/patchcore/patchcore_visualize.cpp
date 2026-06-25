#include "patchcore/patchcore_visualize.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace aicore {

cv::Mat ColorizeAnomalyMap(const cv::Mat& anomalyMap, float maxVal) {
    CV_Assert(anomalyMap.type() == CV_32F);
    CV_Assert(anomalyMap.channels() == 1);

    double actualMax = maxVal;
    if (actualMax <= 0) {
        actualMax = 0.001;
        cv::minMaxLoc(anomalyMap, nullptr, &actualMax);
    }

    cv::Mat normalized;
    anomalyMap.convertTo(normalized, CV_32F, 1.0 / actualMax, 0);
    normalized = cv::max(normalized, 0);
    normalized = cv::min(normalized, 1.0f);

    cv::Mat normalized8u;
    normalized.convertTo(normalized8u, CV_8UC1, 255.0);

    cv::Mat colored;
    cv::applyColorMap(normalized8u, colored, cv::COLORMAP_JET);
    return colored;
}

cv::Mat DrawAnomalyOverlay(const cv::Mat& anomalyMap, const cv::Mat& image,
                           float alpha, float threshold) {
    CV_Assert(anomalyMap.size() == image.size());
    CV_Assert(anomalyMap.type() == CV_32F);
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat colored = ColorizeAnomalyMap(anomalyMap);
    colored.convertTo(colored, CV_32FC3, 1.0 / 255.0);

    cv::Mat fImage;
    image.convertTo(fImage, CV_32FC3, 1.0 / 255.0);

    cv::Mat overlay;
    if (threshold > 0) {
        cv::Mat mask;
        cv::threshold(anomalyMap, mask, threshold, 1.0, cv::THRESH_BINARY);
        cv::Mat mask3c;
        cv::cvtColor(mask, mask3c, cv::COLOR_GRAY2BGR);
        overlay = fImage.mul(cv::Scalar(1, 1, 1) - mask3c * alpha)
                + colored.mul(mask3c * alpha);
    } else {
        overlay = fImage * (1.0f - alpha) + colored * alpha;
    }

    overlay.convertTo(overlay, CV_8UC3, 255.0);
    return overlay;
}

cv::Mat MaskFromAnomalyMap(const cv::Mat& anomalyMap, float threshold) {
    CV_Assert(anomalyMap.type() == CV_32F);
    CV_Assert(anomalyMap.channels() == 1);

    cv::Mat mask;
    cv::threshold(anomalyMap, mask, threshold, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    return mask;
}

} // namespace aicore
