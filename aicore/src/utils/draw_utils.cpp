#include "utils/draw_utils.h"

namespace aicore {

void DrawDetections(cv::Mat& image,
                    const std::vector<NodeResult>& detections,
                    const DrawOptions& opts) {
    for (const auto& det : detections) {
        const auto& b = det.bbox;
        int x1 = (int)(b.x - b.w / 2);
        int y1 = (int)(b.y - b.h / 2);
        int x2 = (int)(b.x + b.w / 2);
        int y2 = (int)(b.y + b.h / 2);

        cv::Scalar color = opts.defaultColor;
        auto scoreIt = det.measurements.find("anomaly_score");
        if (scoreIt != det.measurements.end() && opts.drawAnomalyScore) {
            auto anomalyIt = det.measurements.find("is_anomaly");
            bool isAnomaly = (anomalyIt != det.measurements.end() &&
                              anomalyIt->second > 0.5);
            color = isAnomaly ? opts.anomalyColor : opts.normalColor;
        }

        cv::rectangle(image, cv::Rect(x1, y1, x2 - x1, y2 - y1), color, opts.thickness);

        std::string label = det.label + ": " +
            std::to_string((int)(det.confidence * 100)) + "%";
        if (scoreIt != det.measurements.end() && opts.drawAnomalyScore) {
            label += " score:" + std::to_string((int)(scoreIt->second * 100)) + "%";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             opts.textScale, 1, &baseline);
        cv::Point labelPos(x1, y1 - 5);
        if (labelPos.y < textSize.height)
            labelPos.y = y2 + textSize.height + 5;

        cv::rectangle(image,
                      cv::Point(labelPos.x, labelPos.y - textSize.height),
                      cv::Point(labelPos.x + textSize.width, labelPos.y + baseline),
                      color, cv::FILLED);
        cv::putText(image, label, labelPos,
                    cv::FONT_HERSHEY_SIMPLEX, opts.textScale,
                    cv::Scalar(0, 0, 0), 1);
    }
}

void DrawRoiAnomaly(cv::Mat& image, const cv::Rect& rect,
                    const std::string& label, float score,
                    bool isAnomaly, const DrawOptions& opts) {
    cv::Scalar color = isAnomaly ? opts.anomalyColor : opts.normalColor;
    cv::rectangle(image, rect, color, opts.thickness);

    std::string text = label + ": " +
        (isAnomaly ? "NG " : "OK ") +
        std::to_string((int)(score * 100)) + "%";

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                         opts.textScale, 1, &baseline);
    cv::Point labelPos(rect.x, rect.y - 5);
    if (labelPos.y < textSize.height)
        labelPos.y = rect.y + rect.height + textSize.height + 5;

    cv::rectangle(image,
                  cv::Point(labelPos.x, labelPos.y - textSize.height),
                  cv::Point(labelPos.x + textSize.width, labelPos.y + baseline),
                  color, cv::FILLED);
    cv::putText(image, text, labelPos,
                cv::FONT_HERSHEY_SIMPLEX, opts.textScale,
                cv::Scalar(0, 0, 0), 1);
}

} // namespace aicore
