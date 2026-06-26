#include "utils/draw_utils.h"

// ============================================================
// draw_utils.cpp — 检测结果绘制工具
// 功能：在图像上绘制 YOLO 检测框、PatchCore ROI 异常覆盖、
//       标签、得分等可视化信息。
//
// 绘制内容：
//   DrawDetections — 绘制目标检测结果，支持异常颜色编码
//   DrawRoiAnomaly — 绘制 ROI 异常判定结果
// ============================================================

namespace aicore {

// 绘制检测结果：边界框 + 标签 + 置信度 + 异常得分
// 对每个检测结果：
//   1. 根据异常得分选择颜色（异常=红，正常=绿，默认=青）
//   2. 绘制矩形边界框
//   3. 绘制标签背景矩形 + 文本
//   4. 文本包含：类别名、置信度百分比、异常得分
//
// 标签定位自适应：默认在框上方，若框太靠上则移到底部显示
// @param image      目标图像（直接修改）
// @param detections 检测结果列表
// @param opts       绘制选项（颜色、字体大小、粗细等）
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

// 绘制单 ROI 异常判定结果
// 用途：多 ROI 推理模式下，为每个 ROI 区域绘制判定边框和标签
// 显示格式："{label}: OK/NG {score}%"
//   - OK = 正常（绿色边框）
//   - NG = 异常（红色边框）
// @param image     目标图像
// @param rect      ROI 区域坐标
// @param label     ROI 名称标签
// @param score     异常得分
// @param isAnomaly 是否为异常
// @param opts      绘制选项
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
