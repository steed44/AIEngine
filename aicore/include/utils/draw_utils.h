#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

// ============================================================
// draw_utils.h — 检测结果绘制接口
// 功能：提供 YOLO 检测框和 PatchCore ROI 异常的绘制函数
// ============================================================

namespace aicore {

// 绘制选项结构体
// 控制检测框、标签文本、异常颜色等的显示样式
struct DrawOptions {
    float textScale = 0.5f;            // 标签文本字号
    int thickness = 2;                 // 边框线宽（像素）
    bool drawAnomalyScore = true;      // 是否显示异常得分
    cv::Scalar normalColor = cv::Scalar(0, 255, 0);    // 正常样本框颜色（绿）
    cv::Scalar anomalyColor = cv::Scalar(0, 0, 255);   // 异常样本框颜色（红）
    cv::Scalar defaultColor = cv::Scalar(255, 255, 0);  // 默认框颜色（青）
};

// 绘制检测结果（边界框 + 标签 + 置信度）
void DrawDetections(cv::Mat& image,
                    const std::vector<NodeResult>& detections,
                    const DrawOptions& options = {});

// 绘制 ROI 异常判定（边框 + OK/NG + 得分）
void DrawRoiAnomaly(cv::Mat& image, const cv::Rect& rect,
                    const std::string& label, float score,
                    bool isAnomaly, const DrawOptions& options = {});

} // namespace aicore
