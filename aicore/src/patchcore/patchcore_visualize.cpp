#include "patchcore/patchcore_visualize.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

// ============================================================
// patchcore_visualize.cpp — PatchCore 异常热力图可视化
// 功能：将 anomaly score 映射到颜色空间，生成直观的热力图
//       叠加到原图上。
//
// 可视化流程：
//   1. 异常得分图（CV_32F）归一化到 [0, 1]
//   2. 映射到 COLORMAP_JET 颜色表（蓝 → 青 → 黄 → 红）
//   3. 与原图按 alpha 比例融合（阈值区域的局部融合）
//   4. 输出 CV_8UC3 彩色覆盖图
//
// 颜色映射理解：
//   0.0 (正常)   → 蓝色/透明
//   0.5 (可疑)   → 黄绿色
//   1.0 (异常)   → 红色
// ============================================================

namespace aicore {

// 异常得分 → 伪彩色映射
// 步骤：
//   1. 找最大异常值（若未指定 maxVal），用实际最大值归一化
//   2. 得分裁剪到 [0, 1]
//   3. CV_32F → CV_8UC1（0-255）
//   4. OpenCV COLORMAP_JET 伪彩色编码
// @param anomalyMap  CV_32F 单通道异常得分图
// @param maxVal      归一化最大值，0 则自动取实际最大值
// @return CV_8UC3 彩色热力图
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

// 异常热力图叠加到原图
// 支持两种融合模式：
//   1. threshold > 0：仅异常分数超过阈值的区域融合热力图，
//      正常区域保持原图。产生「只在缺陷处着色」的效果。
//   2. threshold = 0：整图融合，产生「半透明热力图叠加」效果。
//
// 融合公式（threshold > 0）：
//   overlay = img × (1 - mask × alpha) + heatmap × (mask × alpha)
//   即：mask=1 区域用 alpha 比例混合热力图和原图
//       mask=0 区域保持原图
//
// @param anomalyMap  CV_32F 单通道异常得分图（已上采样到 image 尺寸）
// @param image       CV_8UC3 原图
// @param alpha       融合透明度 [0, 1]，越大热力图越明显
// @param threshold   异常阈值，>0 时仅异常区域显示热力图
// @return CV_8UC3 叠加结果
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
        // 阈值模式：只在异常区域叠加热力图
        cv::Mat mask;
        cv::threshold(anomalyMap, mask, threshold, 1.0, cv::THRESH_BINARY);
        cv::Mat mask3c;
        cv::cvtColor(mask, mask3c, cv::COLOR_GRAY2BGR);
        overlay = fImage.mul(cv::Scalar(1, 1, 1) - mask3c * alpha)
                + colored.mul(mask3c * alpha);
    } else {
        // 全图融合模式：整图按 alpha 比例混合
        overlay = fImage * (1.0f - alpha) + colored * alpha;
    }

    overlay.convertTo(overlay, CV_8UC3, 255.0);
    return overlay;
}

// 从异常得分图生成二值掩膜
// 用途：后处理中提取异常区域轮廓、计算异常面积比例等
// @param anomalyMap  CV_32F 单通道异常得分图
// @param threshold   二值化阈值，>threshold 视为异常像素
// @return CV_8UC1 二值掩膜（0 或 255）
cv::Mat MaskFromAnomalyMap(const cv::Mat& anomalyMap, float threshold) {
    CV_Assert(anomalyMap.type() == CV_32F);
    CV_Assert(anomalyMap.channels() == 1);

    cv::Mat mask;
    cv::threshold(anomalyMap, mask, threshold, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    return mask;
}

} // namespace aicore
