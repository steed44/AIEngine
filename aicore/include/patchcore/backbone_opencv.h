// ============================================================
// backbone_opencv.h — 基于 OpenCV DNN 模块的 Backbone 实现
// 功能：使用 OpenCV 的 DNN 模块加载 ONNX 模型并提取中间层特征
//       无需额外依赖，部署最简便，适合快速原型验证
// ============================================================
#pragma once
#include "patchcore/backbone.h"
#include <opencv2/dnn.hpp>

namespace aicore {

// -------------------------------------------------------
// OpenCVDnnBackbone — OpenCV DNN 特征提取器
// 职责：通过 cv::dnn::readNetFromONNX 加载 ONNX 模型，
//       执行前向推理并从指定中间层提取特征
// 典型使用场景：需要最小化第三方依赖，或快速验证 PatchCore
//       算法的检测效果时的首选 backbone 实现
// -------------------------------------------------------
class OpenCVDnnBackbone : public IBackbone {
public:
    // 初始化：读取 ONNX 模型文件，配置输出层名称和输入尺寸
    Status Init(const NodeConfig& config) override;
    // 提取特征：blobFromImage 预处理 → DNN 前向 → 各层输出转 PatchFeature
    std::vector<PatchFeature> Extract(const cv::Mat& image) override;
    std::string GetType() const override { return "opencv_dnn"; }

private:
    cv::dnn::Net net_;                          // OpenCV DNN 网络对象
    std::vector<std::string> outputLayerNames_;  // 待提取的中间层名称列表
    int inputSize_ = 224;                        // 输入图像缩放宽高
};

} // namespace aicore
