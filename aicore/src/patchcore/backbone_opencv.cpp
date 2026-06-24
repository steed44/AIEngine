// ============================================================
// backbone_opencv.cpp — OpenCV DNN Backbone 实现
// 功能：使用 OpenCV DNN 模块加载 ONNX 模型并提取中间层特征
// 优势：无需安装 PyTorch，纯 OpenCV 依赖，部署最简便
// ============================================================
#include "patchcore/backbone_opencv.h"
#include <opencv2/dnn.hpp>

namespace aicore {

// -------------------------------------------------------
// 初始化：从 ONNX 文件读取网络，配置输出层和输入尺寸
// 配置参数：
//   - model_path:      ONNX 模型文件路径（必需）
//   - backbone_layers: 待提取的中间层名称（逗号分隔，可选）
//   - input_size:      输入缩放尺寸（可选，默认 224）
// -------------------------------------------------------
Status OpenCVDnnBackbone::Init(const NodeConfig& config) {
    auto it = config.find("model_path");
    if (it == config.end()) {
        return Status{StatusCode::ErrorConfigParse, "opencv_dnn: model_path required"};
    }
    net_ = cv::dnn::readNetFromONNX(it->second);

    auto layers = config.find("backbone_layers");
    if (layers != config.end()) {
        outputLayerNames_ = SplitLayerNames(layers->second);
    }

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    return Status{};
}

// -------------------------------------------------------
// 提取特征：blobFromImage 自动处理 resize/减均值/缩放/RGB 置换，
// 前向推理获取指定中间层输出，遍历每个空间位置生成 PatchFeature
// -------------------------------------------------------
std::vector<PatchFeature> OpenCVDnnBackbone::Extract(const cv::Mat& image) {
    // blobFromImage 完成：resize → 1/255 → BGR→RGB → 减均值
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255,
        cv::Size(inputSize_, inputSize_),
        cv::Scalar(0.485, 0.456, 0.406), true, false);

    // ImageNet std 除法: 每通道除以对应 std 值
    // blob 布局 NCHW, N=1, C=3
    float* data = blob.ptr<float>();
    float stdVals[3] = {0.229f, 0.224f, 0.225f};
    int channels = blob.size[1];
    int hw = blob.size[2] * blob.size[3];
    for (int c = 0; c < channels; c++) {
        float* channelData = data + c * hw;
        float s = stdVals[c];
        for (int i = 0; i < hw; i++) {
            channelData[i] /= s;
        }
    }

    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, outputLayerNames_);

    // 遍历各层输出，逐位置提取特征
    std::vector<PatchFeature> features;
    for (int li = 0; li < static_cast<int>(outputs.size()); li++) {
        auto& feat = outputs[li];
        int channels = feat.size[1];
        int h = feat.size[2];
        int w = feat.size[3];
        float* data = feat.ptr<float>();

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.layerIdx = li;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                for (int c = 0; c < channels; c++) {
                    pf.features[c] = data[(c * h + row) * w + col];
                }
                features.push_back(pf);
            }
        }
    }
    return features;
}

} // namespace aicore
