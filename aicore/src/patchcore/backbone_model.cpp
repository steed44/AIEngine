// ============================================================
// backbone_model.cpp — 基于 IModelBackend 通用后端的 Backbone 实现
// 功能：通过 BackendFactory 创建 ONNX Runtime / TensorRT / LibTorch
//       等后端，屏蔽底层推理框架差异，以统一方式提取特征
// ============================================================
#include "patchcore/backbone_model.h"
#include "backend/backend_factory.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// -------------------------------------------------------
// 初始化：根据 config 获取 backend_type（默认 ONNX Runtime），
// 通过 BackendFactory 创建后端实例，加载模型文件
// 配置参数：
//   - model_path:   模型文件路径（必需）
//   - backend_type: 后端类型（onnxruntime / tensorrt / libtorch，可选）
//   - input_size:   输入缩放尺寸（可选，默认 224）
// -------------------------------------------------------
Status ModelBackendBackbone::Init(const NodeConfig& config) {
    auto it = config.find("model_path");
    if (it == config.end()) {
        return Status{StatusCode::ErrorConfigParse, "model_backend: model_path required"};
    }

    auto bk = config.find("backend_type");
    BackendType backendType = BackendType::kONNXRuntime;
    if (bk != config.end()) {
        if (bk->second == "tensorrt") backendType = BackendType::kTensorRT;
        else if (bk->second == "libtorch") backendType = BackendType::kLibTorch;
    }

    // 使用工厂方法创建后端实例
    backend_ = BackendFactory::Create(backendType);
    if (!backend_) {
        return Status{StatusCode::ErrorConfigParse, "model_backend: unknown backend_type"};
    }

    ModelInfo info;
    info.modelPath = it->second;
    auto s = backend_->Load(info);
    if (!s) return s;

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    return Status{};
}

// -------------------------------------------------------
// 提取特征：HWC → CHW 数据重排 → 后端推理 → PatchFeature 列表
// 注意：OpenCV 默认 HWC 布局，需手动转换为 NCHW 布局再送入后端
// -------------------------------------------------------
std::vector<PatchFeature> ModelBackendBackbone::Extract(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputSize_, inputSize_));
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0 / 255);

    // ---- HWC → CHW 数据重排 + ImageNet 标准化 ----
    Tensor input;
    input.dtype = DataType::kFloat32;
    input.shape = {1, 3, inputSize_, inputSize_};
    input.bytes = 1 * 3 * inputSize_ * inputSize_ * sizeof(float);
    std::vector<float> chw(input.bytes / sizeof(float));
    float* src = floatImg.ptr<float>();
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < inputSize_; h++) {
            for (int w = 0; w < inputSize_; w++) {
                float val = src[h * inputSize_ * 3 + w * 3 + c];
                chw[c * inputSize_ * inputSize_ + h * inputSize_ + w] = (val - mean[c]) / std[c];
            }
        }
    }
    input.data = chw.data();

    // ---- 后端推理 ----
    std::vector<Tensor> outputs;
    auto s = backend_->Infer({input}, outputs);
    if (!s) return {};

    // ---- 提取各输出层的 Patch 特征 ----
    std::vector<PatchFeature> features;
    for (auto& out : outputs) {
        int channels = static_cast<int>(out.shape[1]);
        int h = static_cast<int>(out.shape[2]);
        int w = static_cast<int>(out.shape[3]);
        float* data = static_cast<float*>(out.data);
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                for (int c = 0; c < channels; c++)
                    pf.features[c] = data[(c * h + row) * w + col];
                features.push_back(pf);
            }
        }
    }
    return features;
}

} // namespace aicore
