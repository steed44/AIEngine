// ============================================================
// backbone_model.h — 基于 IModelBackend 通用后端的 Backbone 实现
// 功能：通过统一的模型后端接口（ONNX Runtime / TensorRT / LibTorch）
//       加载并运行特征提取网络，不依赖特定推理框架的 API
// ============================================================
#pragma once
#include "patchcore/backbone.h"
#include "core/model_backend.h"

namespace aicore {

// -------------------------------------------------------
// ModelBackendBackbone — 通用模型后端特征提取器
// 职责：通过 IModelBackend 抽象层调用底层推理框架，
//       从模型中提取中间层特征，转换为 PatchFeature 列表
// 典型使用场景：需要跨推理框架（ONNX / TRT / LibTorch）的
//       统一部署场景，使用 BackendFactory 创建具体后端
// -------------------------------------------------------
class ModelBackendBackbone : public IBackbone {
public:
    // 初始化：配置 backend_type、model_path，创建后端实例
    Status Init(const NodeConfig& config) override;
    // 提取特征：HWC→CHW 转换 → 后端推理 → 输出转 PatchFeature
    std::vector<PatchFeature> Extract(const cv::Mat& image) override;
    std::string GetType() const override { return "model_backend"; }

private:
    std::unique_ptr<IModelBackend> backend_;  // 底层推理后端实例
    int inputSize_ = 224;                     // 输入图像缩放宽高
};

} // namespace aicore
