// ============================================================
// backbone_libtorch.h — 基于 LibTorch (C++ PyTorch) 的 Backbone
// 功能：使用 PyTorch JIT 加载 TorchScript 模型并提取中间层特征
// 注意：需要编译时定义 AICORE_HAS_LIBTORCH 宏，且链接 libtorch 库
// ============================================================
#pragma once
#include "patchcore/backbone.h"

#ifdef AICORE_HAS_LIBTORCH
#include <torch/script.h>
#endif

namespace aicore {

// -------------------------------------------------------
// LibTorchBackbone — LibTorch 特征提取器
// 职责：加载 TorchScript 格式的模型，执行前向推理，
//       从指定中间层提取特征并转换为 PatchFeature 列表
// 典型使用场景：已有 PyTorch 训练的模型导出为 TorchScript，
//       需要直接使用 LibTorch 运行时进行 C++ 部署
// -------------------------------------------------------
class LibTorchBackbone : public IBackbone {
public:
    // 初始化：加载 TorchScript 模型，设置输出层名称和输入尺寸
    Status Init(const NodeConfig& config) override;
    // 提取特征：BGR→RGB 归一化 → Tensor 转换 → 模型推理 → 层输出转 PatchFeature
    std::vector<PatchFeature> Extract(const cv::Mat& image) override;
    std::string GetType() const override { return "libtorch"; }

private:
#ifdef AICORE_HAS_LIBTORCH
    torch::jit::Module module_;       // TorchScript JIT 模型模块
#endif
    std::vector<std::string> outputLayerNames_;  // 需要提取的中间层名列表
    int inputSize_ = 224;                        // 输入图像缩放宽高
};

} // namespace aicore
