// YOLOv8 模型定义 — 纯 C++ LibTorch 实现
// 包含 YOLOv8 网络架构的所有构建块：ConvBnSiLU, C2f, SPPF, Detect
// 支持 TorchScript 权重加载
#pragma once
#include "core/types.h"
#include "trainer/model/model_factory.h"
#include <torch/nn.h>
#include <torch/serialize.h>
#include <memory>
#include <vector>

namespace aicore {

// ─── 基础模块 ─────────────────────────────────────────────

// Conv + BatchNorm + SiLU 标准卷积块
// YOLOv8 基础构建单元，替换传统 Conv2d + 激活函数
struct ConvBnSiLUImpl : torch::nn::Module {
    ConvBnSiLUImpl(int inCh, int outCh, int k = 3, int s = 2, int p = 1);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
};
TORCH_MODULE(ConvBnSiLU);

// C2f — Cross Stage Partial with 2 convolutions
// YOLOv8 的核心特征提取模块，相比 C3 更轻量
struct C2fImpl : torch::nn::Module {
    C2fImpl(int inCh, int outCh, int n = 1, bool shortcut = true,
            float e = 0.5);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d cv1{ nullptr };
    torch::nn::Conv2d cv2{ nullptr };
    torch::nn::ModuleList m; // Bottleneck list
    bool add = true;
    int c = 0;
};
TORCH_MODULE(C2f);

// Bottleneck (C2f 内部使用) — ConvBnSiLU × 2 + shortcut
struct BottleneckImpl : torch::nn::Module {
    BottleneckImpl(int inCh, int outCh, bool shortcut = true);
    torch::Tensor forward(torch::Tensor x);

    ConvBnSiLU cv1{ nullptr };
    ConvBnSiLU cv2{ nullptr };
    bool add = true;
};
TORCH_MODULE(Bottleneck);

// SPPF — Spatial Pyramid Pooling Fast
// 通过多尺度最大池化融合上下文信息，相比 SPP 更快
struct SPPFImpl : torch::nn::Module {
    SPPFImpl(int inCh, int outCh, int k = 5);
    torch::Tensor forward(torch::Tensor x);

    ConvBnSiLU cv1{ nullptr };
    ConvBnSiLU cv2{ nullptr };
    torch::nn::MaxPool2d m1{ nullptr }, m2{ nullptr }, m3{ nullptr };
    int c = 0;
};
TORCH_MODULE(SPPF);

// Detect — YOLOv8 解耦检测头 (标准架构: 4*regMax reg + nc cls)
struct DetectImpl : torch::nn::Module {
    DetectImpl(int nc, const std::vector<int>& ch);
    torch::Tensor forward(const std::vector<torch::Tensor>& xs);

    int nc = 0;
    int no = 0;
    int nl = 0;
    std::vector<torch::nn::Conv2d> cvCls;
    std::vector<torch::nn::Conv2d> cvReg;
    torch::Tensor proj;
};
TORCH_MODULE(Detect);

// ─── YOLOv8 完整网络 ──────────────────────────────────────

// YOLOv8 模型，继承 IModel（项目接口）+ torch::nn::Module（LibTorch）
// 支持 TorchScript trace 权重加载
class YOLOv8Model : public IModel, public torch::nn::Module {
public:
    // 注册网络模块
    YOLOv8Model(int nc = 80);

    // IModel 接口
    Status Build(const ModelConfig& config) override;
    Status Save(const std::string& path) override;
    Status Load(const std::string& path) override;
    std::string GetArchName() const override { return "yolov8"; }

    // 前向传播：返回三个尺度的原始预测（训练用）
    std::vector<torch::Tensor> Forward(torch::Tensor x);

    // 推理入口：输入 [N,3,H,W] → 原始检测输出
    torch::Tensor predict(torch::Tensor x);

    // 直接访问底层 Sequential 模块（用于加载 TorchScript trace）
    torch::nn::Sequential& model() { return model_; }

private:
    torch::nn::Sequential model_{ nullptr };
    Detect detect_{ nullptr };
    int nc_ = 80;
};

} // namespace aicore
