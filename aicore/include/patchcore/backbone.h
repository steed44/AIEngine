// ============================================================
// backbone.h — PatchCore 特征提取 Backbone 抽象接口
// 功能：定义统一的特征提取器接口，支持多种后端实现
// 使用工厂方法 CreateBackbone 根据类型字符串创建具体实现
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include "core/processor.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

namespace aicore {

// -------------------------------------------------------
// IBackbone — 特征提取器抽象基类
// 职责：将输入图像转换为 PatchFeature 向量列表，供后续
//       MemoryBank 比对使用。支持 OpenCV DNN / LibTorch /
//       ModelBackend 等多种实现
//
// 设计模式：策略模式（Strategy）+ 工厂方法
//   IBackbone 定义特征提取的统一策略接口，
//   CreateBackbone(type, config) 工厂根据类型字符串创建具体策略实现。
//   三种实现：OpenCVDnnBackbone（CPU，零依赖）、
//   LibTorchBackbone（GPU，需 LibTorch SDK）、
//   ModelBackendBackbone（通用，通过 IModelBackend 接入任意推理框架）
//
// 生命周期：
//   1. 构造 → 2. Init(config) → 3. Extract(image) 可重复调用 → 4. 析构
//   多 ROI 场景下，多个 RoiModelSlot 共享同一 IBackbone 实例
// -------------------------------------------------------
class IBackbone {
public:
    virtual ~IBackbone() = default;
    // 初始化 backbone：加载模型文件、配置输出层和输入尺寸
    // @param config 包含 model_path、backbone_layers、input_size 等
    // 前置条件：config 包含 "model_path" 键
    // 后置条件：成功时 backbone 就绪，Extract() 可被调用
    virtual Status Init(const NodeConfig& config) = 0;
    // 提取图像特征
    // @param image  输入图像（BGR 格式，CV_8UC3）
    // @return 从中间层输出的 PatchFeature 列表，每个特征对应一个局部块
    // 前置条件：Init() 已成功调用，image 非空
    // 后置条件：返回的 PatchFeature 列表按 layerIdx 顺序排列
    virtual std::vector<PatchFeature> Extract(const cv::Mat& image) = 0;
    // 返回 backbone 类型标识（"opencv_dnn" / "libtorch" / "model_backend"）
    // 线程安全：const 方法，线程安全
    virtual std::string GetType() const = 0;
};

// 根据类型字符串创建对应的 Backbone 实例
// @param type   backbone 类型：opencv_dnn / model_backend / libtorch
// @param config 配置参数（透传给具体实现的 Init）
// @return 创建的 IBackbone 实例，若类型未知则返回 nullptr
std::unique_ptr<IBackbone> CreateBackbone(const std::string& type, const NodeConfig& config);

// 工具函数: 将逗号分隔的层名字符串解析为 vector
// 如 "layer2,layer3" → {"layer2", "layer3"}
std::vector<std::string> SplitLayerNames(const std::string& s);

} // namespace aicore
