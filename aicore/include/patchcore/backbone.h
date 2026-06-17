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
// -------------------------------------------------------
class IBackbone {
public:
    virtual ~IBackbone() = default;
    // 初始化 backbone：加载模型文件、配置输出层和输入尺寸
    // @param config 包含 model_path、backbone_layers、input_size 等
    virtual Status Init(const NodeConfig& config) = 0;
    // 提取图像特征
    // @param image  输入图像（BGR 格式）
    // @return 从中间层输出的 PatchFeature 列表，每个特征对应一个局部块
    virtual std::vector<PatchFeature> Extract(const cv::Mat& image) = 0;
    // 返回 backbone 类型标识（"opencv_dnn" / "libtorch" / "model_backend"）
    virtual std::string GetType() const = 0;
};

// 根据类型字符串创建对应的 Backbone 实例
// @param type   backbone 类型：opencv_dnn / model_backend / libtorch
// @param config 配置参数（透传给具体实现的 Init）
// @return 创建的 IBackbone 实例，若类型未知则返回 nullptr
std::unique_ptr<IBackbone> CreateBackbone(const std::string& type, const NodeConfig& config);

} // namespace aicore
