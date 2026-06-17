// ============================================================
// 文件：tensorrt_builder.cpp
// 用途：实现 TensorRT 引擎构建器，当前为桩实现，
//   正式使用时需链接 TensorRT SDK 并调用 nvinfer1::builder
//   完成 ONNX -> engine 的转换。
// ============================================================
#include "optimizer/tensorrt_builder.h"

namespace aicore {

TensorRtBuilder::TensorRtBuilder() {}
TensorRtBuilder::~TensorRtBuilder() {}

// 构建 TensorRT 引擎：桩实现
// 正式实现流程：
//   1. 创建 nvinfer1::IBuilder 实例
//   2. 通过 ONNX 解析器加载模型
//   3. 配置精度、工作空间、动态批次等参数
//   4. 构建序列化引擎保存至 config.enginePath
Status TensorRtBuilder::Build(const BuildConfig& config) {
    // config 参数暂未使用，后续集成 TensorRT SDK
    (void)config;
    return Status{StatusCode::ErrorInternal, "TensorRT build not available in stub"};
}

std::string TensorRtBuilder::GetLastError() const { return lastError_; }

} // namespace aicore
