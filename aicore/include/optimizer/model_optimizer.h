// ============================================================
// 文件：model_optimizer.h
// 用途：定义模型优化器主类 ModelOptimizer，编排 ONNX 导出
//   与 TensorRT 引擎构建的完整优化流程。
// ============================================================
#pragma once
#include "core/types.h"
#include "optimizer/tensorrt_builder.h"
#include "optimizer/onnx_exporter.h"
#include <string>

namespace aicore {

// ModelOptimizer：模型优化的核心编排类
// 职责：解析 JSON 配置，依次调用 OnnxExporter 导出 ONNX，
//   然后调用 TensorRtBuilder 构建 TensorRT 引擎。
// 典型使用场景：用户提供模型路径和优化参数，一键完成优化。
class AICORE_OPTIMIZER_API ModelOptimizer {
public:
    ModelOptimizer();

    // 执行模型优化流程
    // 参数 configJson : JSON 配置字符串，支持字段：
    //   model_path  - 原始模型路径（必填）
    //   onnx_path   - 导出的 ONNX 保存路径（可选）
    //   engine_path - TensorRT 引擎保存路径（可选）
    //   precision   - 精度模式：fp16 / int8 / fp32（可选，默认 fp16）
    // 返回值        : Status 状态码
    Status Optimize(const std::string& configJson);

    // 获取最后一条错误信息
    std::string GetLastError() const;

private:
    OnnxExporter exporter_;       // ONNX 导出器实例
    TensorRtBuilder trtBuilder_;  // TensorRT 引擎构建器实例
    std::string lastError_;       // 最近一次操作的错误信息
};

} // namespace aicore
