// 模型导出器，支持将训练好的模型导出为 ONNX 和 TensorRT 格式
// 便于模型在不同推理框架和平台间部署
#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

// 模型导出器类
// 将训练好的模型转换为标准中间格式（ONNX）和优化推理引擎（TensorRT）
class ModelExporter {
public:
    ModelExporter();

    // 将模型导出为 ONNX 格式
    // @param modelPath  原始模型权重文件路径
    // @param onnxPath   导出的 ONNX 文件保存路径
    // @return 成功返回 Success
    Status ExportToONNX(const std::string& modelPath, const std::string& onnxPath);

    // 将 ONNX 模型转换为 TensorRT 推理引擎
    // @param onnxPath     ONNX 模型文件路径
    // @param enginePath   导出的 TensorRT 引擎文件保存路径
    // @return 成功返回 Success
    Status ExportToTensorRT(const std::string& onnxPath, const std::string& enginePath);

    // 获取最近一次导出的错误信息
    // @return 错误描述字符串
    std::string GetLastError() const;

private:
    std::string lastError_;  // 最近一次错误的描述信息
};

} // namespace aicore
