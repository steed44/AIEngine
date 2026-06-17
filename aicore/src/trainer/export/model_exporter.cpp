// ============================================================================
// 文件：model_exporter.cpp
// 用途：模型导出器实现，将训练好的模型转换为部署格式
// 功能：导出为 ONNX、TensorRT 引擎等推理优化格式
// ============================================================================

#include "trainer/export/model_exporter.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
ModelExporter::ModelExporter() {}

// 将训练好的模型导出为 ONNX 格式
// 参数 modelPath - 原始模型文件路径（如 .pt 或 .pth）
// 参数 onnxPath  - 输出 ONNX 文件路径
// 返回值        - 操作状态
// 注意：当前为桩实现，真实场景需要 PythonEmbedding 加载模型并导出
Status ModelExporter::ExportToONNX(const std::string& modelPath,
                                    const std::string& onnxPath) {
    (void)modelPath; (void)onnxPath;
    return Status{};
}

// 将 ONNX 模型进一步转换为 TensorRT 引擎（优化推理速度）
// 参数 onnxPath   - 输入 ONNX 文件路径
// 参数 enginePath - 输出 TensorRT 引擎文件路径
// 返回值         - 操作状态
// 注意：当前为桩实现，真实场景需要 TensorRtBuilder 进行图优化和序列化
Status ModelExporter::ExportToTensorRT(const std::string& onnxPath,
                                        const std::string& enginePath) {
    (void)onnxPath; (void)enginePath;
    return Status{};
}

// 获取最近一次导出操作的错误描述
// 返回值 - 错误信息字符串
std::string ModelExporter::GetLastError() const { return lastError_; }

} // namespace aicore
