// ============================================================
// 文件：onnx_exporter.h
// 用途：定义 ONNX 模型导出器，将训练好的模型转换为
//   ONNX 格式并进行简化优化。
// ============================================================
#pragma once
#include "core/types.h"
#include "optimizer/python_embedding.h"
#include <string>

namespace aicore {

// OnnxExporter：ONNX 模型导出器
// 职责：通过嵌入的 Python 运行时调用 PyTorch 导出脚本，
//   将原始模型转换为 ONNX 格式，并执行 ONNX 简化（常量折叠、冗余消除等）。
// 典型使用场景：训练完成后将 .pt / .pth 模型导出为 .onnx 用于跨平台部署。
class AICORE_OPTIMIZER_API OnnxExporter {
public:
    OnnxExporter();

    // 将原始模型导出为 ONNX 格式
    // 参数 modelPath  : 原始模型文件路径（.pt / .pth）
    // 参数 onnxPath   : 输出的 ONNX 文件路径
    // 参数 inputShape : 模型输入张量的形状（如 {1,3,640,640}）
    // 参数 opset      : ONNX opset 版本（默认 17）
    Status Export(const std::string& modelPath, const std::string& onnxPath,
                  const std::vector<int64_t>& inputShape, int opset = 17);

    // 简化 ONNX 模型（常量折叠、节点消除等）
    // 参数 onnxPath : ONNX 模型文件路径
    Status Simplify(const std::string& onnxPath);

    // 获取最后一条错误信息
    std::string GetLastError() const;

private:
    PythonEmbedding py_;   // Python 运行时嵌入实例，用于调用导出脚本
    std::string lastError_; // 最近一次操作的错误信息
};

} // namespace aicore
