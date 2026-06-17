// ============================================================
// 文件：onnx_exporter.cpp
// 用途：实现 ONNX 模型导出器，通过嵌入 Python 解释器
//   调用 PyTorch 的 torch.onnx.export() 完成模型转换。
// ============================================================
#include "optimizer/onnx_exporter.h"

namespace aicore {

OnnxExporter::OnnxExporter() {}

// 导出 ONNX 模型：
// 1. 将导出参数（模型路径、ONNX 路径、输入形状、opset）拼接为 JSON
// 2. 通过 Python 嵌入调用 scripts/export_onnx.py 脚本执行导出
// 3. 导出脚本使用 torch.onnx.export() 完成实际转换
Status OnnxExporter::Export(const std::string& modelPath,
                            const std::string& onnxPath,
                            const std::vector<int64_t>& inputShape,
                            int opset) {
    // 构建配置 JSON 字符串，传递给 Python 导出脚本
    std::string cfg = "{\"model_path\":\"" + modelPath +
                      "\",\"onnx_path\":\"" + onnxPath +
                      "\",\"input_shape\":" + std::to_string(inputShape.size()) +
                      ",\"opset\":" + std::to_string(opset) + "}";
    std::string output;
    // 调用嵌入的 Python 运行时执行导出
    auto s = py_.RunScript("scripts/export_onnx.py", cfg, output);
    if (!s) return s;
    return Status{};
}

// 简化 ONNX 模型：桩实现
// 正式实现应使用 onnx-simplifier 或自定义优化 pass
// 执行常量折叠、冗余节点消除、算子融合等操作
Status OnnxExporter::Simplify(const std::string& onnxPath) {
    // onnxPath 参数暂未使用
    (void)onnxPath;
    return Status{};
}

std::string OnnxExporter::GetLastError() const { return lastError_; }

} // namespace aicore
