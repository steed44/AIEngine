#include "optimizer/onnx_exporter.h"

namespace aicore {

OnnxExporter::OnnxExporter() {}

Status OnnxExporter::Export(const std::string& modelPath,
                            const std::string& onnxPath,
                            const std::vector<int64_t>& inputShape,
                            int opset) {
    std::string cfg = "{\"model_path\":\"" + modelPath +
                      "\",\"onnx_path\":\"" + onnxPath +
                      "\",\"input_shape\":" + std::to_string(inputShape.size()) +
                      ",\"opset\":" + std::to_string(opset) + "}";
    std::string output;
    auto s = py_.RunScript("scripts/export_onnx.py", cfg, output);
    if (!s) return s;
    return Status{};
}

Status OnnxExporter::Simplify(const std::string& onnxPath) {
    (void)onnxPath;
    return Status{};
}

std::string OnnxExporter::GetLastError() const { return lastError_; }

} // namespace aicore
