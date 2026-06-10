#include "trainer/export/model_exporter.h"

namespace aicore {

ModelExporter::ModelExporter() {}

Status ModelExporter::ExportToONNX(const std::string& modelPath,
                                    const std::string& onnxPath) {
    // Stub: always succeed (real impl requires PythonEmbedding)
    (void)modelPath; (void)onnxPath;
    return Status{};
}

Status ModelExporter::ExportToTensorRT(const std::string& onnxPath,
                                        const std::string& enginePath) {
    // Stub: always succeed (real impl requires TensorRtBuilder)
    (void)onnxPath; (void)enginePath;
    return Status{};
}

std::string ModelExporter::GetLastError() const { return lastError_; }

} // namespace aicore
