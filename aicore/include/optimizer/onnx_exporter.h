#pragma once
#include "core/types.h"
#include "optimizer/python_embedding.h"
#include <string>

namespace aicore {

class AICORE_OPTIMIZER_API OnnxExporter {
public:
    OnnxExporter();
    Status Export(const std::string& modelPath, const std::string& onnxPath,
                  const std::vector<int64_t>& inputShape, int opset = 17);
    Status Simplify(const std::string& onnxPath);
    std::string GetLastError() const;

private:
    PythonEmbedding py_;
    std::string lastError_;
};

} // namespace aicore
