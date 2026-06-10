#pragma once
#include "core/types.h"
#include "optimizer/tensorrt_builder.h"
#include "optimizer/onnx_exporter.h"
#include <string>

namespace aicore {

class AICORE_OPTIMIZER_API ModelOptimizer {
public:
    ModelOptimizer();
    Status Optimize(const std::string& configJson);
    std::string GetLastError() const;

private:
    OnnxExporter exporter_;
    TensorRtBuilder trtBuilder_;
    std::string lastError_;
};

} // namespace aicore
