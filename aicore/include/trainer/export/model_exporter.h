#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

class ModelExporter {
public:
    ModelExporter();
    Status ExportToONNX(const std::string& modelPath, const std::string& onnxPath);
    Status ExportToTensorRT(const std::string& onnxPath, const std::string& enginePath);
    std::string GetLastError() const;

private:
    std::string lastError_;
};

} // namespace aicore
