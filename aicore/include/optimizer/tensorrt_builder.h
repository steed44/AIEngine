#pragma once
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

enum class Precision { kFP32, kFP16, kINT8 };

struct BuildConfig {
    std::string onnxPath;
    std::string enginePath;
    Precision precision = Precision::kFP16;
    int maxBatchSize = 1;
    int deviceId = 0;
    size_t workspaceSize = 1ULL << 30;
    bool dynamicBatch = false;
    std::string calibDir;
    int numCalib = 100;
};

class AICORE_OPTIMIZER_API TensorRtBuilder {
public:
    TensorRtBuilder();
    ~TensorRtBuilder();

    Status Build(const BuildConfig& config);
    std::string GetLastError() const;

private:
    std::string lastError_;
};

} // namespace aicore
