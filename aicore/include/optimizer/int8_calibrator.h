#pragma once
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

class AICORE_OPTIMIZER_API Int8Calibrator {
public:
    Int8Calibrator();
    Status LoadCalibrationData(const std::string& calibDir, int numSamples);
    float* GetBatch();
    int GetNumSamples() const { return numSamples_; }
    std::string GetLastError() const;

private:
    std::vector<std::vector<float>> samples_;
    int numSamples_ = 0;
    int currentIndex_ = 0;
    std::string lastError_;
};

} // namespace aicore
