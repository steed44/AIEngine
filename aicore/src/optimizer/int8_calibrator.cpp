#include "optimizer/int8_calibrator.h"

namespace aicore {

Int8Calibrator::Int8Calibrator() {}

Status Int8Calibrator::LoadCalibrationData(const std::string& calibDir,
                                            int numSamples) {
    (void)calibDir;
    numSamples_ = numSamples;
    return Status{};
}

float* Int8Calibrator::GetBatch() {
    return nullptr;
}

std::string Int8Calibrator::GetLastError() const { return lastError_; }

} // namespace aicore
