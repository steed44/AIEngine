// ============================================================
// 文件：int8_calibrator.cpp
// 用途：实现 INT8 量化校准器，当前为桩实现（stub），
//   后续接入 TensorRT 的 IInt8Calibrator 接口。
// ============================================================
#include "optimizer/int8_calibrator.h"

namespace aicore {

Int8Calibrator::Int8Calibrator() {}

// 加载校准数据集：目前为桩实现，
// 仅记录样本数，实际文件读取需对接具体数据格式。
Status Int8Calibrator::LoadCalibrationData(const std::string& calibDir,
                                            int numSamples) {
    // calibDir 参数暂未使用，后续实现文件扫描与加载
    (void)calibDir;
    numSamples_ = numSamples;
    return Status{};
}

// 获取下一个校准批次：桩实现，返回空指针，
// 实际应返回 samples_ 中当前批次数据的指针并递增 currentIndex_。
float* Int8Calibrator::GetBatch() {
    return nullptr;
}

std::string Int8Calibrator::GetLastError() const { return lastError_; }

} // namespace aicore
