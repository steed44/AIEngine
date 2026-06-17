// ============================================================
// 文件：int8_calibrator.h
// 用途：定义 INT8 量化校准器，为 TensorRT INT8 推理提供
//   校准数据集加载和批次获取功能。
// ============================================================
#pragma once
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

// Int8Calibrator：INT8 量化校准数据管理器
// 职责：加载校准数据集并逐批次提供给 TensorRT 校准器，
//   用于计算 INT8 量化的动态范围（scale / zero_point）。
// 典型使用场景：启用 INT8 精度优化时，在构建引擎前准备校准数据。
class AICORE_OPTIMIZER_API Int8Calibrator {
public:
    Int8Calibrator();

    // 从指定目录加载校准数据
    // 参数 calibDir   : 校准数据目录路径
    // 参数 numSamples : 使用的样本数量
    Status LoadCalibrationData(const std::string& calibDir, int numSamples);

    // 获取下一个校准批次数据
    // 返回值 : float 数组指针，按 CHW 或 NHWC 排列
    float* GetBatch();

    // 获取校准样本总数
    int GetNumSamples() const { return numSamples_; }

    // 获取最后一条错误信息
    std::string GetLastError() const;

private:
    std::vector<std::vector<float>> samples_;  // 存储所有校准样本数据
    int numSamples_ = 0;                        // 校准样本总数
    int currentIndex_ = 0;                      // 当前已分发的批次索引
    std::string lastError_;                     // 最近一次操作的错误信息
};

} // namespace aicore
