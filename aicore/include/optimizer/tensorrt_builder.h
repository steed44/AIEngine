// ============================================================
// 文件：tensorrt_builder.h
// 用途：定义 TensorRT 引擎构建器的枚举、配置结构和类，
//   负责将 ONNX 模型转换为 TensorRT 优化引擎。
// ============================================================
#pragma once
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

// Precision：TensorRT 推理精度枚举
enum class Precision { kFP32, kFP16, kINT8 };

// BuildConfig：TensorRT 引擎构建配置
// 包含 ONNX 路径、引擎保存路径、精度、批次大小、设备等参数
struct BuildConfig {
    std::string onnxPath;       // 输入 ONNX 模型文件路径
    std::string enginePath;     // 输出 TensorRT 引擎文件路径
    Precision precision = Precision::kFP16; // 推理精度（默认 FP16）
    int maxBatchSize = 1;       // 最大批次大小
    int deviceId = 0;           // GPU 设备 ID
    size_t workspaceSize = 1ULL << 30; // TensorRT 工作空间大小（默认 1GB）
    bool dynamicBatch = false;  // 是否启用动态批次
    std::string calibDir;       // INT8 校准数据目录（INT8 精度时使用）
    int numCalib = 100;         // INT8 校准样本数
};

// TensorRtBuilder：TensorRT 引擎构建器
// 职责：加载 ONNX 模型，应用精度优化和层融合，
//   生成可在 NVIDIA GPU 上高效运行的推理引擎。
// 典型使用场景：ONNX 导出完成后，构建 TensorRT 引擎用于部署。
class AICORE_OPTIMIZER_API TensorRtBuilder {
public:
    TensorRtBuilder();
    ~TensorRtBuilder();

    // 根据配置构建 TensorRT 引擎
    // 参数 config : 含 ONNX 路径、精度、批次大小等参数的构建配置
    Status Build(const BuildConfig& config);

    // 获取最后一条错误信息
    std::string GetLastError() const;

private:
    std::string lastError_;  // 最近一次操作的错误信息
};

} // namespace aicore
