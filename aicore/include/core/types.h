#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <opencv2/core.hpp>

namespace aicore {

#ifdef AICORE_EXPORTS
#define AICORE_API __declspec(dllexport)
#else
#define AICORE_API __declspec(dllimport)
#endif

#ifdef AICORE_OPTIMIZER_EXPORTS
#define AICORE_OPTIMIZER_API __declspec(dllexport)
#else
#define AICORE_OPTIMIZER_API __declspec(dllimport)
#endif

#ifdef AICORE_TRAINER_EXPORTS
#define AICORE_TRAINER_API __declspec(dllexport)
#else
#define AICORE_TRAINER_API __declspec(dllimport)
#endif

enum class MemoryType { kCPU, kGPU, kPinned };
enum class DataType { kUInt8, kFloat32, kFloat16 };

struct Tensor {
    DataType dtype = DataType::kFloat32;
    std::vector<int64_t> shape;
    MemoryType memory = MemoryType::kCPU;
    void* data = nullptr;
    size_t bytes = 0;
    size_t allocId = 0;
};

enum class StatusCode {
    OK = 0,
    Skip,
    ErrorConfigParse,
    ErrorModelLoad,
    ErrorModelInfer,
    ErrorPreprocess,
    ErrorPostprocess,
    ErrorResourceExhaust,
    ErrorTimeout,
    ErrorInvalidInput,
    ErrorInternal,
    ErrorGpuDevice
};

struct Status {
    StatusCode code = StatusCode::OK;
    std::string message;
    operator bool() const { return code == StatusCode::OK; }
};

struct BBox {
    float x = 0, y = 0, w = 0, h = 0;
};

struct NodeResult {
    std::string nodeId;
    std::string label;
    float confidence = 0;
    BBox bbox;
    cv::Mat roi;
    std::map<std::string, double> measurements;
};

struct NodeMetric {
    double latencyMs = 0;
    size_t inputBytes = 0;
    size_t outputBytes = 0;
    StatusCode status = StatusCode::OK;
};

struct Result {
    uint64_t timestamp = 0;
    double totalLatencyMs = 0;
    std::vector<NodeResult> detections;
    std::map<std::string, NodeMetric> nodeMetrics;
    StatusCode status = StatusCode::OK;
    std::string errorMsg;
};

} // namespace aicore
