// ============================================================
// types.h — AIEngine 核心基础类型定义
//
// 本文件定义了引擎全局使用的基础数据结构：
// - 导出声明宏（AICORE_API / AICORE_OPTIMIZER_API / AICORE_TRAINER_API）
// - 内存/数据类型枚举
// - 通用张量结构 Tensor
// - 状态码与错误状态
// - 检测框、检测结果、节点指标等推理输出结构
// ============================================================

#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <opencv2/core.hpp>

namespace aicore {

// ========== DLL 导出宏 ==========
// 三个模块分别使用独立的导出宏，确保跨 DLL 边界符号正确可见：
//   AICORE_EXPORTS        → aicore.dll（推理核心）
//   AICORE_OPTIMIZER_EXPORTS → aicore_optimizer.dll（模型优化）
//   AICORE_TRAINER_EXPORTS   → aicore_trainer.dll（训练模块）
// 编译 DLL 时由 CMake 添加对应定义，消费者（exe 或其他 DLL）则引入导入声明。

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

// 内存位置类型：CPU 内存 / GPU 显存 / 固定内存（pinned memory，用于加速 CPU↔GPU 传输）
enum class MemoryType { kCPU, kGPU, kPinned };

// 张量数据类型
enum class DataType { kUInt8, kFloat32, kFloat16 };

// 通用张量结构，是引擎内部传递数据的基本单元
// 通过裸指针 data 指向实际内存，配合 shape/dtype/bytes 描述数据布局
struct Tensor {
    DataType dtype = DataType::kFloat32;      // 数据类型
    std::vector<int64_t> shape;               // 形状（如 {N, C, H, W}）
    MemoryType memory = MemoryType::kCPU;     // 内存位置
    void* data = nullptr;                     // 数据指针（不管理生命周期）
    size_t bytes = 0;                         // 占用字节数
    size_t allocId = 0;                       // 分配 ID，用于内存池追踪
};

// 引擎全局状态码枚举
// 覆盖推理全链路中所有可能的成功/失败场景，便于上层统一错误处理
enum class StatusCode {
    OK = 0,                  // 成功
    Skip,                    // 节点跳过（如条件分支不满足时）
    ErrorConfigParse,        // 配置文件解析失败
    ErrorModelLoad,          // 模型加载失败
    ErrorModelInfer,         // 模型推理失败
    ErrorPreprocess,         // 预处理失败
    ErrorPostprocess,        // 后处理失败
    ErrorResourceExhaust,    // 资源耗尽（显存/内存不足）
    ErrorTimeout,            // 推理超时
    ErrorInvalidInput,       // 输入数据无效
    ErrorInternal,           // 内部未知错误
    ErrorGpuDevice           // GPU 设备错误（驱动/设备丢失）
};

// 统一状态返回结构，引擎中所有可能失败的函数均返回此类型
// 支持 bool 隐式转换：if (status) 等价于 status.code == OK
struct Status {
    StatusCode code = StatusCode::OK;   // 状态码
    std::string message;                // 错误描述信息
    operator bool() const { return code == StatusCode::OK; }
};

// 检测框，采用中心点坐标 + 宽高表示
struct BBox {
    float x = 0, y = 0, w = 0, h = 0;   // 中心 x, 中心 y, 宽度, 高度
};

// 单个检测节点的输出结果
// 每个 NodeResult 对应一个检测目标（如一个缺陷/一个物体）
struct NodeResult {
    std::string nodeId;                            // 来源流水线节点 ID
    std::string label;                             // 分类标签（如 "scratch", "dent"）
    float confidence = 0;                          // 置信度 [0, 1]
    BBox bbox;                                     // 检测框（原始图像坐标系）
    cv::Mat roi;                                    // 裁剪的 ROI 图像（可选）
    std::map<std::string, double> measurements;    // 附加测量值（如面积、长度等）
};

// 单节点执行指标，用于性能监控和调试
struct NodeMetric {
    double latencyMs = 0;          // 节点执行耗时（毫秒）
    size_t inputBytes = 0;         // 输入数据量
    size_t outputBytes = 0;        // 输出数据量
    StatusCode status = StatusCode::OK;  // 节点执行状态
};

// 流水线一次推理的完整输出结果
struct Result {
    uint64_t timestamp = 0;                      // 推理时间戳（毫秒）
    double totalLatencyMs = 0;                   // 流水线总耗时（毫秒）
    std::vector<NodeResult> detections;          // 所有检测节点输出的检测结果
    std::map<std::string, NodeMetric> nodeMetrics;  // 各节点性能指标（key = nodeId）
    StatusCode status = StatusCode::OK;          // 整体执行状态
    std::string errorMsg;                        // 错误描述（status 非 OK 时有效）
};

} // namespace aicore
