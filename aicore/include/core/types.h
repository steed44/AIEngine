// ============================================================
// types.h — AIEngine 核心基础类型定义
//
// 本文件定义了引擎全局使用的基础数据结构：
// - 导出声明宏（AICORE_API / AICORE_OPTIMIZER_API / AICORE_TRAINER_API）
// - 内存/数据类型枚举
// - 通用张量结构 Tensor
// - 状态码与错误状态
// - 检测框、检测结果、节点指标等推理输出结构
//
// 依赖关系：types.h 被所有其他头文件引用，是整个引擎的类型基础
// 设计原则：不依赖其他内部头文件，仅依赖标准库和 OpenCV 核心类型
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
// kPinned：页锁定内存，cudaHostAlloc 分配，用于异步 DMA 传输
// kGPU：显存，cudaMalloc 分配，由 MemoryManager 统一管理
// kCPU：常规堆内存，new/malloc 分配
enum class MemoryType { kCPU, kGPU, kPinned };

// 张量数据类型
// kUInt8：   8位无符号整数（图像输入、INT8 量化模型）
// kFloat32： 32位浮点数（主流推理精度，默认选择）
// kFloat16： 16位浮点数（FP16 加速，TensorRT 支持）
enum class DataType { kUInt8, kFloat32, kFloat16 };

// 通用张量结构，是引擎内部传递数据的基本单元
// 通过裸指针 data 指向实际内存，配合 shape/dtype/bytes 描述数据布局
//
// 所有权规则：
//   allocId == 0：data 由外部管理，Tensor 只引用不拥有
//   allocId == 1：data 由后端分配（new float[]），调用方需 delete[] 释放
//   allocId > 1： data 由 MemoryManager 管理，通过 Free(allocId) 归还
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
//
// 错误处理模式：所有可能失败的函数返回 Status（OK | Skip | Error*），
//   调用方通过 if (status) 或 status.code 判断是否成功。
//   错误信息通过 status.message 获取详细描述。
//
// 状态码按阶段分类：
//   OK/Skip     — 正常状态（0~1）
//   Config      — 配置相关错误（10~19）
//   Model       — 模型相关错误（20~29）
//   Pre/Post    — 前后处理错误（30~39）
//   Resource    — 资源不足错误（40~49）
//   Runtime     — 运行时错误（50~59）
enum class StatusCode {
    OK = 0,                  // 成功：操作正常完成
    Skip,                    // 节点跳过：条件分支不满足时主动跳过当前节点
    ErrorConfigParse,        // 配置文件解析失败：JSON 格式错误或缺少必要字段
    ErrorModelLoad,          // 模型加载失败：文件不存在、格式不兼容或设备不匹配
    ErrorModelInfer,         // 模型推理失败：推理框架内部错误（如形状不匹配）
    ErrorPreprocess,         // 预处理失败：图像解码、缩放、归一化等预处理步骤出错
    ErrorPostprocess,        // 后处理失败：NMS、解码、阈值筛选等后处理步骤出错
    ErrorResourceExhaust,    // 资源耗尽：GPU 显存不足或系统内存不足
    ErrorTimeout,            // 推理超时：超过预设的时间阈值仍未完成
    ErrorInvalidInput,       // 输入数据无效：空图像、格式不支持或尺寸异常
    ErrorInternal,           // 内部未知错误：未分类的运行时错误
    ErrorGpuDevice           // GPU 设备错误：CUDA 驱动问题或设备丢失
};

// 统一状态返回结构，引擎中所有可能失败的函数均返回此类型
// 支持 bool 隐式转换：if (status) 等价于 status.code == OK
//
// 使用模式：
//   Status s = DoSomething();
//   if (!s) { LOG_ERROR << s.message; return s; }
//   // 或
//   RETURN_IF_ERROR(DoSomething());
struct Status {
    StatusCode code = StatusCode::OK;   // 状态码
    std::string message;                // 错误描述信息
    operator bool() const noexcept { return code == StatusCode::OK; }
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
    cv::Mat anomalyMap;                            // PatchCore 异常热力图（CV_32F，与原图同尺寸）
};

// 单节点执行指标，用于性能监控和调试
struct NodeMetric {
    double latencyMs = 0;          // 节点执行耗时（毫秒）
    size_t inputBytes = 0;         // 输入数据量
    size_t outputBytes = 0;        // 输出数据量
    StatusCode status = StatusCode::OK;  // 节点执行状态
};

// 流水线一次推理的完整输出结果
// 包含所有检测节点的输出、各节点性能指标和整体执行状态
struct Result {
    uint64_t timestamp = 0;                      // 推理时间戳（毫秒，steady_clock）
    double totalLatencyMs = 0;                   // 流水线总耗时（毫秒，端到端）
    std::vector<NodeResult> detections;          // 所有检测节点输出的检测结果
    std::map<std::string, NodeMetric> nodeMetrics;  // 各节点性能指标（key = nodeId）
    StatusCode status = StatusCode::OK;          // 整体执行状态
    std::string errorMsg;                        // 错误描述（status 非 OK 时有效）
};

} // namespace aicore
