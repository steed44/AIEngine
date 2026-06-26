// pipeline 配置解析与序列化
// 提供 JSON 格式配置的解析和序列化能力，将配置转换为 PipelineConfig 结构体
//
// JSON 配置格式示例：
// {
//   "pipeline": {
//     "name": "yolo_demo",
//     "max_concurrency": 4,
//     "enable_profiling": true,
//     "nodes": [
//       {"id": "resize", "type": "resize", "params": {"width": "640", "height": "640"}},
//       ...
//     ],
//     "edges": [
//       {"from": "input", "to": "resize"},
//       ...
//     ]
//   }
// }
//
// 设计说明：使用 JSON 作为配置格式，便于人类阅读和版本控制。
//   内部使用 PipelineConfig 结构体表示，避免运行时 JSON 解析开销。
#pragma once
#include "core/types.h"
#include "core/model_backend.h"
#include <string>
#include <vector>
#include <unordered_map>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 单个处理器节点的配置结构
struct ProcessorConfig {
    std::string id;                                         // 节点唯一标识
    std::string type;                                       // 节点类型（如 "Preprocessor", "Inference"）
    BackendType backend = BackendType::kUnknown;            // 推理后端类型（仅推理节点有效）
    std::string modelPath;                                  // 模型文件路径（仅推理节点有效）
    int deviceId = 0;                                       // 推理设备 ID
    int batchSize = 1;                                      // 批处理大小
    std::unordered_map<std::string, std::string> params;    // 节点自定义参数（键值对）
};

// 处理器节点之间的连接边配置
struct EdgeConfig {
    std::string from;   // 源节点 ID
    std::string to;     // 目标节点 ID
};

// 完整 pipeline 的配置结构
struct PipelineConfig {
    std::string name;                               // pipeline 名称
    std::vector<ProcessorConfig> nodes;             // 处理器节点列表
    std::vector<EdgeConfig> edges;                  // 节点连接关系列表
    int maxConcurrency = 4;                         // 最大并行处理数，默认 4
    bool enableProfiling = false;                   // 是否启用性能分析，默认关闭
};

// 配置解析器
// 负责 JSON 字符串与 PipelineConfig 结构体之间的双向转换
class ConfigParser {
public:
    // 将 JSON 字符串解析为 PipelineConfig 结构体
    // @param jsonStr JSON 格式的配置字符串
    // @param config  解析结果（引用传出）
    // @return 成功返回 Status::kOk
    // 前置条件：jsonStr 是合法的 JSON 格式（非空）
    // 后置条件：成功时 config 填充所有节点和边信息；失败时 config 不变
    // 解析策略：先整体解析 JSON 结构，再逐字段校验，遇错立即返回
    Status Parse(const std::string& jsonStr, PipelineConfig& config);
    // 将 PipelineConfig 结构体序列化为 JSON 字符串
    // @param config  pipeline 配置结构体
    // @param jsonStr 输出的 JSON 字符串（引用传出）
    // @return 成功返回 Status::kOk
    // 使用场景：调试输出或保存运行时的 pipeline 配置快照
    Status Serialize(const PipelineConfig& config, std::string& jsonStr);
    // 获取上次操作（Parse/Serialize）的错误信息
    // @return 错误描述字符串，无错误时返回空字符串
    // 注意：错误信息在一次新的操作后会被重置覆盖
    std::string GetLastError() const;

private:
    std::string lastError_; // 最近一次操作的错误信息缓存
};

} // namespace aicore
