// pipeline 构建器
// 根据配置描述创建完整的 IPipeline 实例，组装处理器节点并建立连接
//
// 设计模式：工厂方法（Factory Method）
//   PipelineBuilder::Build() 根据 ProcessorConfig::type 字符串，
//   通过 if-else 链创建对应的 IProcessor 子类实例。
//   这种设计使得新增节点类型只需在 builder 中添加一个分支，
//   无需修改其他代码。
//
// 构建流程：
//   ConfigParser::Parse(json) → PipelineConfig
//   → PipelineBuilder::Build(config) → IPipeline
//     → for each node: 根据 type 创建 IProcessor → Init(config)
//     → for each edge: 连接源节点输出到目标节点输入
//     → 用 EnginePool 管理模型后端实例（可选）
#pragma once
#include "core/processor.h"
#include "core/pipeline.h"
#include "config/config_parser.h"
#include "engine/engine_pool.h"
#include <memory>
#include <unordered_map>
#include <vector>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// pipeline 构建器（静态工具类）
// 将 PipelineConfig 描述的拓扑结构实例化为可运行的 IPipeline 对象
class PipelineBuilder {
public:
    // 根据配置构建 pipeline 实例
    // @param config   pipeline 配置结构体（含节点和边定义）
    // @param pipeline 输出的 pipeline 唯一指针（引用传出）
    // @param pool     引擎池（可选），用于复用模型后端实例
    // @return 成功返回 Status::kOk
    Status Build(const PipelineConfig& config,
                 std::unique_ptr<IPipeline>& pipeline,
                 std::shared_ptr<EnginePool> pool = nullptr);
};

} // namespace aicore
