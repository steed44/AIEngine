// pipeline 构建器
// 根据配置描述创建完整的 IPipeline 实例，组装处理器节点并建立连接
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
