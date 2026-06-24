// Pipeline 实现 — 基于 DAG 拓扑的推理流水线引擎
#pragma once
#include "core/pipeline.h"
#include "core/processor.h"
#include "engine/thread_pool.h"
#include "engine/engine_pool.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace aicore {

// DAG 节点结构
// 包含节点 ID、输入输出依赖关系、处理器实例和执行状态
struct DagNode {
    std::string id;                        // 节点唯一标识
    std::vector<std::string> outputs;      // 下游节点 ID 列表
    std::vector<std::string> inputs;       // 上游节点 ID 列表
    std::shared_ptr<IProcessor> processor; // 处理器实例
    bool processed = false;                // 当前执行轮次中是否已处理
};

// Pipeline 实现类
// 管理 DAG 的构建、拓扑排序执行和异步调度
class PipelineImpl : public IPipeline {
public:
    explicit PipelineImpl(std::shared_ptr<EnginePool> pool = nullptr);
    ~PipelineImpl() override;

    Status Build(const std::string& configJson) override;
    Status Execute(const Frame& input, Result& output) override;
    Status ExecuteAsync(const Frame& input,
                        std::function<void(const Result&)> callback) override;
    Status WaitAll() override;
    void Stop() override;
    PipelineState GetState() const override;
    std::string GetConfig() const override;

    // 手动添加节点到 DAG（用于非 JSON 配置方式构建）
    void AddNode(const std::string& id, std::shared_ptr<IProcessor> processor,
                 const std::vector<std::string>& inputs);
    // 添加节点间的有向依赖边
    void AddEdge(const std::string& from, const std::string& to);
    // 重新标记 pipeline 为就绪状态
    void MarkReady();

    void SetThreadPool(std::shared_ptr<ThreadPool> pool) { threadPool_ = std::move(pool); }
    ThreadPool* GetThreadPool() const { return threadPool_.get(); }

private:
    // 执行单个节点（内部方法，支持并发调用）
    Status ExecuteNode(const std::string& nodeId,
                       const std::unordered_map<std::string, std::vector<Frame>>& nodeInputs,
                       std::unordered_map<std::string, std::vector<Frame>>& nodeOutputs,
                       std::map<std::string, NodeMetric>& metrics,
                       uint64_t timestamp,
                       std::mutex* outputsMutex = nullptr);
    std::unordered_map<std::string, DagNode> nodes_;     // 所有 DAG 节点
    std::vector<std::string> entryNodes_;                  // 入口节点（无上游依赖）
    std::vector<std::string> exitNodes_;                   // 出口节点（无下游依赖）
    std::shared_ptr<ThreadPool> threadPool_;               // 线程池
    std::shared_ptr<EnginePool> enginePool_;               // 引擎池
    PipelineState state_ = PipelineState::kCreated;        // 生命周期状态
    std::string configJson_;                               // 原始 JSON 配置
    mutable std::mutex mutex_;                             // 线程安全锁
};

} // namespace aicore
