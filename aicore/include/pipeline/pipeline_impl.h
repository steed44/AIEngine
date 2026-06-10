#pragma once
#include "core/pipeline.h"
#include "core/processor.h"
#include "engine/thread_pool.h"
#include "engine/engine_pool.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace aicore {

struct DagNode {
    std::string id;
    std::vector<std::string> outputs;
    std::vector<std::string> inputs;
    std::shared_ptr<IProcessor> processor;
    bool processed = false;
};

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

    void AddNode(const std::string& id, std::shared_ptr<IProcessor> processor,
                 const std::vector<std::string>& inputs);
    void AddEdge(const std::string& from, const std::string& to);
    void MarkReady();

private:
    Status ExecuteNode(const std::string& nodeId,
                       const std::unordered_map<std::string, std::vector<Frame>>& nodeInputs,
                       std::unordered_map<std::string, std::vector<Frame>>& nodeOutputs,
                       std::map<std::string, NodeMetric>& metrics,
                       uint64_t timestamp);

    std::unordered_map<std::string, DagNode> nodes_;
    std::vector<std::string> entryNodes_;
    std::vector<std::string> exitNodes_;
    std::unique_ptr<ThreadPool> threadPool_;
    std::shared_ptr<EnginePool> enginePool_;
    PipelineState state_ = PipelineState::kCreated;
    std::string configJson_;
    mutable std::mutex mutex_;
};

} // namespace aicore
