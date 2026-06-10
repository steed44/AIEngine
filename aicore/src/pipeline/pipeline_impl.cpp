#include "pipeline/pipeline_impl.h"
#include "config/config_parser.h"
#include "pipeline/model_node.h"
#include "backend/backend_factory.h"
#include <algorithm>
#include <queue>
#include <sstream>

namespace aicore {

PipelineImpl::PipelineImpl(std::shared_ptr<EnginePool> pool)
    : enginePool_(pool ? pool : std::make_shared<EnginePool>()) {}

PipelineImpl::~PipelineImpl() { Stop(); }

void PipelineImpl::AddNode(const std::string& id,
                           std::shared_ptr<IProcessor> processor,
                           const std::vector<std::string>& inputs) {
    DagNode node;
    node.id = id;
    node.processor = std::move(processor);
    node.inputs = inputs;
    nodes_[id] = std::move(node);
}

void PipelineImpl::AddEdge(const std::string& from, const std::string& to) {
    auto it = nodes_.find(from);
    if (it != nodes_.end()) {
        it->second.outputs.push_back(to);
    }
}

Status PipelineImpl::Build(const std::string& configJson) {
    ConfigParser parser;
    PipelineConfig config;
    auto s = parser.Parse(configJson, config);
    if (!s) return s;

    configJson_ = configJson;

    for (auto& pc : config.nodes) {
        std::shared_ptr<IProcessor> processor;

        if (pc.type == "model") {
            auto backend = BackendFactory::Create(pc.backend);
            if (!backend)
                return Status{StatusCode::ErrorConfigParse,
                              "unknown backend for " + pc.id};
            ModelInfo info;
            info.modelPath = pc.modelPath;
            info.backend = pc.backend;
            info.deviceId = pc.deviceId;
            info.batchSize = pc.batchSize;
            s = backend->Load(info);
            if (!s) return s;
            processor = std::make_shared<ModelNode>(std::shared_ptr<IModelBackend>(std::move(backend)));
        } else {
            return Status{StatusCode::ErrorConfigParse,
                          "unknown node type: " + pc.type};
        }

        s = processor->Init(pc.params);
        if (!s) return s;

        AddNode(pc.id, processor, {});
    }

    for (auto& edge : config.edges) {
        AddEdge(edge.from, edge.to);

        auto toIt = nodes_.find(edge.to);
        if (toIt != nodes_.end())
            toIt->second.inputs.push_back(edge.from);
    }

    for (auto& [id, node] : nodes_) {
        if (node.inputs.empty())
            entryNodes_.push_back(id);
        if (node.outputs.empty())
            exitNodes_.push_back(id);
    }

    threadPool_ = std::make_unique<ThreadPool>(config.maxConcurrency);
    state_ = PipelineState::kReady;
    return Status{};
}

void PipelineImpl::MarkReady() {
    entryNodes_.clear();
    exitNodes_.clear();
    for (auto& [id, node] : nodes_) {
        if (node.inputs.empty() || (node.inputs.size() == 1 && node.inputs[0] == "input"))
            entryNodes_.push_back(id);
        if (node.outputs.empty())
            exitNodes_.push_back(id);
    }
    state_ = PipelineState::kReady;
}

Status PipelineImpl::Execute(const Frame& input, Result& output) {
    if (state_ != PipelineState::kReady && state_ != PipelineState::kRunning)
        return Status{StatusCode::ErrorInternal, "pipeline not ready"};
    state_ = PipelineState::kRunning;

    auto start = std::chrono::steady_clock::now();
    output.timestamp = input.timestamp;

    std::unordered_map<std::string, std::vector<Frame>> nodeInputs;
    std::unordered_map<std::string, std::vector<Frame>> nodeOutputs;

    nodeInputs["input"].push_back(input);
    for (auto& entry : entryNodes_) {
        if (entry != "input")
            nodeInputs[entry].push_back(input);
    }

    for (auto& [id, node] : nodes_)
        node.processed = false;

    std::map<std::string, NodeMetric> metrics;
    bool complete = true;

    while (true) {
        bool progressed = false;
        for (auto& [id, node] : nodes_) {
            if (node.processed) continue;

            bool allInputsReady = true;
            for (auto& inId : node.inputs) {
                if (!inId.empty() && nodeInputs.find(inId) == nodeInputs.end()) {
                    allInputsReady = false;
                    break;
                }
            }
            if (!allInputsReady) continue;

            auto s = ExecuteNode(id, nodeInputs, nodeOutputs, metrics, input.timestamp);
            if (!s) {
                output.status = s.code;
                output.errorMsg = s.message;
                complete = false;
                break;
            }
            node.processed = true;
            progressed = true;
        }
        if (!complete) break;
        if (!progressed) break;

        nodeInputs = nodeOutputs;
        bool allProcessed = true;
        for (auto& [id, node] : nodes_) {
            if (!node.processed) { allProcessed = false; break; }
        }
        if (allProcessed) break;
    }

    auto end = std::chrono::steady_clock::now();
    output.totalLatencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    output.nodeMetrics = std::move(metrics);

    // 从出口节点输出帧的 roiMap 填充 detections
    for (auto& exitId : exitNodes_) {
        auto it = nodeOutputs.find(exitId);
        if (it == nodeOutputs.end()) continue;
        for (auto& f : it->second) {
            if (f.roiMap.empty()) continue;
            NodeResult nr;
            nr.nodeId = exitId;
            nr.measurements.insert(f.roiMap.begin(), f.roiMap.end());
            output.detections.push_back(std::move(nr));
        }
    }

    state_ = PipelineState::kReady;
    return complete ? Status{} : Status{output.status, output.errorMsg};
}

Status PipelineImpl::ExecuteNode(
    const std::string& nodeId,
    const std::unordered_map<std::string, std::vector<Frame>>& nodeInputs,
    std::unordered_map<std::string, std::vector<Frame>>& nodeOutputs,
    std::map<std::string, NodeMetric>& metrics,
    uint64_t timestamp) {

    auto nodeStart = std::chrono::steady_clock::now();
    auto& node = nodes_[nodeId];

    std::vector<Frame> inputs;
    for (auto& inId : node.inputs) {
        auto it = nodeInputs.find(inId);
        if (it != nodeInputs.end()) {
            for (auto& f : it->second)
                inputs.push_back(f);
        }
    }

    NodeMetric metric;
    metric.inputBytes = 0;
    for (auto& f : inputs)
        metric.inputBytes += f.image.total() * f.image.elemSize();

    std::vector<Frame> outputs;
    auto s = node.processor->Process(inputs, outputs);
    metric.status = s.code;

    if (s || s.code == StatusCode::Skip) {
        nodeOutputs[nodeId] = std::move(outputs);
    }

    metric.outputBytes = 0;
    for (auto& f : nodeOutputs[nodeId])
        metric.outputBytes += f.image.total() * f.image.elemSize();

    auto nodeEnd = std::chrono::steady_clock::now();
    metric.latencyMs = std::chrono::duration<double, std::milli>(nodeEnd - nodeStart).count();
    metrics[nodeId] = metric;

    return s;
}

Status PipelineImpl::ExecuteAsync(const Frame& input,
                                   std::function<void(const Result&)> callback) {
    threadPool_->Enqueue([this, input, cb = std::move(callback)] {
        Result result;
        Execute(input, result);
        if (cb) cb(result);
    });
    return Status{};
}

Status PipelineImpl::WaitAll() {
    if (threadPool_) threadPool_->WaitAll();
    return Status{};
}

void PipelineImpl::Stop() {
    state_ = PipelineState::kStopped;
}

PipelineState PipelineImpl::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string PipelineImpl::GetConfig() const {
    return configJson_;
}

} // namespace aicore
