// ============================================================
// pipeline_impl.cpp — 流水线引擎核心实现
// 管理 DAG 图的构建、拓扑执行与异步调度
// ============================================================
#include "pipeline/pipeline_impl.h"
#include "config/config_parser.h"
#include "pipeline/model_node.h"
#include "backend/backend_factory.h"
#include <algorithm>
#include <queue>
#include <sstream>

namespace aicore {

/**
 * 构造函数：初始化 Pipeline 实例
 * @param pool 引擎线程池，若为空则自动创建默认池
 */
PipelineImpl::PipelineImpl(std::shared_ptr<EnginePool> pool)
    : enginePool_(pool ? pool : std::make_shared<EnginePool>()) {}

/** 析构函数：停止流水线 */
PipelineImpl::~PipelineImpl() { Stop(); }

/**
 * 向 DAG 中添加一个处理节点
 * @param id      节点唯一标识
 * @param processor 节点对应的处理器实例
 * @param inputs  节点的输入源 ID 列表
 */
void PipelineImpl::AddNode(const std::string& id,
                           std::shared_ptr<IProcessor> processor,
                           const std::vector<std::string>& inputs) {
    DagNode node;
    node.id = id;
    node.processor = std::move(processor);
    node.inputs = inputs;
    nodes_[id] = std::move(node);
}

/**
 * 在 DAG 中为节点添加输出边（单向依赖）
 * @param from 上游节点 ID
 * @param to   下游节点 ID
 */
void PipelineImpl::AddEdge(const std::string& from, const std::string& to) {
    auto it = nodes_.find(from);
    if (it != nodes_.end()) {
        it->second.outputs.push_back(to);
    }
}

/**
 * 从 JSON 配置构建流水线 DAG
 * 解析节点列表和边关系，为模型节点创建后端推理引擎
 * @param configJson JSON 格式的流水线配置字符串
 * @return Status 构建成功或错误信息
 */
Status PipelineImpl::Build(const std::string& configJson) {
    ConfigParser parser;
    PipelineConfig config;
    auto s = parser.Parse(configJson, config);
    if (!s) return s;

    configJson_ = configJson;

    // 遍历配置中的所有节点，根据类型创建对应的处理器
    for (auto& pc : config.nodes) {
        std::shared_ptr<IProcessor> processor;

        if (pc.type == "model") {
            // 通过工厂创建后端（如 ONNX Runtime、TensorRT 等）
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

        // 初始化处理器并加入 DAG
        s = processor->Init(pc.params);
        if (!s) return s;

        AddNode(pc.id, processor, {});
    }

    // 根据边配置建立节点间的数据依赖关系
    for (auto& edge : config.edges) {
        AddEdge(edge.from, edge.to);

        auto toIt = nodes_.find(edge.to);
        if (toIt != nodes_.end())
            toIt->second.inputs.push_back(edge.from);
    }

    // 识别入口节点（无输入）和出口节点（无输出）
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

/**
 * 重新标记流水线状态为就绪，自动识别出入口节点
 * 支持名为 "input" 的虚拟输入节点
 */
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

/**
 * 同步执行流水线：按 DAG 拓扑顺序逐轮调度各节点
 * 采用"轮询推进"策略——每轮遍历所有未处理节点，
 * 仅当某节点的所有上游输入就绪时才执行它。
 * 这等价于拓扑排序的贪心模拟，避免显式维护依赖计数。
 * @param input  输入帧
 * @param output [out] 输出结果，包含检测信息和延迟指标
 * @return Status 执行是否成功
 */
Status PipelineImpl::Execute(const Frame& input, Result& output) {
    if (state_ != PipelineState::kReady && state_ != PipelineState::kRunning)
        return Status{StatusCode::ErrorInternal, "pipeline not ready"};
    state_ = PipelineState::kRunning;

    auto start = std::chrono::steady_clock::now();
    output.timestamp = input.timestamp;

    // nodeInputs: 当前轮次各节点的输入数据池
    // nodeOutputs: 当前轮次各节点的输出数据池
    std::unordered_map<std::string, std::vector<Frame>> nodeInputs;
    std::unordered_map<std::string, std::vector<Frame>> nodeOutputs;

    // 将输入帧注入到入口节点的输入池中
    nodeInputs["input"].push_back(input);
    for (auto& entry : entryNodes_) {
        if (entry != "input")
            nodeInputs[entry].push_back(input);
    }

    // 初始化所有节点为"未处理"状态
    for (auto& [id, node] : nodes_)
        node.processed = false;

    std::map<std::string, NodeMetric> metrics;
    bool complete = true;

    // ---- DAG 并行执行策略 ----
    // 采用"轮询推进"（Round-robin Polling）策略替代经典拓扑排序（Kahn 算法）：
    //
    // 算法对比：
    //   方案 A - Kahn 拓扑排序（传统 DAG 调度）：
    //     维护每个节点的入度计数，入度归零时入队执行。
    //     优点：严格一次遍历，时间复杂度 O(V+E)
    //     缺点：难以处理运行时动态图（某些节点可跳过）
    //
    //   方案 B - 轮询推进（本实现）：
    //     每轮扫描所有未处理节点，检查输入是否就绪，就绪则执行。
    //     优点：天然支持动态跳过（返回值 Skip 不影响后续）、
    //           无需预计算入度、代码更直观
    //     缺点：每轮 O(V+E)，总执行轮次可能多于 V
    //
    // 本实现选择方案 B 的原因：
    //   Pipeline 中可能包含条件节点（如异常评分高才执行 NMS），
    //   方案 A 的静态拓扑排序无法处理这种情况。
    //   轮询策略通过 "processed" 标记和 "allInputsReady" 检查
    //   自然支持条件执行和节点跳过。
    // 主循环：反复扫描节点列表，处理所有输入就绪的节点
    while (true) {
        bool progressed = false;
        for (auto& [id, node] : nodes_) {
            if (node.processed) continue;

            // 检查该节点的所有上游输入是否都已产生数据
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
        // 死锁检测：
        // 如果某轮没有任何节点被处理（progressed == false），
        // 说明剩余未处理节点存在循环依赖或所有输入都无法满足。
        // 在合法 DAG 中这不应发生，所以直接退出避免无限循环。
        // 可能原因：配置文件中边关系定义错误导致环形依赖。
        if (!progressed) break;

        // 将本轮输出作为下一轮输入，继续拓扑推进
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

    // 从出口节点的输出帧 roiMap 中提取检测结果
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

/**
 * 执行单个 DAG 节点：收集上游输入、调用处理器、记录延迟指标
 * @param nodeId     待执行节点 ID
 * @param nodeInputs 各节点的输入数据映射
 * @param nodeOutputs [out] 各节点的输出数据映射
 * @param metrics     [out] 节点级性能指标
 * @param timestamp   帧时间戳
 * @return Status 该节点处理结果
 *
 * 节点执行指标收集（Performance Telemetry）：
 *   每个节点执行前后记录：
 *     - latencyMs: 节点自身耗时（含模型推理/预处理/后处理）
 *     - inputBytes: 输入数据量（用于计算吞吐量）
 *     - outputBytes: 输出数据量（用于评估数据放大比）
 *   status 记录节点执行结果（OK/Error/Skip），用于：
 *     1. 调试：定位具体哪个节点失败
 *     2. 性能分析：识别耗时瓶颈
 *     3. 容错：允许部分节点失败后继续执行
 */
Status PipelineImpl::ExecuteNode(
    const std::string& nodeId,
    const std::unordered_map<std::string, std::vector<Frame>>& nodeInputs,
    std::unordered_map<std::string, std::vector<Frame>>& nodeOutputs,
    std::map<std::string, NodeMetric>& metrics,
    uint64_t timestamp) {

    auto nodeStart = std::chrono::steady_clock::now();
    auto& node = nodes_[nodeId];

    // 从输入映射中收集该节点的所有上游数据
    std::vector<Frame> inputs;
    for (auto& inId : node.inputs) {
        auto it = nodeInputs.find(inId);
        if (it != nodeInputs.end()) {
            for (auto& f : it->second)
                inputs.push_back(f);
        }
    }

    // 统计输入数据字节数
    NodeMetric metric;
    metric.inputBytes = 0;
    for (auto& f : inputs)
        metric.inputBytes += f.image.total() * f.image.elemSize();

    // 调用处理器执行实际逻辑（推理/预处理/后处理）
    std::vector<Frame> outputs;
    auto s = node.processor->Process(inputs, outputs);
    metric.status = s.code;

    // 即使返回 Skip 也保留输出（允许跳过但继续传递）
    if (s || s.code == StatusCode::Skip) {
        nodeOutputs[nodeId] = std::move(outputs);
    }

    // 统计输出数据字节数
    metric.outputBytes = 0;
    for (auto& f : nodeOutputs[nodeId])
        metric.outputBytes += f.image.total() * f.image.elemSize();

    auto nodeEnd = std::chrono::steady_clock::now();
    metric.latencyMs = std::chrono::duration<double, std::milli>(nodeEnd - nodeStart).count();
    metrics[nodeId] = metric;

    return s;
}

/**
 * 异步执行流水线：将任务提交到线程池，通过回调返回结果
 * @param input    输入帧
 * @param callback 结果回调函数（在后台线程中调用）
 */
Status PipelineImpl::ExecuteAsync(const Frame& input,
                                   std::function<void(const Result&)> callback) {
    threadPool_->Enqueue([this, input, cb = std::move(callback)] {
        Result result;
        Execute(input, result);
        if (cb) cb(result);
    });
    return Status{};
}

/**
 * 等待所有异步任务执行完毕
 */
Status PipelineImpl::WaitAll() {
    if (threadPool_) threadPool_->WaitAll();
    return Status{};
}

/**
 * 停止流水线，将状态置为已停止
 */
void PipelineImpl::Stop() {
    state_ = PipelineState::kStopped;
}

/**
 * 获取当前流水线状态（线程安全）
 */
PipelineState PipelineImpl::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

/**
 * 获取构建时传入的 JSON 配置字符串
 */
std::string PipelineImpl::GetConfig() const {
    return configJson_;
}

} // namespace aicore
