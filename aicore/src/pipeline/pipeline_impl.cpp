// ============================================================
// pipeline_impl.cpp — 流水线引擎核心实现
// 管理 DAG 图的构建、拓扑执行与异步调度
//
// DAG 执行引擎设计：
//
// 节点编排（Node Chaining）：
//   input → [nodeA] → [nodeB (依赖 A)] → [nodeC (依赖 B)] → output
//   每个节点从 nodeInputs 收集上游输出帧，处理后存入 nodeOutputs。
//   下一轮将 nodeOutputs 整体移为 nodeInputs，形成数据链。
//
// 并行调度（Round-Robin + Parallel）：
//   每轮扫描 nodes_，收集所有输入已就绪的节点；
//   就绪节点 > 1 时通过 ThreadPool 并发执行；
//   每个并发执行的节点通过 outputsMutex 保护 nodeOutputs 写操作。
//
// 线程安全模型：
//   nodeInputs/Outputs 在主循环线程写入，工作线程只读（已就绪的数据）；
//   ExecuteNode 内部写 nodeOutputs 时通过 outputsMutex 互斥；
//   每个节点的 status/processed 在 futures.get() 后主线程串行更新。
// ============================================================
#include "pipeline/pipeline_impl.h"
#include "config/config_parser.h"
#include "pipeline/model_node.h"
#include "postprocess/nms_node.h"
#include "backend/backend_factory.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <future>
#include <atomic>
#include <mutex>

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
    auto toIt = nodes_.find(to);
    if (toIt != nodes_.end()) {
        toIt->second.inputs.push_back(from);
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
            processor = std::make_shared<ModelNode>(std::move(backend));
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
 * 同步执行流水线：按 DAG 拓扑顺序每轮并行调度所有就绪节点
 *
 * 采用"轮询推进 + 并发"策略：
 *   每轮扫描所有未处理节点，收集输入就绪的节点；
 *   若有线程池且就绪节点数 > 1，通过 ThreadPool 并行执行；
 *   否则串行执行（退化路径）。
 *
 * 算法对比：
 *   方案 A - Kahn 拓扑排序（传统 DAG 调度）：
 *     维护入度计数，入度归零时入队执行。
 *     缺点：难以处理运行时动态图（某些节点可跳过）。
 *
 *   方案 B - 轮询推进 + 并行（本实现）：
 *     每轮扫描 → 收集就绪节点 → 并发/串行执行 → 推进下一轮。
 *     优点：天然支持动态跳过（返回值 Skip 不影响后续）、
 *           无需预计算入度、每轮独立节点自动并行。
 *     缺点：每轮 O(V+E)，总执行轮次可能多于 V。
 */
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

    // 主循环：Kahn 拓扑排序 + 并行调度
    //
    // Kahn 算法 vs 轮询推进：
    //   轮询推进：每轮扫描所有未处理节点 → O(V²) 总复杂度
    //   Kahn 算法：维护入度计数，节点就绪时入队 → O(V+E) 总复杂度
    //
    // 本实现保留了轮询推进的兼容性（支持动态跳过节点），
    // 但改用入度计数驱动就绪检测，减少无效扫描。
    //
    // 算法步骤：
    //   1. 初始化：entryNodes 入度为 0，入 readyQueue
    //   2. 循环：从 readyQueue 取节点 → 执行 → 完成后
    //      对每个下游节点减少入度 → 入度归零则入队
    //   3. 所有节点 processed 时退出
    //
    // 并行执行：同一拓扑层（同时入队的节点）可并行执行
    // 因为它们的输入都已就绪，互不依赖。

    // 入度计数：每个节点的未就绪上游节点数
    std::unordered_map<std::string, int> inDegree;
    for (auto& [id, node] : nodes_) {
        int degree = 0;
        for (auto& inId : node.inputs) {
            if (!inId.empty() && inId != "input") degree++;
        }
        inDegree[id] = degree;
    }

    // 就绪队列：入度为 0 的节点
    std::vector<std::string> readyQueue;
    for (auto& entry : entryNodes_) {
        if (entry != "input") {
            readyQueue.push_back(entry);
            // entry 节点入度应为 0（已验证）
        }
    }

    // Kahn 主循环：按拓扑层推进
    while (!readyQueue.empty()) {
        std::vector<std::string> currentLayer = std::move(readyQueue);
        readyQueue.clear();

        // 并行：同层节点无依赖，可并发执行
        if (currentLayer.size() > 1 && threadPool_) {
            std::mutex outputsMutex;
            std::atomic<bool> failed{false};
            std::vector<std::future<Status>> futures;
            futures.reserve(currentLayer.size());

            for (auto& id : currentLayer) {
                futures.push_back(threadPool_->Enqueue([&, id]() -> Status {
                    if (failed.load()) return Status{StatusCode::ErrorInternal, ""};
                    auto s = ExecuteNode(id, nodeInputs, nodeOutputs,
                                         metrics, input.timestamp, &outputsMutex);
                    if (!s && s.code != StatusCode::Skip) {
                        failed.store(true);
                    }
                    return s;
                }));
            }

            for (size_t i = 0; i < currentLayer.size(); i++) {
                auto s = futures[i].get();
                if (s || s.code == StatusCode::Skip) {
                    nodes_[currentLayer[i]].processed = true;
                    // 下游节点入度减 1，归零则入队
                    for (auto& outId : nodes_[currentLayer[i]].outputs) {
                        inDegree[outId]--;
                        if (inDegree[outId] <= 0) {
                            readyQueue.push_back(outId);
                        }
                    }
                } else {
                    output.status = s.code;
                    output.errorMsg = s.message;
                    complete = false;
                }
            }
        } else {
            // 串行路径（单节点或无线程池）
            for (auto& id : currentLayer) {
                auto s = ExecuteNode(id, nodeInputs, nodeOutputs,
                                     metrics, input.timestamp);
                if (!s) {
                    output.status = s.code;
                    output.errorMsg = s.message;
                    complete = false;
                    break;
                }
                nodes_[id].processed = true;
                // 下游节点入度减 1，归零则入队
                for (auto& outId : nodes_[id].outputs) {
                    inDegree[outId]--;
                    if (inDegree[outId] <= 0) {
                        readyQueue.push_back(outId);
                    }
                }
            }
        }

        if (!complete) break;

        // 本层输出 → 下层输入
        nodeInputs = std::move(nodeOutputs);
        nodeOutputs.clear();
    }

    auto end = std::chrono::steady_clock::now();
    output.totalLatencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    output.nodeMetrics = std::move(metrics);

    // 从出口节点的输出帧提取检测结果和异常热力图
    for (auto& exitId : exitNodes_) {
        auto it = nodeOutputs.find(exitId);
        if (it == nodeOutputs.end()) continue;
        for (auto& f : it->second) {
            if (f.roiMap.empty()) continue;
            NodeResult nr;
            nr.nodeId = exitId;
            nr.measurements.insert(f.roiMap.begin(), f.roiMap.end());
            // 若帧图像是单通道浮点图（异常热力图），一并存入结果
            if (f.image.type() == CV_32F && f.image.channels() == 1) {
                nr.anomalyMap = f.image.clone();
            }
            output.detections.push_back(std::move(nr));
        }
    }

    // 如果 pipeline 中有 nms 节点，对检测结果执行 NMS 后处理
    if (nodes_.count("nms") && !output.detections.empty()) {
        NmsNode nmsNode;
        auto it = nodes_.find("nms");
        if (it != nodes_.end()) {
            // 从节点配置中获取 NMS 参数
            NodeConfig nmsConfig;
            auto procIt = nodes_.find("nms");
            if (procIt != nodes_.end()) {
                // NmsNode 已通过 Init 初始化，复用其处理器实例
                auto nmsProcessor = procIt->second.processor;
                // 执行 NMS
                std::vector<Frame> nmsInputs;
                Frame nmsFrame;
                nmsFrame.detections = std::move(output.detections);
                nmsInputs.push_back(std::move(nmsFrame));
                std::vector<Frame> nmsOutputs;
                nmsProcessor->Process(nmsInputs, nmsOutputs);
                if (!nmsOutputs.empty()) {
                    output.detections = std::move(nmsOutputs[0].detections);
                }
            }
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
    uint64_t timestamp,
    std::mutex* outputsMutex) {

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

    // 并行安全：写 nodeOutputs + 读回 + metrics 保持锁连续
    std::unique_lock<std::mutex> ulock;
    if (outputsMutex) ulock = std::unique_lock<std::mutex>(*outputsMutex);
    if (s || s.code == StatusCode::Skip) {
        nodeOutputs[nodeId] = std::move(outputs);
    }
    metric.outputBytes = 0;
    auto outIt = nodeOutputs.find(nodeId);
    if (outIt != nodeOutputs.end()) {
        for (auto& f : outIt->second)
            metric.outputBytes += f.image.total() * f.image.elemSize();
    }
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
