// AI 引擎单例实现
// 管理 pipeline 的初始化、执行、异步调度和资源释放
//
// 引擎生命周期：
//   1. Init(configJson)   — 解析 JSON → PipelineBuilder 创建 DAG → 初始化 EnginePool
//   2. Execute/Async      — 同步/异步推理，线程安全（mutex 保护）
//   3. Shutdown()         — 停止 Pipeline → 释放 EnginePool
//
// EnginePool 管理：
//   所有模型后端（TensorRT/ONNX/LibTorch）通过 EnginePool 池化复用。
//   PipelineBuilder 在构建时从 EnginePool 分配后端，避免重复加载模型。
//   池满策略：超出 maxEngines 的释放直接丢弃（非 LRU，简单实现）。
#include "engine/ai_engine.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"

namespace aicore {

// 获取单例实例（线程安全的局部静态变量）
AiEngine& AiEngine::GetInstance() {
    static AiEngine instance;
    return instance;
}

// 析构时自动关闭引擎
AiEngine::~AiEngine() {
    Shutdown();
}

// 初始化：解析 JSON 配置 → 构建 Pipeline → 准备引擎池
Status AiEngine::Init(const std::string& configJson) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine already initialized, call Shutdown first"};
    }

    enginePool_ = std::make_shared<EnginePool>();

    // 解析 JSON 配置
    ConfigParser parser;
    PipelineConfig pipelineConfig;
    auto parseStatus = parser.Parse(configJson, pipelineConfig);
    if (!parseStatus) {
        enginePool_.reset();
        return parseStatus;
    }

    // 根据配置构建 Pipeline DAG
    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    auto s = builder.Build(pipelineConfig, pipeline, enginePool_);
    if (!s) {
        enginePool_.reset();
        return s;
    }

    pipeline_ = std::move(pipeline);
    return Status{};
}

// 同步执行推理（线程安全）
Status AiEngine::Execute(const Frame& input, Result& output) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine not initialized"};
    }

    return pipeline_->Execute(input, output);
}

// 异步执行推理（线程安全）
Status AiEngine::ExecuteAsync(const Frame& input,
                               std::function<void(const Result&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine not initialized"};
    }

    return pipeline_->ExecuteAsync(input, std::move(callback));
}

// 关闭引擎：停止流水线 + 释放引擎池
void AiEngine::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pipeline_) {
        pipeline_->Stop();
        pipeline_.reset();
    }
    enginePool_.reset();
}

// 查询管道状态（未初始化时返回 Stopped）
PipelineState AiEngine::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return PipelineState::kStopped;
    }
    return pipeline_->GetState();
}

} // namespace aicore
