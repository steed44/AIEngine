#include "engine/ai_engine.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"

namespace aicore {

AiEngine& AiEngine::GetInstance() {
    static AiEngine instance;
    return instance;
}

AiEngine::~AiEngine() {
    Shutdown();
}

Status AiEngine::Init(const std::string& configJson) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine already initialized, call Shutdown first"};
    }

    enginePool_ = std::make_shared<EnginePool>();

    ConfigParser parser;
    PipelineConfig pipelineConfig;
    auto parseStatus = parser.Parse(configJson, pipelineConfig);
    if (!parseStatus) {
        enginePool_.reset();
        return parseStatus;
    }

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

Status AiEngine::Execute(const Frame& input, Result& output) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine not initialized"};
    }

    return pipeline_->Execute(input, output);
}

Status AiEngine::ExecuteAsync(const Frame& input,
                               std::function<void(const Result&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return Status{StatusCode::ErrorInternal, "engine not initialized"};
    }

    return pipeline_->ExecuteAsync(input, std::move(callback));
}

void AiEngine::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pipeline_) {
        pipeline_->Stop();
        pipeline_.reset();
    }
    enginePool_.reset();
}

PipelineState AiEngine::GetState() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pipeline_) {
        return PipelineState::kStopped;
    }
    return pipeline_->GetState();
}

} // namespace aicore
