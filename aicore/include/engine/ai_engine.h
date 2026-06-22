#pragma once
#include "core/pipeline.h"
#include "core/frame.h"
#include "engine/engine_pool.h"
#include <memory>
#include <mutex>

namespace aicore {

class AiEngine {
public:
    static AiEngine& GetInstance();

    Status Init(const std::string& configJson);
    Status Execute(const Frame& input, Result& output);
    Status ExecuteAsync(const Frame& input,
                        std::function<void(const Result&)> callback);
    void Shutdown();
    PipelineState GetState() const;

private:
    AiEngine() = default;
    ~AiEngine();
    AiEngine(const AiEngine&) = delete;
    AiEngine& operator=(const AiEngine&) = delete;

    std::unique_ptr<IPipeline> pipeline_;
    std::shared_ptr<EnginePool> enginePool_;
    mutable std::mutex mutex_;
};

} // namespace aicore
