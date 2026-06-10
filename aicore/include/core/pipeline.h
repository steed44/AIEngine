#pragma once
#include "core/types.h"
#include "core/frame.h"
#include "core/processor.h"
#include <memory>
#include <string>
#include <vector>

namespace aicore {

enum class PipelineState { kCreated, kReady, kRunning, kStopped, kError };

class IPipeline {
public:
    virtual ~IPipeline() = default;

    virtual Status Build(const std::string& configJson) = 0;
    virtual Status Execute(const Frame& input, Result& output) = 0;
    virtual Status ExecuteAsync(const Frame& input,
                                std::function<void(const Result&)> callback) = 0;
    virtual Status WaitAll() = 0;
    virtual void Stop() = 0;
    virtual PipelineState GetState() const = 0;
    virtual std::string GetConfig() const = 0;
};

} // namespace aicore
