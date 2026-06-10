#pragma once
#include "core/processor.h"
#include "core/pipeline.h"
#include "config/config_parser.h"
#include "engine/engine_pool.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace aicore {

class PipelineBuilder {
public:
    Status Build(const PipelineConfig& config,
                 std::unique_ptr<IPipeline>& pipeline,
                 std::shared_ptr<EnginePool> pool = nullptr);
};

} // namespace aicore
