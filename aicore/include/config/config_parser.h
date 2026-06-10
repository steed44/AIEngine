#pragma once
#include "core/types.h"
#include "core/model_backend.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace aicore {

struct ProcessorConfig {
    std::string id;
    std::string type;
    BackendType backend = BackendType::kUnknown;
    std::string modelPath;
    int deviceId = 0;
    int batchSize = 1;
    std::unordered_map<std::string, std::string> params;
};

struct EdgeConfig {
    std::string from;
    std::string to;
};

struct PipelineConfig {
    std::string name;
    std::vector<ProcessorConfig> nodes;
    std::vector<EdgeConfig> edges;
    int maxConcurrency = 4;
    bool enableProfiling = false;
};

class ConfigParser {
public:
    Status Parse(const std::string& jsonStr, PipelineConfig& config);
    Status Serialize(const PipelineConfig& config, std::string& jsonStr);
    std::string GetLastError() const;

private:
    std::string lastError_;
};

} // namespace aicore
