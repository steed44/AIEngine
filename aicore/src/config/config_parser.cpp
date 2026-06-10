#include "config/config_parser.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace aicore {

static BackendType ParseBackend(const std::string& s) {
    if (s == "tensorrt") return BackendType::kTensorRT;
    if (s == "onnxruntime") return BackendType::kONNXRuntime;
    if (s == "libtorch") return BackendType::kLibTorch;
    return BackendType::kUnknown;
}

Status ConfigParser::Parse(const std::string& jsonStr, PipelineConfig& config) {
    try {
        auto j = json::parse(jsonStr);

        auto& pipeline = j["pipeline"];
        config.name = pipeline.value("name", "unnamed");
        config.maxConcurrency = pipeline.value("max_concurrency", 4);
        config.enableProfiling = pipeline.value("enable_profiling", false);

        for (auto& node : pipeline["nodes"]) {
            ProcessorConfig pc;
            pc.id = node["id"];
            pc.type = node["type"];
            pc.modelPath = node.value("model_path", "");
            pc.deviceId = node.value("device_id", 0);
            pc.batchSize = node.value("batch_size", 1);
            pc.backend = ParseBackend(node.value("backend", ""));

            if (node.contains("params")) {
                for (auto& [key, val] : node["params"].items()) {
                    if (val.is_string())
                        pc.params[key] = val.get<std::string>();
                    else
                        pc.params[key] = val.dump();
                }
            }
            config.nodes.push_back(std::move(pc));
        }

        for (auto& edge : pipeline["edges"]) {
            EdgeConfig ec;
            ec.from = edge["from"];
            ec.to = edge["to"];
            config.edges.push_back(std::move(ec));
        }

        return Status{};
    }
    catch (const json::exception& e) {
        lastError_ = std::string("JSON parse error: ") + e.what();
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
}

Status ConfigParser::Serialize(const PipelineConfig& config, std::string& jsonStr) {
    try {
        json j;
        auto& pipeline = j["pipeline"];
        pipeline["name"] = config.name;
        pipeline["max_concurrency"] = config.maxConcurrency;
        pipeline["enable_profiling"] = config.enableProfiling;

        for (auto& node : config.nodes) {
            json jn;
            jn["id"] = node.id;
            jn["type"] = node.type;
            jn["model_path"] = node.modelPath;
            jn["device_id"] = node.deviceId;
            jn["batch_size"] = node.batchSize;
            jn["backend"] = [](BackendType bt) {
                switch (bt) {
                case BackendType::kTensorRT: return "tensorrt";
                case BackendType::kONNXRuntime: return "onnxruntime";
                case BackendType::kLibTorch: return "libtorch";
                default: return "";
                }
            }(node.backend);

            if (!node.params.empty()) {
                for (auto& [k, v] : node.params)
                    jn["params"][k] = v;
            }
            pipeline["nodes"].push_back(std::move(jn));
        }

        for (auto& edge : config.edges) {
            json je;
            je["from"] = edge.from;
            je["to"] = edge.to;
            pipeline["edges"].push_back(std::move(je));
        }

        jsonStr = j.dump(2);
        return Status{};
    }
    catch (const json::exception& e) {
        lastError_ = std::string("JSON serialize error: ") + e.what();
        return Status{StatusCode::ErrorConfigParse, lastError_};
    }
}

std::string ConfigParser::GetLastError() const {
    return lastError_;
}

} // namespace aicore
