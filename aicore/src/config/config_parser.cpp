// ============================================================
// config_parser.cpp — JSON 管线配置解析/序列化
// 读取 JSON 格式的管线描述，解析为 PipelineConfig 结构体；
// 也支持将 PipelineConfig 反向序列化为 JSON 字符串
// ============================================================

#include "config/config_parser.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace aicore {

/**
 * 将后端字符串标识解析为 BackendType 枚举值
 * @param s 后端字符串（"tensorrt" / "onnxruntime" / "libtorch"）
 * @return 对应的 BackendType 枚举，不识别时返回 kUnknown
 */
static BackendType ParseBackend(const std::string& s) {
    if (s == "tensorrt") return BackendType::kTensorRT;
    if (s == "onnxruntime") return BackendType::kONNXRuntime;
    if (s == "libtorch") return BackendType::kLibTorch;
    return BackendType::kUnknown;
}

/**
 * 解析 JSON 格式的管线配置
 * 预期结构：
 *   { "pipeline": { "name": "...", "max_concurrency": N, "enable_profiling": bool,
 *       "nodes": [{ "id":"...", "type":"...", "model_path":"...", ... }],
 *       "edges": [{ "from":"...", "to":"..." }] } }
 * @param jsonStr JSON 字符串
 * @param config  输出参数，解析结果写入此 PipelineConfig
 * @return Status 成功返回空 Status，解析失败返回带错误码的 Status
 */
Status ConfigParser::Parse(const std::string& jsonStr, PipelineConfig& config) {
    try {
        auto j = json::parse(jsonStr);

        // 提取顶层管线配置
        auto& pipeline = j["pipeline"];
        config.name = pipeline.value("name", "unnamed");
        config.maxConcurrency = pipeline.value("max_concurrency", 4);
        config.enableProfiling = pipeline.value("enable_profiling", false);

        // 逐个解析处理节点
        for (auto& node : pipeline["nodes"]) {
            ProcessorConfig pc;
            pc.id = node["id"];
            pc.type = node["type"];
            pc.modelPath = node.value("model_path", "");
            pc.deviceId = node.value("device_id", 0);
            pc.batchSize = node.value("batch_size", 1);
            pc.backend = ParseBackend(node.value("backend", ""));

            // 解析节点自定义参数（字符串或任意 JSON 值）
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

        // 逐个解析节点间连接边
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

/**
 * 将 PipelineConfig 序列化为 JSON 字符串
 * 与 Parse 互为逆操作，输出格式化的 JSON（缩进 2 空格）
 * @param config  待序列化的管线配置
 * @param jsonStr 输出参数，序列化后的 JSON 字符串
 * @return Status 成功返回空 Status，序列化失败返回错误 Status
 */
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

/**
 * 获取最近一次解析/序列化操作的错误描述
 * @return 错误字符串，空串表示无错误
 */
std::string ConfigParser::GetLastError() const {
    return lastError_;
}

} // namespace aicore
