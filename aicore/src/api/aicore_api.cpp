#include "api/aicore_api.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"
#include "pipeline/pipeline_impl.h"
#include <string>
#include <mutex>
#include <unordered_map>
#include <sstream>

namespace aicore {

static std::mutex gMutex;
static std::unordered_map<void*, std::string> gErrors;

static const char* StoreError(const std::string& err) {
    std::lock_guard<std::mutex> lock(gMutex);
    auto ptr = reinterpret_cast<void*>(const_cast<char*>(err.c_str()));
    static std::string lastError;
    lastError = err;
    return lastError.c_str();
}

} // namespace aicore

using namespace aicore;

const char* aicore_version() {
    return "0.1.0";
}

AICorePipeline aicore_pipeline_create(const char* configJson,
                                       const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return nullptr;
    }

    ConfigParser parser;
    PipelineConfig config;
    auto s = parser.Parse(configJson, config);
    if (!s) {
        if (errorOut) *errorOut = StoreError(parser.GetLastError());
        return nullptr;
    }

    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    s = builder.Build(config, pipeline);
    if (!s) {
        if (errorOut) *errorOut = StoreError(s.message);
        return nullptr;
    }

    return static_cast<AICorePipeline>(pipeline.release());
}

int aicore_pipeline_execute(AICorePipeline pipeline,
                             const unsigned char* imageData,
                             int width, int height, int channels,
                             AICoreResult* resultOut,
                             const char** errorOut) {
    if (!pipeline || !imageData || width <= 0 || height <= 0) {
        if (errorOut) *errorOut = "invalid parameters";
        return -1;
    }

    auto* pipe = static_cast<IPipeline*>(pipeline);

    cv::Mat img(height, width,
                channels == 1 ? CV_8UC1 :
                channels == 4 ? CV_8UC4 : CV_8UC3,
                const_cast<unsigned char*>(imageData));

    Result result;
    auto s = pipe->Execute(Frame(img), result);
    if (!s) {
        if (errorOut) *errorOut = StoreError(s.message);
        return -1;
    }

    *resultOut = static_cast<AICoreResult>(new Result(std::move(result)));
    return 0;
}

const char* aicore_result_to_json(AICoreResult result) {
    if (!result) return "{}";
    auto* r = static_cast<Result*>(result);
    std::ostringstream json;
    json << "{"
         << "\"timestamp\":" << r->timestamp << ","
         << "\"latency_ms\":" << r->totalLatencyMs << ","
         << "\"status\":" << static_cast<int>(r->status) << ","
         << "\"detections\":[";

    bool first = true;
    for (auto& d : r->detections) {
        if (!first) json << ",";
        first = false;
        json << "{"
             << "\"node_id\":\"" << d.nodeId << "\","
             << "\"label\":\"" << d.label << "\","
             << "\"confidence\":" << d.confidence << ","
             << "\"bbox\":{" << "\"x\":" << d.bbox.x
             << ",\"y\":" << d.bbox.y
             << ",\"w\":" << d.bbox.w
             << ",\"h\":" << d.bbox.h << "}";
        if (!d.measurements.empty()) {
            json << ",\"measurements\":{";
            bool mfirst = true;
            for (auto& [k, v] : d.measurements) {
                if (!mfirst) json << ",";
                mfirst = false;
                json << "\"" << k << "\":" << v;
            }
            json << "}";
            auto it = d.measurements.find("anomaly_score");
            if (it != d.measurements.end()) {
                json << ",\"anomaly_score\":" << it->second;
            }
        }
        json << "}";
    }
    json << "]}";
    return StoreError(json.str());
}

void aicore_result_free(AICoreResult result) {
    delete static_cast<Result*>(result);
}

void aicore_pipeline_destroy(AICorePipeline pipeline) {
    delete static_cast<IPipeline*>(pipeline);
}

int aicore_pipeline_get_state(AICorePipeline pipeline) {
    if (!pipeline) return -1;
    return static_cast<int>(static_cast<IPipeline*>(pipeline)->GetState());
}
