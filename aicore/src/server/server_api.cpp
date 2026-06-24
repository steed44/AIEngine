#include "server/server_api.h"
#include "server/inference_server.h"
#include "core/types.h"
#include <opencv2/core.hpp>
#include <sstream>

using namespace aicore;

int aicore_server_load(const char* modelName, const char* modelPath,
                        const char* backend, int vramMB) {
    if (!modelName || !modelPath || !backend) return -1;
    auto& server = InferenceServer::Instance();
    auto s = server.LoadModel(modelName, modelPath, backend, (size_t)vramMB, 1);
    return s ? 0 : -1;
}

int aicore_server_unload(const char* modelName) {
    if (!modelName) return -1;
    auto& server = InferenceServer::Instance();
    if (!server.IsLoaded(modelName)) return -1;
    server.ListModels(); // 仅用于验证
    return 0;
}

int aicore_server_infer(const char* modelName,
                         const unsigned char* data,
                         int w, int h, int c,
                         AICoreResult* out, const char** err) {
    if (!modelName || !data || w <= 0 || h <= 0 || !out) {
        if (err) *err = "invalid parameters";
        return -1;
    }
    cv::Mat img(h, w,
                c == 1 ? CV_8UC1 : c == 4 ? CV_8UC4 : CV_8UC3,
                const_cast<unsigned char*>(data));

    auto& server = InferenceServer::Instance();
    std::vector<cv::Mat> outputs;
    auto s = server.InferSync({img}, outputs, modelName);
    if (!s) {
        if (err) *err = s.message.c_str();
        return -1;
    }

    // 包装为 Result
    auto result = std::make_unique<Result>();
    for (auto& outMat : outputs) {
        NodeResult nr;
        nr.nodeId = modelName;
        nr.measurements["output_rows"] = outMat.rows;
        nr.measurements["output_cols"] = outMat.cols;
        result->detections.push_back(std::move(nr));
    }
    *out = static_cast<AICoreResult>(result.release());
    return 0;
}

const char* aicore_server_list() {
    static std::string cached;
    cached = InferenceServer::Instance().ListModels();
    return cached.c_str();
}

void aicore_server_shutdown() {
    InferenceServer::Instance().Shutdown();
}