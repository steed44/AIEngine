// 服务器 C API 实现 — 封装 InferenceServer 为 C 风格接口供外部调用
//
// 设计：薄包装层，将 C++ 异常和 Status 转换为 C 返回码 + errorOut 字符串。
//   aicore_server_load/infer → InferenceServer 的方法调用。
//   参数校验在入口处完成，空指针和非法参数提前返回 -1。
#include "server/server_api.h"
#include "server/inference_server.h"
#include "core/types.h"
#include <opencv2/core.hpp>
#include <sstream>

using namespace aicore;

// 加载模型到推理服务器
// 参数校验 → 委托 InferenceServer::LoadModel → 返回 0(成功)/-1(失败)
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

// 推理服务器同步推理入口
// 将原始像素数据包装为 cv::Mat → InferSync → 结果包装为 AICoreResult
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

// 列出服务器中所有已加载模型的 JSON 字符串
// 结果缓存在 server_list_cache 中，下次调用自动释放旧缓存
static char* server_list_cache = nullptr;

const char* aicore_server_list() {
    if (server_list_cache) {
        free(server_list_cache);
    }
    auto json = InferenceServer::Instance().ListModels();
    server_list_cache = strdup(json.c_str());
    return server_list_cache;
}

void aicore_server_shutdown() {
    InferenceServer::Instance().Shutdown();
}