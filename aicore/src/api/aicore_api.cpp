// ============================================================
// aicore_api.cpp — AIEngine C 公共 API 实现
// 提供 C 语言接口供外部调用，包含管线创建/执行/销毁和结果序列化
//
// C API 包装设计：
//   1. 不透明句柄模式：C++ 对象指针（IPipeline / Result）通过 reinterpret_cast
//      转为 void* 句柄，外部仅通过不透明句柄操作。隐藏所有 C++ 实现细节。
//
//   2. 错误处理：C++ Status 中的错误码和信息通过 StoreError() 写入静态
//      std::string，返回 const char* 指针。保证指针在调用者处理期间有效。
//      不跨线程使用（C API 通常是单线程调用）。
//
//   3. 内存管理：aicore_pipeline_create 返回 new 创建的 IPipeline，
//      调用者负责通过 aicore_pipeline_destroy 销毁。
//      aicore_pipeline_execute 返回 new 创建的 Result，调用者通过
//      aicore_result_free 释放。
// ============================================================

#include "api/aicore_api.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"
#include "engine/ai_engine.h"
#include "pipeline/pipeline_impl.h"
#include <string>
#include <mutex>
#include <unordered_map>
#include <sstream>

namespace aicore {

/// 全局互斥锁，保护错误信息存储
static std::mutex gMutex;
/// 错误信息缓存表（当前未使用，保留扩展）
static std::unordered_map<void*, std::string> gErrors;

/**
 * 存储错误信息并返回 C 字符串指针
 * 使用静态 lastError 缓存最近一条错误，保证返回的指针在调用期内有效
 * @param err 源错误字符串
 * @return 指向缓存的 C 风格错误字符串的指针
 */
static const char* StoreError(const std::string& err) {
    std::lock_guard<std::mutex> lock(gMutex);
    auto ptr = reinterpret_cast<void*>(const_cast<char*>(err.c_str()));
    static std::string lastError;
    lastError = err;
    return lastError.c_str();
}

} // namespace aicore

using namespace aicore;

/**
 * 返回当前 AIEngine 核心库的版本号
 * 版本语义：MAJOR.MINOR.PATCH，遵循语义化版本规范
 * @return "0.1.0" 版本字符串
 */
const char* aicore_version() {
    return "0.1.0";
}

/**
 * 根据 JSON 配置字符串创建一条推理管线
 * 内部依次：解析 JSON → 构建管线拓扑 → 返回不透明句柄
 *
 * 内存管理：
 *   new 创建 IPipeline 对象，返回 void* 不透明句柄。
 *   调用方使用完后必须通过 aicore_pipeline_destroy 释放。
 *
 * @param configJson JSON 格式的管线配置字符串
 * @param errorOut   输出参数，失败时返回错误描述，可传入 nullptr
 * @return 成功返回管线句柄（AICorePipeline），失败返回 nullptr
 */
AICorePipeline aicore_pipeline_create(const char* configJson,
                                       const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return nullptr;
    }

    // Phase 1: 解析 JSON 配置
    ConfigParser parser;
    PipelineConfig config;
    auto s = parser.Parse(configJson, config);
    if (!s) {
        if (errorOut) *errorOut = StoreError(parser.GetLastError());
        return nullptr;
    }

    // Phase 2: PipelineBuilder 按配置创建 DAG 拓扑
    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    s = builder.Build(config, pipeline);
    if (!s) {
        if (errorOut) *errorOut = StoreError(s.message);
        return nullptr;
    }

    // 移交所有权给调用方，后续通过句柄操作
    return static_cast<AICorePipeline>(pipeline.release());
}

/**
 * 执行管线推理
 * 将输入图像数据包装为 cv::Mat，送入管线执行，结果通过 resultOut 返回
 *
 * 注意：imageData 的像素排列假设为行优先连续存储。
 * OpenCV 使用 BGR 通道顺序，若调用方提供 RGB 数据需要事先转换。
 *
 * @param pipeline  管线句柄（由 aicore_pipeline_create 创建）
 * @param imageData 原始图像像素数据指针（RGB/BGR/灰度）
 * @param width     图像宽度（像素）
 * @param height    图像高度（像素）
 * @param channels  图像通道数（1=灰度，3=RGB/BGR，4=RGBA）
 * @param resultOut 输出参数，推理结果句柄
 * @param errorOut  输出参数，失败时返回错误描述，可传入 nullptr
 * @return 0 成功，-1 失败
 */
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

/**
 * 将推理结果序列化为 JSON 字符串
 * 包含时间戳、延迟、检测框、置信度、测量值等信息
 * @param result 推理结果句柄
 * @return JSON 字符串指针（调用者无需释放，内部静态缓存）
 */
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

/**
 * 释放由 aicore_pipeline_execute 返回的结果句柄
 * @param result 要释放的结果句柄
 */
void aicore_result_free(AICoreResult result) {
    if (!result) return;
    delete static_cast<Result*>(result);
}

int aicore_result_get_anomaly_map(AICoreResult result, int detIndex,
                                    float** outData, int* outW, int* outH) {
    if (!result || !outData || !outW || !outH) return -1;
    auto* r = static_cast<Result*>(result);
    if (detIndex < 0 || detIndex >= (int)r->detections.size()) return -1;
    auto& nr = r->detections[detIndex];
    if (nr.anomalyMap.empty()) return -1;
    if (nr.anomalyMap.type() != CV_32F) return -1;

    *outW = nr.anomalyMap.cols;
    *outH = nr.anomalyMap.rows;
    size_t bytes = (*outW) * (*outH) * sizeof(float);
    float* data = (float*)malloc(bytes);
    if (!data) return -1;
    memcpy(data, nr.anomalyMap.ptr<float>(), bytes);
    *outData = data;
    return 0;
}

void aicore_result_free_anomaly_map(float* data) {
    free(data);
}

/**
 * 销毁管线对象，释放所有后端和处理器资源
 * @param pipeline 要销毁的管线句柄
 */
void aicore_pipeline_destroy(AICorePipeline pipeline) {
    if (!pipeline) return;
    delete static_cast<IPipeline*>(pipeline);
}

/**
 * 获取管线当前运行状态
 * @param pipeline 管线句柄
 * @return 管线状态枚举整数值，pipeline 为空时返回 -1
 */
int aicore_pipeline_get_state(AICorePipeline pipeline) {
    if (!pipeline) return -1;
    return static_cast<int>(static_cast<IPipeline*>(pipeline)->GetState());
}

// ──────────────────────────────────────────
// 便捷单例接口实现
// ──────────────────────────────────────────

int aicore_engine_init(const char* configJson, const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return -1;
    }

    auto& engine = AiEngine::GetInstance();
    auto s = engine.Init(configJson);
    if (!s) {
        if (errorOut) *errorOut = StoreError(s.message);
        return -1;
    }
    return 0;
}

int aicore_engine_execute(const unsigned char* imageData,
                           int width, int height, int channels,
                           AICoreResult* resultOut,
                           const char** errorOut) {
    if (!imageData || width <= 0 || height <= 0) {
        if (errorOut) *errorOut = "invalid parameters";
        return -1;
    }

    auto& engine = AiEngine::GetInstance();

    cv::Mat img(height, width,
                channels == 1 ? CV_8UC1 :
                channels == 4 ? CV_8UC4 : CV_8UC3,
                const_cast<unsigned char*>(imageData));

    Result result;
    auto s = engine.Execute(Frame(img), result);
    if (!s) {
        if (errorOut) *errorOut = StoreError(s.message);
        return -1;
    }

    *resultOut = static_cast<AICoreResult>(new Result(std::move(result)));
    return 0;
}

void aicore_engine_shutdown() {
    AiEngine::GetInstance().Shutdown();
}
