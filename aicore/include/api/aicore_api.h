// AI 引擎 C 语言 API 接口
// 提供 extern "C" 风格的动态库导出，供 C/C++ 及跨语言调用
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// 导出/导入宏定义
// AICORE_EXPORTS 在构建 DLL 时定义，导出符号；外部使用者则导入符号
#ifdef AICORE_EXPORTS
#define AICORE_C_API __declspec(dllexport)
#else
#define AICORE_C_API __declspec(dllimport)
#endif

// 不透明句柄类型，隐藏内部实现细节
typedef void* AICorePipeline;   // pipeline 句柄（指向 IPipeline*）
typedef void* AICoreResult;     // 推理结果句柄（指向 Result*）

/**
 * 获取引擎版本号
 * @return 版本字符串（如 "0.1.0"），静态分配，无需释放
 */
AICORE_C_API const char* aicore_version();

/**
 * 根据 JSON 配置创建 Pipeline 实例
 *
 * 内部流程：
 *   1. ConfigParser 解析 JSON → PipelineConfig
 *   2. PipelineBuilder 按配置构建 DAG（含 EnginePool 分配）
 *   3. 返回 IPipeline* 的不透明句柄
 *
 * @param configJson  JSON 配置字符串（见 README 格式说明）
 * @param errorOut    失败时输出错误信息，无需释放
 * @return 非空句柄成功，nullptr 失败
 */
AICORE_C_API AICorePipeline aicore_pipeline_create(
    const char* configJson, const char** errorOut);

/**
 * 同步执行 Pipeline 推理
 *
 * @param pipeline  管线句柄
 * @param imageData 图像像素数据（行优先，BGR/RGB/灰度）
 * @param width     图像宽度（px）
 * @param height    图像高度（px）
 * @param channels  通道数（1/3/4）
 * @param resultOut 输出结果句柄，使用后需 aicore_result_free
 * @param errorOut  失败时输出错误信息
 * @return 0 成功，-1 失败
 */
AICORE_C_API int aicore_pipeline_execute(
    AICorePipeline pipeline,
    const unsigned char* imageData,
    int width, int height, int channels,
    AICoreResult* resultOut,
    const char** errorOut);

/**
 * 序列化推理结果为 JSON 字符串
 *
 * 输出格式：
 * {
 *   "timestamp": uint64,
 *   "latency_ms": double,
 *   "status": int,
 *   "detections": [{
 *     "node_id": string,
 *     "label": string,
 *     "confidence": float,
 *     "bbox": {"x":float, "y":float, "w":float, "h":float},
 *     "measurements": {key: value, ...},
 *     "anomaly_score": float
 *   }]
 * }
 *
 * @param result 结果句柄
 * @return JSON 字符串（内部静态缓存，线程不安全）
 */
AICORE_C_API const char* aicore_result_to_json(AICoreResult result);

/**
 * 释放推理结果句柄
 * @param result 待释放的句柄（可传入 nullptr）
 */
AICORE_C_API void aicore_result_free(AICoreResult result);

/**
 * 销毁 Pipeline 实例，释放所有后端和处理器资源
 * @param pipeline 待销毁的句柄（可传入 nullptr）
 */
AICORE_C_API void aicore_pipeline_destroy(AICorePipeline pipeline);

/**
 * 获取 Pipeline 当前运行状态
 * @return PipelineState 枚举值，pipeline 为空时返回 -1
 */
AICORE_C_API int aicore_pipeline_get_state(AICorePipeline pipeline);

// ──────────────────────────────────────────
// 便捷单例接口（隐藏 handle 管理）
// 适用于只需要单条推理管线的场景，无需手动管理 Pipeline 句柄
// ──────────────────────────────────────────

/**
 * 初始化引擎单例（Meyer's Singleton）
 *
 * 内部等价于：
 *   aicore_pipeline_create(configJson, errorOut) 创建内部管线
 *   引擎池 EnginePool 准备就绪
 *
 * @param configJson pipeline 配置的 JSON 字符串
 * @param errorOut   失败时输出错误信息，无需释放
 * @return 0 成功，非 0 失败
 */
AICORE_C_API int aicore_engine_init(const char* configJson,
                                     const char** errorOut);

/**
 * 使用引擎单例同步执行推理
 *
 * 内部等价于：
 *   包装图像数据为 Frame → AiEngine::Execute → 包装 Result
 *
 * @param imageData 图像原始像素数据（行优先连续内存）
 * @param width     图像宽度（像素）
 * @param height    图像高度（像素）
 * @param channels  图像通道数（1=灰度，3=BGR，4=BGRA）
 * @param resultOut 输出结果句柄，使用后需 aicore_result_free 释放
 * @param errorOut  失败时输出错误信息，无需释放
 * @return 0 成功，非 0 失败
 */
AICORE_C_API int aicore_engine_execute(const unsigned char* imageData,
                                        int width, int height, int channels,
                                        AICoreResult* resultOut,
                                        const char** errorOut);

/**
 * 销毁引擎单例，释放所有资源
 *
 * 调用 AiEngine::Shutdown()，依次：
 *   停止 Pipeline → 释放 EnginePool → 清理所有后端实例
 */
AICORE_C_API void aicore_engine_shutdown();

// ──────────────────────────────────────────
// 异常热力图 API
// PatchCore 异常检测结果的热力图数据访问
// ──────────────────────────────────────────

/**
 * 获取检测结果中指定索引的异常热力图数据
 *
 * 热力图数据为 CV_32F 格式的浮点矩阵（行优先），
 * 每个像素值表示该位置的异常得分（值越高异常概率越大）。
 * 调用者使用完后必须通过 aicore_result_free_anomaly_map 释放。
 *
 * @param result   推理结果句柄
 * @param detIndex 检测结果索引（0-based）
 * @param outData  输出浮点数据指针，行优先存储
 * @param outW     输出热力图宽度
 * @param outH     输出热力图高度
 * @return 0 成功，-1 失败（无热力图或索引越界）
 */
AICORE_C_API int aicore_result_get_anomaly_map(AICoreResult result, int detIndex,
                                                 float** outData, int* outW, int* outH);

/**
 * 释放由 aicore_result_get_anomaly_map 返回的热力图数据
 * @param data 待释放的热力图数据指针（malloc 分配，free 释放）
 */
AICORE_C_API void aicore_result_free_anomaly_map(float* data);

#ifdef __cplusplus
}
#endif
