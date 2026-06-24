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
typedef void* AICorePipeline;   // pipeline 句柄
typedef void* AICoreResult;     // 推理结果句柄

// 获取引擎版本号
// @return 版本号字符串（如 "1.0.0"），无需释放
AICORE_C_API const char* aicore_version();

// 根据 JSON 配置创建 pipeline 实例
// @param configJson  pipeline 配置的 JSON 字符串
// @param errorOut    传出错误信息（若创建失败），调用者无需释放
// @return pipeline 句柄，失败返回 NULL
AICORE_C_API AICorePipeline aicore_pipeline_create(const char* configJson,
                                                     const char** errorOut);

// 同步执行推理：传入图像数据，获取推理结果
// @param pipeline  pipeline 句柄
// @param imageData 图像原始像素数据（连续内存，RGB 或 BGR 顺序）
// @param width     图像宽度（像素）
// @param height    图像高度（像素）
// @param channels  图像通道数（通常为 3）
// @param resultOut 传出结果句柄，使用后需调用 aicore_result_free 释放
// @param errorOut  传出错误信息（若执行失败），调用者无需释放
// @return 0 表示成功，非 0 表示失败
AICORE_C_API int aicore_pipeline_execute(AICorePipeline pipeline,
                                          const unsigned char* imageData,
                                          int width, int height, int channels,
                                          AICoreResult* resultOut,
                                          const char** errorOut);

// 将推理结果序列化为 JSON 字符串
// @param result 结果句柄
// @return JSON 字符串，调用者无需释放
AICORE_C_API const char* aicore_result_to_json(AICoreResult result);

// 释放推理结果句柄
// @param result 结果句柄，释放后不可再使用
AICORE_C_API void aicore_result_free(AICoreResult result);

// 销毁 pipeline 实例，释放所有相关资源
// @param pipeline pipeline 句柄，销毁后不可再使用
AICORE_C_API void aicore_pipeline_destroy(AICorePipeline pipeline);

// 获取 pipeline 当前状态
// @param pipeline pipeline 句柄
// @return PipelineState 枚举的整数值
AICORE_C_API int aicore_pipeline_get_state(AICorePipeline pipeline);

// ──────────────────────────────────────────
// 便捷单例接口（隐藏 handle 管理）
// ──────────────────────────────────────────

// 初始化引擎单例，根据 JSON 配置创建 pipeline
// @param configJson pipeline 配置的 JSON 字符串
// @param errorOut   传出错误信息（若初始化失败），调用者无需释放
// @return 0 成功，非 0 失败
AICORE_C_API int aicore_engine_init(const char* configJson,
                                     const char** errorOut);

// 使用引擎单例同步执行推理
// @param imageData 图像原始像素数据（连续内存）
// @param width     图像宽度（像素）
// @param height    图像高度（像素）
// @param channels  图像通道数
// @param resultOut 传出结果句柄，使用后需调用 aicore_result_free 释放
// @param errorOut  传出错误信息（若执行失败），调用者无需释放
// @return 0 成功，非 0 失败
AICORE_C_API int aicore_engine_execute(const unsigned char* imageData,
                                        int width, int height, int channels,
                                        AICoreResult* resultOut,
                                        const char** errorOut);

// ──────────────────────────────────────────
// 异常热力图 API
// ──────────────────────────────────────────

// 获取检测结果中指定索引的异常热力图数据
// @param result      推理结果句柄
// @param detIndex    检测结果索引（0-based）
// @param outData     传出浮点像素数据指针（CV_32F，行优先），调用者需通过 aicore_result_free_anomaly_map 释放
// @param outW        传出热力图宽度
// @param outH        传出热力图高度
// @return 0 成功，-1 失败（无热力图或索引越界）
AICORE_C_API int aicore_result_get_anomaly_map(AICoreResult result, int detIndex,
                                                 float** outData, int* outW, int* outH);

// 释放由 aicore_result_get_anomaly_map 返回的热力图数据
AICORE_C_API void aicore_result_free_anomaly_map(float* data);

// 销毁引擎单例，释放所有资源
AICORE_C_API void aicore_engine_shutdown();

#ifdef __cplusplus
}
#endif
