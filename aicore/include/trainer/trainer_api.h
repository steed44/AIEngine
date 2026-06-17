// 训练模块的 C 风格 API 接口
// 提供版本查询、启动训练、任务调度等核心功能
// 通过 extern "C" 导出，支持跨语言调用
#pragma once
#include "core/types.h"

// 导出 C 风格的 API 函数，供 Python/C# 等语言调用
#ifdef __cplusplus
extern "C" {
#endif

// 获取训练模块版本号
// @return 版本字符串，如 "1.0.0"
AICORE_TRAINER_API const char* aicore_trainer_version();

// 运行单次训练任务
// @param configJson  JSON 格式的训练配置字符串
// @param errorOut    返回错误信息（当返回值非 0 时有效）
// @return 0 表示成功，非 0 表示失败
AICORE_TRAINER_API int aicore_train_run(const char* configJson, const char** errorOut);

// 执行训练任务调度（批量处理多个训练任务）
// @param tasksJson  JSON 格式的任务列表字符串
// @param errorOut   返回错误信息（当返回值非 0 时有效）
// @return 0 表示成功，非 0 表示失败
AICORE_TRAINER_API int aicore_train_schedule(const char* tasksJson, const char** errorOut);

#ifdef __cplusplus
}
#endif
