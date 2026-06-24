#pragma once
#include "api/aicore_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// 设置 GPU 调度优先级模式
// mode: "inference" / "training" / "balanced"
AICORE_C_API void aicore_scheduler_set_priority(const char* mode);

// 获取当前 GPU 调度优先级模式
// 返回 "inference" / "training" / "balanced"
AICORE_C_API const char* aicore_scheduler_get_priority();

// 手动触发 kBalanced 模式下的显存重新探测
AICORE_C_API void aicore_scheduler_recheck();

#ifdef __cplusplus
}
#endif