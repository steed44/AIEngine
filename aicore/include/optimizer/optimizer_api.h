// ============================================================
// 文件：optimizer_api.h
// 用途：提供 Optimizer 模块的纯 C 语言 API 接口，
//   允许 C/C++ 混合项目调用模型优化功能。
// ============================================================
#pragma once
#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// 获取 Optimizer 模块的版本号字符串
AICORE_OPTIMIZER_API const char* aicore_optimizer_version();

// 执行模型优化流程（导出 ONNX -> 构建 TensorRT 引擎）
// 参数 configJson : JSON 格式的配置字符串，包含 model_path、onnx_path、engine_path、precision 等
// 参数 errorOut   : 输出参数，指向错误描述字符串的指针（失败时有效）
// 返回值          : 0 成功，-1 失败
AICORE_OPTIMIZER_API int aicore_optimize(const char* configJson, const char** errorOut);

#ifdef __cplusplus
}
#endif
