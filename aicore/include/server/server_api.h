// 推理服务器 C API — 对外暴露模型加载/卸载/推理接口
//
// 所有函数使用 C 调用约定（extern "C"），便于 Python CFFI / C# P/Invoke 等跨语言绑定。
// 错误处理：返回 int（0=成功，-1=失败），错误详情通过 errorOut 参数返回。
#pragma once
#include "api/aicore_api.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_C_API int aicore_server_load(const char* modelName, const char* modelPath,
                                     const char* backend, int vramMB);
AICORE_C_API int aicore_server_unload(const char* modelName);
AICORE_C_API int aicore_server_infer(const char* modelName,
                                      const unsigned char* data,
                                      int w, int h, int c,
                                      AICoreResult* out, const char** err);
AICORE_C_API const char* aicore_server_list();
AICORE_C_API void aicore_server_shutdown();

#ifdef __cplusplus
}
#endif