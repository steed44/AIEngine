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