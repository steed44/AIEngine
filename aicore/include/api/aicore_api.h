#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef AICORE_EXPORTS
#define AICORE_C_API __declspec(dllexport)
#else
#define AICORE_C_API __declspec(dllimport)
#endif

typedef void* AICorePipeline;
typedef void* AICoreResult;

AICORE_C_API const char* aicore_version();

AICORE_C_API AICorePipeline aicore_pipeline_create(const char* configJson,
                                                     const char** errorOut);

AICORE_C_API int aicore_pipeline_execute(AICorePipeline pipeline,
                                          const unsigned char* imageData,
                                          int width, int height, int channels,
                                          AICoreResult* resultOut,
                                          const char** errorOut);

AICORE_C_API const char* aicore_result_to_json(AICoreResult result);

AICORE_C_API void aicore_result_free(AICoreResult result);

AICORE_C_API void aicore_pipeline_destroy(AICorePipeline pipeline);

AICORE_C_API int aicore_pipeline_get_state(AICorePipeline pipeline);

#ifdef __cplusplus
}
#endif
