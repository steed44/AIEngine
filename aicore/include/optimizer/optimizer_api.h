#pragma once
#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_OPTIMIZER_API const char* aicore_optimizer_version();
AICORE_OPTIMIZER_API int aicore_optimize(const char* configJson, const char** errorOut);

#ifdef __cplusplus
}
#endif
