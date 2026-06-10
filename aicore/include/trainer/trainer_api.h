#pragma once
#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_TRAINER_API const char* aicore_trainer_version();
AICORE_TRAINER_API int aicore_train_run(const char* configJson, const char** errorOut);
AICORE_TRAINER_API int aicore_train_schedule(const char* tasksJson, const char** errorOut);

#ifdef __cplusplus
}
#endif
