#include "optimizer/optimizer_api.h"
#include "optimizer/model_optimizer.h"
#include <string>
#include <cstring>

static std::string gLastError;

const char* aicore_optimizer_version() {
    return "0.1.0";
}

int aicore_optimize(const char* configJson, const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return -1;
    }
    aicore::ModelOptimizer optimizer;
    auto s = optimizer.Optimize(configJson);
    if (!s) {
        gLastError = s.message;
        if (errorOut) *errorOut = gLastError.c_str();
        return -1;
    }
    return 0;
}
