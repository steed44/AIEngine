// ============================================================
// 文件：optimizer_api.cpp
// 用途：实现 Optimizer 模块的 C 语言 API 接口，
//   作为 C/C++ 混合调用的入口层。
// ============================================================
#include "optimizer/optimizer_api.h"
#include "optimizer/model_optimizer.h"
#include <string>
#include <cstring>

// 全局静态变量：保存最后一次调用的错误信息
// 使用 static 限制在当前文件作用域，避免命名冲突
static std::string gLastError;

// 返回 Optimizer 模块的当前版本号
const char* aicore_optimizer_version() {
    return "0.1.0";
}

// 模型优化入口函数
// 参数校验后创建 ModelOptimizer 实例并执行优化流程，
// 优化失败时将错误信息通过 errorOut 指针返回给调用方。
int aicore_optimize(const char* configJson, const char** errorOut) {
    // 校验输入参数：configJson 不允许为空
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return -1;
    }
    // 创建模型优化器并执行优化
    aicore::ModelOptimizer optimizer;
    auto s = optimizer.Optimize(configJson);
    // 优化失败时记录错误信息并通过输出参数返回
    if (!s) {
        gLastError = s.message;
        if (errorOut) *errorOut = gLastError.c_str();
        return -1;
    }
    return 0;
}
