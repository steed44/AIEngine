// ============================================================
// 文件: cli/model_optimizer_main.cpp
// 用途: 模型优化命令行工具入口
//   PyTorch → ONNX → TensorRT 引擎转换全链路
// ============================================================

#include "optimizer/model_optimizer.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    // ---- 1. 检查命令行参数 ----
    if (argc < 2) {
        std::cerr << "Usage: ModelOptimizer <config.json>" << std::endl;
        return 1;
    }

    // ---- 2. 读取 JSON 配置文件 ----
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Cannot open: " << argv[1] << std::endl;
        return 1;
    }
    std::stringstream ss;
    ss << file.rdbuf();

    // ---- 3. 执行模型优化 ----
    aicore::ModelOptimizer optimizer;
    auto s = optimizer.Optimize(ss.str());
    if (!s) {
        std::cerr << "Optimization failed: " << s.message << std::endl;
        return 1;
    }
    std::cout << "Optimization completed successfully." << std::endl;
    return 0;
}
