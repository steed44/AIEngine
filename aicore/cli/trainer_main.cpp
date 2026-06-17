// ============================================================
// 文件: cli/trainer_main.cpp
// 用途: YOLO 模型训练命令行工具入口
//   读取 JSON 配置文件 → 调用训练 API 执行训练全流程
// ============================================================

#include "trainer/trainer_api.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    // ---- 1. 检查命令行参数 ----
    if (argc < 2) {
        std::cerr << "Usage: AICoreTrainer <config.json>" << std::endl;
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

    // ---- 3. 调用 C API 执行训练 ----
    std::string json = ss.str();
    const char* err = nullptr;
    int ret = aicore_train_run(json.c_str(), &err);
    if (ret != 0) {
        std::cerr << "Training failed: " << (err ? err : "unknown") << std::endl;
        return 1;
    }
    std::cout << "Training completed successfully." << std::endl;
    return 0;
}
