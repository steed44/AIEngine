// ============================================================================
// 文件：trainer_api.cpp
// 用途：AIEngine 训练模块的 C 语言 API 接口实现
// 功能：提供训练运行、任务调度等对外接口
// ============================================================================

#include "trainer/trainer_api.h"
#include "trainer/model/python_trainer.h"
#include "trainer/training/training_scheduler.h"
#include <string>
#include <nlohmann/json.hpp>

// 全局错误信息缓冲区，用于跨函数传递错误描述
static std::string gTrainerError;

// 返回当前训练模块的版本号
const char* aicore_trainer_version() {
    return "0.1.0";
}

// 对外接口：启动一次模型训练
// 参数 configJson - JSON 格式的训练配置字符串
// 参数 errorOut   - 输出参数，返回错误描述（可为 nullptr）
// 返回值：0 表示成功，-1 表示失败
int aicore_train_run(const char* configJson, const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return -1;
    }
    // 创建 Python 训练器实例并执行训练
    aicore::PythonTrainer trainer;
    auto s = trainer.Train(configJson);
    if (!s) {
        gTrainerError = s.message;
        if (errorOut) *errorOut = gTrainerError.c_str();
        return -1;
    }
    return 0;
}

// 对外接口：提交多个训练任务并批量执行调度
// 参数 tasksJson - JSON 数组，每个元素包含 model_id、config_path、priority、gpu_id 等字段
// 参数 errorOut  - 输出参数，返回错误描述（可为 nullptr）
// 返回值：0 表示成功，-1 表示失败
int aicore_train_schedule(const char* tasksJson, const char** errorOut) {
    if (!tasksJson) {
        if (errorOut) *errorOut = "null tasks";
        return -1;
    }
    aicore::TrainingScheduler scheduler;
    try {
        // 解析 JSON 任务列表并逐个添加到调度器
        auto j = nlohmann::json::parse(tasksJson);
        for (auto& task : j) {
            aicore::TrainTask t;
            t.modelId = task.value("model_id", "");
            t.configPath = task.value("config_path", "");
            t.priority = task.value("priority", 0);
            t.gpuId = task.value("gpu_id", 0);
            scheduler.AddTask(t);
        }
    } catch (...) {
        if (errorOut) *errorOut = "invalid tasks JSON";
        return -1;
    }
    // 按序执行所有任务
    auto s = scheduler.RunAll();
    if (!s) {
        gTrainerError = s.message;
        if (errorOut) *errorOut = gTrainerError.c_str();
        return -1;
    }
    return 0;
}
