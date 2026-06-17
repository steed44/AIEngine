// 训练任务调度器，管理多个训练任务的排队与执行
// 支持按优先级调度和 GPU 资源分配
#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <memory>

namespace aicore {

// 训练任务描述结构体
// 定义单个训练任务的元信息，用于调度决策
struct TrainTask {
    std::string modelId;     // 模型标识（用于追踪和日志）
    std::string configPath;  // 训练配置文件路径
    int priority = 0;        // 任务优先级（数值越大优先级越高）
    int gpuId = 0;           // 指定使用的 GPU 设备 ID
};

// 训练调度器类
// 维护任务队列，根据优先级和 GPU 资源顺序执行训练任务
class TrainingScheduler {
public:
    TrainingScheduler();

    // 添加训练任务到调度队列
    // @param task  训练任务描述
    // @return 成功返回 Success
    Status AddTask(const TrainTask& task);

    // 按优先级顺序执行所有排队任务
    // @return 全部成功返回 Success
    Status RunAll();

    // 执行队列中的下一个任务
    // @return 成功返回 Success（无任务时返回特定状态码）
    Status RunNext();

    // 获取队列中待执行的任务数量
    // @return 待处理任务数
    size_t PendingCount() const;

    // 清空所有待处理任务
    void Clear();

private:
    std::vector<TrainTask> tasks_;  // 待处理的任务队列
    int currentGpu_ = 0;            // 当前分配的 GPU 设备（轮询使用）
};

} // namespace aicore
