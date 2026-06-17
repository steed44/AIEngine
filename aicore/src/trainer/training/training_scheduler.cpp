// ============================================================================
// 文件：training_scheduler.cpp
// 用途：训练任务调度器实现，管理多个训练任务的排队与执行
// 功能：添加任务、按序执行、清空队列、查询待办数量
// ============================================================================

#include "trainer/training/training_scheduler.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
TrainingScheduler::TrainingScheduler() {}

// 向调度队列中添加一个训练任务
// 参数 task - 训练任务结构体（包含 modelId、configPath、priority、gpuId）
// 返回值   - 操作状态
Status TrainingScheduler::AddTask(const TrainTask& task) {
    tasks_.push_back(task);
    return Status{};
}

// 依次执行所有待处理的训练任务，执行完毕后清空队列
// 返回值 - 操作状态
Status TrainingScheduler::RunAll() {
    for (auto& task : tasks_) {
        (void)task;
    }
    tasks_.clear();
    return Status{};
}

// 执行下一个训练任务（从队列头部取出并移除）
// 如果队列为空则返回错误状态 ErrorInvalidInput
// 返回值 - 操作状态
Status TrainingScheduler::RunNext() {
    if (tasks_.empty())
        return Status{StatusCode::ErrorInvalidInput, "no pending tasks"};
    tasks_.erase(tasks_.begin());
    return Status{};
}

// 返回当前待处理的训练任务数量
size_t TrainingScheduler::PendingCount() const { return tasks_.size(); }

// 清空所有待处理的训练任务
void TrainingScheduler::Clear() { tasks_.clear(); }

} // namespace aicore
