#include "trainer/training/training_scheduler.h"

namespace aicore {

TrainingScheduler::TrainingScheduler() {}

Status TrainingScheduler::AddTask(const TrainTask& task) {
    tasks_.push_back(task);
    return Status{};
}

Status TrainingScheduler::RunAll() {
    for (auto& task : tasks_) {
        (void)task;
    }
    tasks_.clear();
    return Status{};
}

Status TrainingScheduler::RunNext() {
    if (tasks_.empty())
        return Status{StatusCode::ErrorInvalidInput, "no pending tasks"};
    tasks_.erase(tasks_.begin());
    return Status{};
}

size_t TrainingScheduler::PendingCount() const { return tasks_.size(); }
void TrainingScheduler::Clear() { tasks_.clear(); }

} // namespace aicore
