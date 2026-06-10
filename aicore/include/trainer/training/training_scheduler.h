#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <memory>

namespace aicore {

struct TrainTask {
    std::string modelId;
    std::string configPath;
    int priority = 0;
    int gpuId = 0;
};

class TrainingScheduler {
public:
    TrainingScheduler();
    Status AddTask(const TrainTask& task);
    Status RunAll();
    Status RunNext();
    size_t PendingCount() const;
    void Clear();

private:
    std::vector<TrainTask> tasks_;
    int currentGpu_ = 0;
};

} // namespace aicore
