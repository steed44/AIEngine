#pragma once
#include "core/types.h"
#include <string>

namespace aicore {

class CheckpointManager {
public:
    CheckpointManager(const std::string& saveDir);
    Status Save(int epoch, float metric, const std::string& modelData);
    Status Load(int epoch, std::string& modelData);
    Status GetBest(std::string& modelData);
    Status Cleanup(int keepBest = 3);

private:
    std::string saveDir_;
};

} // namespace aicore
