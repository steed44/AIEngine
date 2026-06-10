#include "trainer/training/checkpoint.h"
#include <fstream>

namespace aicore {

CheckpointManager::CheckpointManager(const std::string& saveDir)
    : saveDir_(saveDir) {}

Status CheckpointManager::Save(int epoch, float metric, const std::string& modelData) {
    (void)epoch; (void)metric; (void)modelData;
    return Status{};
}

Status CheckpointManager::Load(int epoch, std::string& modelData) {
    (void)epoch; (void)modelData;
    return Status{};
}

Status CheckpointManager::GetBest(std::string& modelData) {
    (void)modelData;
    return Status{};
}

Status CheckpointManager::Cleanup(int keepBest) {
    (void)keepBest;
    return Status{};
}

} // namespace aicore
