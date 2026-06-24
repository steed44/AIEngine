#include "patchcore/scheduler.h"
#include <cuda_runtime.h>

namespace aicore {

Scheduler& Scheduler::Instance() {
    static Scheduler instance;
    return instance;
}

Scheduler::Scheduler() {
    ProbeBothFitOnGPU();
}

void Scheduler::SetPriority(PriorityMode mode) {
    priorityMode_.store(mode);
    if (mode == PriorityMode::kBalanced) {
        RecheckGPU();
    }
}

bool Scheduler::InferenceUseGPU() const {
    // 无 GPU 设备时所有模式返回 false
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) return false;
    auto mode = priorityMode_.load();
    return mode != PriorityMode::kTraining;
}

bool Scheduler::TrainingUseGPU() const {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) return false;
    auto mode = priorityMode_.load();
    if (mode == PriorityMode::kTraining) return true;
    if (mode == PriorityMode::kInference) return false;
    // kBalanced: 双 GPU 显存够则训练也用 GPU
    return balancedResult_.load();
}

void Scheduler::RecheckGPU() {
    if (priorityMode_.load() != PriorityMode::kBalanced) return;
    ProbeBothFitOnGPU();
}

void Scheduler::SetGPUReservation(int inferMB, int trainMB, int headroomMB) {
    inferMB_.store(inferMB);
    trainMB_.store(trainMB);
    headroomMB_.store(headroomMB);
}

void Scheduler::ProbeBothFitOnGPU() {
    size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) {
        balancedResult_.store(false);
        return;
    }
    size_t required = (size_t)(inferMB_.load() + trainMB_.load() + headroomMB_.load()) * 1024 * 1024;
    balancedResult_.store(freeBytes >= required);
}

} // namespace aicore