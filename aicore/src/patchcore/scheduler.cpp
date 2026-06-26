#include "patchcore/scheduler.h"
#include <cuda_runtime.h>

// ============================================================
// scheduler.cpp — GPU 显存调度器
// 功能：全局单例，管理 PatchCore 推理与训练的 GPU 显存分配策略。
//       解决推理和训练并行时 GPU 显存竞争问题。
//
// 调度策略：
//   kInference — 推理优先：推理用 GPU，训练用 CPU
//   kTraining  — 训练优先：训练用 GPU，推理用 CPU
//   kBalanced  — 自动平衡：探测双 GPU 显存是否充足，
//                够则都跑 GPU，不够则推理优先（因为推理对实时性敏感）
//
// 使用场景：
//   现场环境中，GPU 需要同时服务于实时推理（on-device）和
//   模型更新训练（periodic retraining）。调度器确保推理
//   不受训练任务影响。
// ============================================================

namespace aicore {

Scheduler& Scheduler::Instance() {
    static Scheduler instance;
    return instance;
}

// 构造函数：自动探测 GPU 数量和显存容量
// cudaGetDeviceCount 返回可用 CUDA 设备数（包括 TCC/WDDM 模式）
// deviceCount < 0 表示无 CUDA 设备，所有 GPU 操作回退 CPU
Scheduler::Scheduler() {
    int count = -1;
    cudaGetDeviceCount(&count);
    deviceCount_.store(count < 0 ? 0 : count);
    ProbeBothFitOnGPU();
}

void Scheduler::SetPriority(PriorityMode mode) {
    priorityMode_.store(mode);
    if (mode == PriorityMode::kBalanced) {
        RecheckGPU();
    }
}

// 判断推理 backbone 是否使用 GPU
// 规则：只要不是 kTraining 模式（训练占满 GPU），推理就用 GPU。
// 推理对实时性要求高，优先保证 GPU 资源。
bool Scheduler::InferenceUseGPU() const {
    if (deviceCount_ == 0) return false;
    auto mode = priorityMode_.load();
    return mode != PriorityMode::kTraining;
}

// 判断训练 backbone 是否使用 GPU
// 规则：
//   kTraining  → 必须用 GPU（训练速度敏感）
//   kInference → 不用 GPU（推理独占）
//   kBalanced  → 探测显存是否够两者同时跑
bool Scheduler::TrainingUseGPU() const {
    if (deviceCount_ == 0) return false;
    auto mode = priorityMode_.load();
    if (mode == PriorityMode::kTraining) return true;
    if (mode == PriorityMode::kInference) return false;
    // kBalanced: 如果显存足够推理+训练+headroom，训练也用 GPU
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

// 选择 GPU 设备并设置当前 CUDA 上下文
// 策略：简单 round-robin 轮询，在多 GPU 系统中分散负载
// @return 选中的设备 ID（无 GPU 时返回 0）
int Scheduler::SelectDevice() {
    int count = deviceCount_.load();
    if (count <= 0) return 0;
    int dev = nextDevice_.fetch_add(1) % count;
    cudaSetDevice(dev);
    return dev;
}

// 探测当前显存是否够推理+训练同时跑
// 计算所需 = inferMB + trainMB + headroomMB
// headroom 为安全余量（默认 1GB），防止少量显存碎片导致分配失败
// @note 只对 kBalanced 模式生效，其他模式不需要探测
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