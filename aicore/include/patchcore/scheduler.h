#pragma once
#include <atomic>
#include "core/types.h"

namespace aicore {

// GPU 显存调度优先级模式
// 控制推理和训练 backbone 的 GPU/CPU 分配策略
enum class PriorityMode {
    kInference,  // 推理 GPU，训练 CPU
    kTraining,   // 训练 GPU，推理 CPU
    kBalanced    // 自动探测显存，尽量双 GPU，不够则推理优先
};

// Scheduler — GPU 显存调度器（全局单例）
// 解决 PatchCore 训练和推理并行时 GPU 显存竞争问题
class AICORE_API Scheduler {
public:
    static Scheduler& Instance();

    void SetPriority(PriorityMode mode);
    PriorityMode GetPriority() const { return priorityMode_.load(); }

    bool InferenceUseGPU() const;   // 推理 backbone 是否用 GPU
    bool TrainingUseGPU() const;    // 训练 backbone 是否用 GPU

    void RecheckGPU();              // kBalanced 模式下重新探测显存
    void SetGPUReservation(int inferMB, int trainMB, int headroomMB);

private:
    Scheduler();
    void ProbeBothFitOnGPU();

    std::atomic<PriorityMode> priorityMode_{PriorityMode::kBalanced};
    std::atomic<bool> balancedResult_{false};
    std::atomic<int> inferMB_{2048}, trainMB_{6144}, headroomMB_{1024};
    std::atomic<int> deviceCount_{-1};
};

} // namespace aicore