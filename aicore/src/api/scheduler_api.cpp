// ============================================================
// scheduler_api.cpp — GPU 调度 C API 实现
// 功能：暴露 aicore_scheduler_set_priority/get_priority/recheck
//       三个 C 函数，供 Python CFFI / C# P/Invoke 等外部调用
// 设计：薄包装层，内部委托 Scheduler::Instance() 单例方法
// ============================================================

#include "api/scheduler_api.h"
#include "patchcore/scheduler.h"

using namespace aicore;

// 设置 GPU 调度优先级模式
// mode: "inference" → 优先推理（GPU 分配给推理）
//       "training"  → 优先训练（GPU 分配给训练）
//       "balanced"  → 平衡模式（交替分配，自动探测）
void aicore_scheduler_set_priority(const char* mode) {
    if (!mode) return;
    std::string m(mode);
    if (m == "inference")
        Scheduler::Instance().SetPriority(PriorityMode::kInference);
    else if (m == "training")
        Scheduler::Instance().SetPriority(PriorityMode::kTraining);
    else if (m == "balanced")
        Scheduler::Instance().SetPriority(PriorityMode::kBalanced);
}

// 获取当前 GPU 调度优先级模式
// 返回 "inference" / "training" / "balanced"
const char* aicore_scheduler_get_priority() {
    switch (Scheduler::Instance().GetPriority()) {
        case PriorityMode::kInference: return "inference";
        case PriorityMode::kTraining:  return "training";
        default:                       return "balanced";
    }
}

// 手动触发 kBalanced 模式下的显存重新探测
// 调用后 Scheduler 会重新扫描可用 GPU 显存，更新调度决策
void aicore_scheduler_recheck() {
    Scheduler::Instance().RecheckGPU();
}