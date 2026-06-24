#include "api/scheduler_api.h"
#include "patchcore/scheduler.h"

using namespace aicore;

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

const char* aicore_scheduler_get_priority() {
    switch (Scheduler::Instance().GetPriority()) {
        case PriorityMode::kInference: return "inference";
        case PriorityMode::kTraining:  return "training";
        default:                       return "balanced";
    }
}

void aicore_scheduler_recheck() {
    Scheduler::Instance().RecheckGPU();
}