#include "trainer/trainer_api.h"
#include "trainer/model/python_trainer.h"
#include "trainer/training/training_scheduler.h"
#include <string>
#include <nlohmann/json.hpp>

static std::string gTrainerError;

const char* aicore_trainer_version() {
    return "0.1.0";
}

int aicore_train_run(const char* configJson, const char** errorOut) {
    if (!configJson) {
        if (errorOut) *errorOut = "null config";
        return -1;
    }
    aicore::PythonTrainer trainer;
    auto s = trainer.Train(configJson);
    if (!s) {
        gTrainerError = s.message;
        if (errorOut) *errorOut = gTrainerError.c_str();
        return -1;
    }
    return 0;
}

int aicore_train_schedule(const char* tasksJson, const char** errorOut) {
    if (!tasksJson) {
        if (errorOut) *errorOut = "null tasks";
        return -1;
    }
    aicore::TrainingScheduler scheduler;
    try {
        auto j = nlohmann::json::parse(tasksJson);
        for (auto& task : j) {
            aicore::TrainTask t;
            t.modelId = task.value("model_id", "");
            t.configPath = task.value("config_path", "");
            t.priority = task.value("priority", 0);
            t.gpuId = task.value("gpu_id", 0);
            scheduler.AddTask(t);
        }
    } catch (...) {
        if (errorOut) *errorOut = "invalid tasks JSON";
        return -1;
    }
    auto s = scheduler.RunAll();
    if (!s) {
        gTrainerError = s.message;
        if (errorOut) *errorOut = gTrainerError.c_str();
        return -1;
    }
    return 0;
}
