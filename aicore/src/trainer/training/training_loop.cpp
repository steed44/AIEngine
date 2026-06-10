#include "trainer/training/training_loop.h"

namespace aicore {

TrainingLoop::TrainingLoop() {}

Status TrainingLoop::Run(const TrainConfig& config,
                          std::shared_ptr<IDataset> trainDataset,
                          std::shared_ptr<IDataset> valDataset,
                          AugmentationPipeline& trainAug) {
    (void)config; (void)trainDataset; (void)valDataset; (void)trainAug;
    running_ = true;
    for (int epoch = 0; epoch < config.epochs && running_; ++epoch) {
        for (auto& cb : callbacks_) cb->OnEpochBegin(epoch);
        for (auto& cb : callbacks_) cb->OnEpochEnd(epoch, 0.0f, 0.0f);
    }
    for (auto& cb : callbacks_) cb->OnTrainEnd(0.0f);
    running_ = false;
    return Status{};
}

void TrainingLoop::AddCallback(std::shared_ptr<ITrainCallback> callback) {
    callbacks_.push_back(std::move(callback));
}

Status TrainingLoop::Stop() { running_ = false; return Status{}; }

} // namespace aicore
