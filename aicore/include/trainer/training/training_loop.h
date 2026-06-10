#pragma once
#include "core/types.h"
#include "trainer/data/data_loader.h"
#include "trainer/data/augmentation.h"
#include "trainer/model/model_factory.h"
#include "trainer/callback.h"
#include <memory>
#include <vector>

namespace aicore {

struct TrainConfig {
    int epochs = 100;
    float learningRate = 0.001f;
    float weightDecay = 0.0001f;
    int batchSize = 16;
    int numClasses = 10;
    int inputSize = 224;
    std::string saveDir = "checkpoints";
    std::vector<int> gpuIds = {0};
};

class TrainingLoop {
public:
    TrainingLoop();
    Status Run(const TrainConfig& config,
               std::shared_ptr<IDataset> trainDataset,
               std::shared_ptr<IDataset> valDataset,
               AugmentationPipeline& trainAug);
    void AddCallback(std::shared_ptr<ITrainCallback> callback);
    Status Stop();

private:
    std::vector<std::shared_ptr<ITrainCallback>> callbacks_;
    bool running_ = false;
};

} // namespace aicore
