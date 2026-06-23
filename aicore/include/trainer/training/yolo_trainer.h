#pragma once
#include "core/types.h"
#include "trainer/model/yolo_model.h"
#include "trainer/model/yolo_loss.h"
#include "trainer/data/yolo_data.h"
#include <torch/nn.h>
#include <torch/serialize.h>
#include <torch/optim.h>
#include <torch/optim/sgd.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/cuda.h>
#include <memory>
#include <functional>
#include <atomic>
#include <string>

namespace aicore {

struct YOLOTrainConfig {
    std::string trainImgDir;
    std::string trainLabelDir;
    std::string valImgDir;
    std::string valLabelDir;
    int imgSize = 640;
    int batchSize = 16;
    int epochs = 300;
    float lr = 0.01f;
    float momentum = 0.937f;
    float weightDecay = 0.0005f;
    std::string saveDir = "./yolo_weights";
    std::string pretrainedPath; // 空字符串 = 从头训练
    int numClasses = 80;
    int warmupEpochs = 3;
};

struct YOLOTrainProgress {
    int epoch = 0;
    int batch = 0;
    int totalBatches = 0;
    float loss = 0;
    float boxLoss = 0;
    float clsLoss = 0;
    float dflLossComponent = 0; // 避开 dflLoss 函数名
    float lr = 0;
    bool done = false;
    std::string errorMsg;
};

class AICORE_TRAINER_API YOLOTrainer {
public:
    explicit YOLOTrainer(const YOLOTrainConfig& config);
    ~YOLOTrainer() = default;

    bool Init();
    void Train();
    void Stop() { stop_ = true; }

    using ProgressCallback = std::function<void(const YOLOTrainProgress&)>;
    void SetProgressCallback(ProgressCallback cb) { progressCb_ = cb; }

    YOLOTrainProgress GetLastProgress() const { return lastProgress_; }

private:
    void trainEpoch(int epoch);
    void saveCheckpoint(int epoch);

    YOLOTrainConfig config_;
    std::unique_ptr<YOLOv8Model> model_;
    std::unique_ptr<YOLOLoss> lossFn_;
    std::unique_ptr<YOLODataLoader> trainLoader_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::optim::LRScheduler> scheduler_;
    std::atomic<bool> stop_{ false };
    ProgressCallback progressCb_;
    YOLOTrainProgress lastProgress_;
};

} // namespace aicore
