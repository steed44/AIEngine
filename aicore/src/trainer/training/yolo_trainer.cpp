#include "trainer/training/yolo_trainer.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aicore {

namespace fs = std::filesystem;

YOLOTrainer::YOLOTrainer(const YOLOTrainConfig& config)
    : config_(config) {
}

bool YOLOTrainer::Init() {
    fs::create_directories(config_.saveDir);

    // 1. 创建数据集 + DataLoader
    auto trainDs = std::make_shared<YOLODataset>();
    if (config_.trainLabelDir.empty()) {
        if (!trainDs->Load(config_.trainImgDir, config_.trainImgDir))
            return false;
    } else {
        if (!trainDs->Load(config_.trainImgDir, config_.trainLabelDir))
            return false;
    }
    if (trainDs->Size() == 0) return false;

    config_.numClasses = trainDs->NumClasses();

    trainLoader_ = std::make_unique<YOLODataLoader>(
        trainDs, config_.batchSize, config_.imgSize, true);

    // 2. 创建模型
    model_ = std::make_unique<YOLOv8Model>(config_.numClasses);

    ModelConfig modelCfg;
    modelCfg.numClasses = config_.numClasses;
    auto buildStatus = model_->Build(modelCfg);
    if (!buildStatus) return false;

    if (!config_.pretrainedPath.empty()) {
        auto status = model_->Load(config_.pretrainedPath);
        if (!status) return false;
    } else {
        // 随机初始化 — 框架默认初始化已生效
    }

    model_->train(true);
    if (torch::cuda::is_available()) {
        model_->to(torch::kCUDA);
    }

    // 3. Loss
    YOLOLossConfig lossCfg;
    lossFn_ = std::make_unique<YOLOLoss>(lossCfg);

    // 4. 优化器
    std::vector<torch::optim::OptimizerParamGroup> paramGroups;
    // 分组: 权重衰减作用在 Conv/Linear 权重上, 不作用在 BN/偏置上
    auto params = model_->parameters();
    optimizer_ = std::make_unique<torch::optim::SGD>(
        params, torch::optim::SGDOptions(config_.lr)
            .momentum(config_.momentum)
            .weight_decay(config_.weightDecay));

    // 5. LR 调度
    scheduler_ = std::make_unique<torch::optim::StepLR>(*optimizer_, 100, 0.5);
    return true;
}

void YOLOTrainer::Train() {
    for (int epoch = 0; epoch < config_.epochs; epoch++) {
        if (stop_) break;
        trainEpoch(epoch);
        scheduler_->step();

        if ((epoch + 1) % 50 == 0 || epoch == config_.epochs - 1) {
            saveCheckpoint(epoch);
        }
    }
    saveCheckpoint(config_.epochs - 1);

    lastProgress_.done = true;
    if (progressCb_) progressCb_(lastProgress_);
}

void YOLOTrainer::trainEpoch(int epoch) {
    trainLoader_->Reset();
    int batchIdx = 0;
    int totalBatches = (int)trainLoader_->NumBatches();

    while (trainLoader_->HasNext()) {
        if (stop_) return;

        auto batch = trainLoader_->Next();
        auto images = batch.images.to(torch::kFloat32).div(255.0f);
        auto targets = batch.targets;
        if (torch::cuda::is_available()) {
            images = images.cuda();
            targets = targets.cuda();
        }

        // warmup: 线性增加 LR
        if (epoch < config_.warmupEpochs && batchIdx == 0) {
            float warmupFactor = (float)(epoch * totalBatches + batchIdx) /
                                (config_.warmupEpochs * totalBatches);
            if (warmupFactor < 0.01f) warmupFactor = 0.01f;
            for (auto& group : optimizer_->param_groups()) {
                static_cast<torch::optim::SGDOptions&>(group.options())
                    .lr(config_.lr * warmupFactor);
            }
        }

        optimizer_->zero_grad();

        YOLOLossOutput lossOut;
        std::vector<torch::Tensor> preds = model_->Forward(images);
        lossOut = (*lossFn_)(preds, targets);

        lossOut.totalLoss.backward();

        // 梯度裁剪
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 10.0);
        optimizer_->step();

        float currentLr = config_.lr;
        if (!optimizer_->param_groups().empty()) {
            currentLr = static_cast<torch::optim::SGDOptions&>(
                optimizer_->param_groups()[0].options()).lr();
        }

        lastProgress_.epoch = epoch;
        lastProgress_.batch = batchIdx;
        lastProgress_.totalBatches = totalBatches;
        if (lossOut.totalLoss.defined()) {
            lastProgress_.loss = lossOut.totalLoss.item<float>();
            lastProgress_.boxLoss = lossOut.boxLoss.item<float>();
            lastProgress_.clsLoss = lossOut.clsLoss.item<float>();
            lastProgress_.dflLossComponent = lossOut.dflLoss.item<float>();
        }
        lastProgress_.lr = currentLr;

        if (progressCb_) progressCb_(lastProgress_);
        batchIdx++;
    }
}

void YOLOTrainer::saveCheckpoint(int epoch) {
    std::stringstream ss;
    ss << config_.saveDir << "/yolov8_epoch_" << std::setw(4) << std::setfill('0')
       << (epoch + 1) << ".pt";
    auto path = ss.str();
    model_->Save(path);
}

} // namespace aicore
