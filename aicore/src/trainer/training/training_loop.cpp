// ============================================================================
// 文件：training_loop.cpp
// 用途：训练主循环实现，控制整个训练过程的 epochs 迭代
// 功能：管理训练流程、回调通知、停止控制
// ============================================================================

#include "trainer/training/training_loop.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
TrainingLoop::TrainingLoop() {}

// 启动训练主循环：按指定 epoch 数迭代，并在每个 epoch 前后触发回调
// 参数 config      - 训练配置（epoch 数、学习率等）
// 参数 trainDataset - 训练数据集
// 参数 valDataset   - 验证数据集
// 参数 trainAug     - 训练数据增强流水线
// 返回值           - 操作状态
Status TrainingLoop::Run(const TrainConfig& config,
                          std::shared_ptr<IDataset> trainDataset,
                          std::shared_ptr<IDataset> valDataset,
                          AugmentationPipeline& trainAug) {
    (void)config; (void)trainDataset; (void)valDataset; (void)trainAug;
    running_ = true;
    // 逐 epoch 训练，每次迭代前后通知回调以记录日志或调整策略
    for (int epoch = 0; epoch < config.epochs && running_; ++epoch) {
        for (auto& cb : callbacks_) cb->OnEpochBegin(epoch);
        for (auto& cb : callbacks_) cb->OnEpochEnd(epoch, 0.0f, 0.0f);
    }
    for (auto& cb : callbacks_) cb->OnTrainEnd(0.0f);
    running_ = false;
    return Status{};
}

// 注册训练回调，用于在训练各阶段（epoch 开始/结束、训练结束）接收通知
// 参数 callback - 回调接口指针，如日志记录器、早停检查器、检查点管理器等
void TrainingLoop::AddCallback(std::shared_ptr<ITrainCallback> callback) {
    callbacks_.push_back(std::move(callback));
}

// 请求停止训练（设置 running_ = false，下一个 epoch 循环退出）
Status TrainingLoop::Stop() { running_ = false; return Status{}; }

} // namespace aicore
