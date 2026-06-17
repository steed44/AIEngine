// 训练回调接口定义，提供训练过程中的事件钩子
// 用户可实现此接口来监控训练进度、记录日志、可视化等
#pragma once
#include "core/types.h"

// AI引擎训练模块命名空间
namespace aicore {

// 训练回调抽象接口
// 通过实现此接口，可在训练各阶段插入自定义逻辑（日志记录、可视化、动态调整超参数等）
// 典型使用：实现日志记录器、TensorBoard 可视化、学习率调整等
class ITrainCallback {
public:
    virtual ~ITrainCallback() = default;

    // 每个 epoch 开始前触发
    // @param epoch  当前 epoch 编号（从 1 开始）
    virtual void OnEpochBegin(int epoch) = 0;

    // 每个 epoch 结束后触发
    // @param epoch    结束的 epoch 编号
    // @param loss     当前 epoch 的平均损失
    // @param metric   当前 epoch 的评估指标（如准确率、mAP）
    virtual void OnEpochEnd(int epoch, float loss, float metric) = 0;

    // 每个 batch 训练结束后触发
    // @param batch  当前 batch 编号
    // @param loss   当前 batch 的损失值
    virtual void OnBatchEnd(int batch, float loss) = 0;

    // 整个训练过程结束时触发
    // @param bestMetric  训练过程中的最佳评估指标值
    virtual void OnTrainEnd(float bestMetric) = 0;
};

} // namespace aicore
