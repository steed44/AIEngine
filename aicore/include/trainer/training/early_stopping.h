// 早停机制，监控验证指标并在不再改善时停止训练
// 防止过拟合，节省训练时间
#pragma once
#include "core/types.h"

namespace aicore {

// 早停控制器类
// 当评估指标在连续 patience 个 epoch 内无显著改善时触发停止
class EarlyStopping {
public:
    // 构造早停控制器
    // @param patience  允许指标无改善的最大 epoch 数（超过则停止），默认 10
    // @param minDelta  判断指标是否改善的最小变化阈值，默认 0.001
    EarlyStopping(int patience = 10, float minDelta = 0.001f);

    // 检查是否应停止训练
    // @param currentMetric  当前 epoch 的评估指标值
    // @return true 表示应停止训练
    bool ShouldStop(float currentMetric);

    // 重置所有内部状态（用于新一轮训练）
    void Reset();

    // 获取最佳指标对应的 epoch 编号
    int GetBestEpoch() const { return bestEpoch_; }

    // 获取迄今为止的最佳指标值
    float GetBestMetric() const { return bestMetric_; }

private:
    int patience_;              // 容忍无改善的 epoch 上限
    float minDelta_;            // 视为"改善"的最小指标变化量
    int bestEpoch_ = 0;         // 最佳指标出现的 epoch
    float bestMetric_ = -1e9f;  // 迄今为止的最佳指标值
    int noImprove_ = 0;         // 连续无改善的 epoch 计数
};

} // namespace aicore
