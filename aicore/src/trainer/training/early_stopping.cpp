// ============================================================================
// 文件：early_stopping.cpp
// 用途：早停机制实现，当验证指标连续多轮不改善时提前终止训练
// 功能：监控指标变化、判断是否触发早停、重置状态
// ============================================================================

#include "trainer/training/early_stopping.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数
// 参数 patience - 容忍的连续不改善轮次数（超过此值则触发早停）
// 参数 minDelta - 判断"改善"的最小指标变化阈值，变化量必须超过此值才算改善
EarlyStopping::EarlyStopping(int patience, float minDelta)
    : patience_(patience), minDelta_(minDelta) {}

// 根据当前指标判断是否应停止训练
// 若当前指标相比最佳指标提升超过 minDelta，则认为有改善并重置计数；
// 否则无改善计数 +1，超过 patience 时触发早停
// 参数 currentMetric - 当前 epoch 的验证指标（如 mAP、准确率等）
// 返回值            - true 表示应停止训练，false 表示继续
bool EarlyStopping::ShouldStop(float currentMetric) {
    if (currentMetric > bestMetric_ + minDelta_) {
        bestMetric_ = currentMetric;
        bestEpoch_++;
        noImprove_ = 0;
    } else {
        noImprove_++;
    }
    return noImprove_ > patience_;
}

// 重置早停状态（开始新一轮训练时调用）
void EarlyStopping::Reset() {
    bestEpoch_ = 0;
    bestMetric_ = -1e9f;
    noImprove_ = 0;
}

} // namespace aicore
