#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <utility>

namespace aicore {

// 评估结果结构体
// 包含在最佳阈值下的混淆矩阵和各种派生指标
struct EvalResult {
    double auc = 0.0;            // ROC 曲线下面积（梯形法）
    double bestThreshold = 0.5;   // 使 F1 最大的最佳阈值
    double bestF1 = 0.0;         // 最佳阈值下的 F1 分数
    double bestPrecision = 0.0;  // 最佳阈值下的精确率
    double bestRecall = 0.0;     // 最佳阈值下的召回率
    double accuracy = 0.0;       // 最佳阈值下的准确率
    int tp = 0, fp = 0, tn = 0, fn = 0;  // 混淆矩阵四值
};

class AnomalyEvaluator {
public:
    // 添加一个样本
    // label=1 异常（正样本），label=0 正常（负样本）
    void AddSample(int groundTruth, float anomalyScore);

    // 按当前累积的所有样本计算评估指标
    // 内部自动做降序排序 → AUC 梯形积分 → 阈值扫描找最佳 F1
    EvalResult Evaluate() const;

    // 在指定阈值下计算混淆矩阵、precision、recall、F1
    static EvalResult EvaluateAt(const std::vector<std::pair<int, float>>& samples, float threshold);

    // 在 score 范围上线性扫描 numSteps 个阈值，找最大化 F1 的阈值
    static EvalResult FindBestThreshold(const std::vector<std::pair<int, float>>& samples, int numSteps = 1000);

    // 清空所有样本
    void Clear() { samples_.clear(); }

private:
    std::vector<std::pair<int, float>> samples_;
};

} // namespace aicore
