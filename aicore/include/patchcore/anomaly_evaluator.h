#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <utility>

namespace aicore {

struct EvalResult {
    double auc = 0.0;
    double bestThreshold = 0.5;
    double bestF1 = 0.0;
    double bestPrecision = 0.0;
    double bestRecall = 0.0;
    double accuracy = 0.0;
    int tp = 0, fp = 0, tn = 0, fn = 0;
};

class AnomalyEvaluator {
public:
    // 添加一个样本: label=1 异常, label=0 正常
    void AddSample(int groundTruth, float anomalyScore);

    // 按当前所有样本计算评估指标
    EvalResult Evaluate() const;

    // 在指定 threshold 下计算混淆矩阵和 F1
    static EvalResult EvaluateAt(const std::vector<std::pair<int, float>>& samples, float threshold);

    // 在多个 threshold 上扫描, 找最优 F1
    static EvalResult FindBestThreshold(const std::vector<std::pair<int, float>>& samples, int numSteps = 1000);

    // 清空
    void Clear() { samples_.clear(); }

private:
    std::vector<std::pair<int, float>> samples_;
};

} // namespace aicore
