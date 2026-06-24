#include "patchcore/anomaly_evaluator.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace aicore {

void AnomalyEvaluator::AddSample(int groundTruth, float anomalyScore) {
    samples_.emplace_back(groundTruth, anomalyScore);
}

EvalResult AnomalyEvaluator::EvaluateAt(
    const std::vector<std::pair<int, float>>& samples, float threshold) {
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (auto& [label, score] : samples) {
        bool pred = (score >= threshold);
        if (label == 1 && pred) tp++;
        else if (label == 1 && !pred) fn++;
        else if (label == 0 && pred) fp++;
        else tn++;
    }
    EvalResult r;
    r.tp = tp; r.fp = fp; r.tn = tn; r.fn = fn;
    r.bestThreshold = threshold;
    double total = tp + fp + tn + fn;
    if (total > 0) r.accuracy = (tp + tn) / total;
    if (tp + fp > 0) r.bestPrecision = static_cast<double>(tp) / (tp + fp);
    if (tp + fn > 0) r.bestRecall = static_cast<double>(tp) / (tp + fn);
    if (r.bestPrecision + r.bestRecall > 0) {
        r.bestF1 = 2.0 * r.bestPrecision * r.bestRecall / (r.bestPrecision + r.bestRecall);
    }
    return r;
}

EvalResult AnomalyEvaluator::FindBestThreshold(
    const std::vector<std::pair<int, float>>& samples, int numSteps) {
    if (samples.empty()) return {};

    float minScore = 1e10f, maxScore = -1e10f;
    for (auto& [l, s] : samples) {
        if (s < minScore) minScore = s;
        if (s > maxScore) maxScore = s;
    }

    EvalResult best;
    float range = maxScore - minScore;
    for (int i = 0; i <= numSteps; i++) {
        float th = minScore + range * i / numSteps;
        auto r = EvaluateAt(samples, th);
        if (r.bestF1 > best.bestF1) {
            best = r;
        }
    }
    return best;
}

EvalResult AnomalyEvaluator::Evaluate() const {
    if (samples_.empty()) return {};

    auto sorted = samples_;
    std::sort(sorted.begin(), sorted.end(),
        [](auto& a, auto& b) { return a.second > b.second; });

    int posTotal = 0, negTotal = 0;
    for (auto& [label, score] : sorted) {
        if (label == 1) posTotal++;
        else negTotal++;
    }
    if (posTotal == 0 || negTotal == 0) return {};

    double tpr = 0, fpr = 0;
    double prevTpr = 0, prevFpr = 0;
    double auc = 0;
    int tp = 0, fp = 0;

    for (auto& [label, score] : sorted) {
        if (label == 1) tp++;
        else fp++;

        tpr = static_cast<double>(tp) / posTotal;
        fpr = static_cast<double>(fp) / negTotal;

        auc += (tpr + prevTpr) * (fpr - prevFpr) * 0.5;
        prevTpr = tpr;
        prevFpr = fpr;
    }

    // 最佳阈值通过扫描找
    auto best = FindBestThreshold(samples_, 1000);

    EvalResult r = best;
    r.auc = std::abs(auc);
    return r;
}

} // namespace aicore
