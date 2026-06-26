#include "patchcore/anomaly_evaluator.h"
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================
// anomaly_evaluator.cpp — PatchCore 异常得分评估器
// 功能：对 anomaly score 做二元分类评估，计算混淆矩阵、
//       F1 分数、准确率、AUC-ROC 等指标。
//
// 评估算法：
//   PatchCore 阈值扫描评估：
//   1. 对验证集每张图，用 memory bank NN 搜索得 anomaly score
//   2. 在 score 范围 [min, max] 内线性扫描所有可能的阈值
//   3. 对每个阈值计算混淆矩阵 → precision/recall/F1
//   4. 选 F1 最大的阈值作为最佳判定阈值
//   5. 用梯形法积分计算 AUC-ROC
//
// 混淆矩阵（Confusion Matrix）：
//             预测异常    预测正常
//   真实异常     TP        FN
//   真实正常     FP        TN
//
// 派生指标：
//   Accuracy  = (TP + TN) / Total
//   Precision = TP / (TP + FP)  — 检出异常中真正异常的比例
//   Recall    = TP / (TP + FN)  — 真实异常被检出的比例
//   F1        = 2PR / (P + R)   — precision 与 recall 的调和平均
//   AUC-ROC   = 梯形法积分 — 随机猜 0.5，完美 1.0
// ============================================================

namespace aicore {

// 添加一个样本到评估队列
// @param groundTruth   真实标签：1=异常（正样本），0=正常（负样本）
// @param anomalyScore  模型输出的异常得分（越高表示越可能异常）
void AnomalyEvaluator::AddSample(int groundTruth, float anomalyScore) {
    samples_.emplace_back(groundTruth, anomalyScore);
}

// 在指定阈值下计算混淆矩阵与派生指标
// 遍历所有样本，将 anomaly score >= threshold 的判为异常（正类），
// 否则判为正常（负类）。与真实标签对比得到 TP/FP/FN/TN 四个计数，
// 然后计算 accuracy/precision/recall/F1。
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

// 在 score 范围内线性扫描找最优 F1 对应的阈值
// PatchCore 的 anomaly score 范围因图像内容而异，固定阈值（如 0.5）
// 可能不适用于所有场景。通过在验证集上扫描找到最大化 F1 的阈值：
//   1. 找样本中 score 的最小值和最大值
//   2. 在 [min, max] 区间均分 numSteps 份
//   3. 对每个阈值调用 EvaluateAt 算 F1
//   4. 保留 F1 最大的阈值
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

    // ---- AUC-ROC 计算（梯形法） ----
    // 原理：按 anomaly score 降序遍历样本。每个点对应 ROC 曲线上
    // 一个 (FPR, TPR) 坐标。相邻点用梯形法算面积并累加。
    //
    // 梯形法公式：
    //   AUC = Σ (TPR_i + TPR_{i-1}) × (FPR_i - FPR_{i-1}) / 2
    //
    // ROC 曲线理解：
    //   x 轴 = FPR = FP / 负样本总数（假阳性率）
    //   y 轴 = TPR = TP / 正样本总数（真阳性率/召回率）
    //   AUC ∈ [0, 1]，0.5 = 随机猜测，1.0 = 完美分类器
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

    // 1000 步扫描找最大化 F1 的最佳阈值
    // 步数选择：过少精度不够（尤其在 score 分布密集区域），
    // 过多则收益递减。1000 步对大多数场景足够。
    auto best = FindBestThreshold(samples_, 1000);

    EvalResult r = best;
    r.auc = std::abs(auc);
    return r;
}

} // namespace aicore
