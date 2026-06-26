// ============================================================
// memory_bank.cpp — MemoryBank 记忆库核心实现
// 功能：提供特征库的构建、序列化、最近邻检索和异常热力图生成
// 文件格式：魔数(4B) + 数量(4B) + 维度(4B) + N 个特征记录
// ============================================================
#include "patchcore/memory_bank.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace aicore {

// MemoryBank（记忆库）是 PatchCore 的核心数据结构：
// 它在训练阶段存储所有正常图像的特征向量（或其核心子集），
// 在推理阶段作为"正常性"的参考基准。
// 类比：记忆库 = 质检员记忆中的"合格品特征字典"
void MemoryBank::Build(const std::vector<PatchFeature>& features) {
    bank_ = features;
    if (!features.empty()) {
        featureDim_ = static_cast<int>(features[0].features.size());
    }
}

// 清空记忆库（释放所有特征，重置维度为 0）
// 通常在重新训练或节点销毁时调用
void MemoryBank::Clear() {
    bank_.clear();
    featureDim_ = 0;
}

Status MemoryBank::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return Status{StatusCode::ErrorInternal, "cannot open for write: " + path};
    }
    uint32_t num = static_cast<uint32_t>(bank_.size());
    uint32_t dim = static_cast<uint32_t>(featureDim_);
    uint32_t ver = kCurrentVersion;

    f.write(reinterpret_cast<const char*>(&kMagicNew), sizeof(kMagicNew));
    f.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    f.write(reinterpret_cast<const char*>(&num), sizeof(num));
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    // 写归一化参数
    uint8_t hasNorm = hasNormParams_ ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&hasNorm), sizeof(hasNorm));
    if (hasNormParams_) {
        f.write(reinterpret_cast<const char*>(normMean_.data()), dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(normStd_.data()), dim * sizeof(float));
    }

    for (auto& pf : bank_) {
        f.write(reinterpret_cast<const char*>(pf.features.data()), dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(&pf.layerIdx), sizeof(pf.layerIdx));
        f.write(reinterpret_cast<const char*>(&pf.patchRow), sizeof(pf.patchRow));
        f.write(reinterpret_cast<const char*>(&pf.patchCol), sizeof(pf.patchCol));
    }
    if (!f.good()) {
        return Status{StatusCode::ErrorInternal, "write failed: " + path};
    }
    return Status{};
}

Status MemoryBank::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return Status{StatusCode::ErrorModelLoad, "cannot open file: " + path};
    }

    uint32_t magic = 0, num = 0, dim = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic == kMagicNew) {
        // 新格式: magic + version + num + dim + hasNorm + [mean/std] + features
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kCurrentVersion) {
            return Status{StatusCode::ErrorModelLoad,
                "unsupported memory bank version: " + std::to_string(version)};
        }
        f.read(reinterpret_cast<char*>(&num), sizeof(num));
        f.read(reinterpret_cast<char*>(&dim), sizeof(dim));

        uint8_t hasNorm = 0;
        f.read(reinterpret_cast<char*>(&hasNorm), sizeof(hasNorm));
        hasNormParams_ = (hasNorm != 0);
        if (hasNormParams_) {
            normMean_.resize(dim);
            normStd_.resize(dim);
            f.read(reinterpret_cast<char*>(normMean_.data()), dim * sizeof(float));
            f.read(reinterpret_cast<char*>(normStd_.data()), dim * sizeof(float));
        }
    } else if (magic == kMagic) {
        // 旧格式: magic + num + dim + features (无 version, 无归一化参数)
        f.read(reinterpret_cast<char*>(&num), sizeof(num));
        f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        hasNormParams_ = false;
        normMean_.clear();
        normStd_.clear();
    } else {
        return Status{StatusCode::ErrorModelLoad,
            "invalid magic in memory bank: " + path};
    }

    bank_.resize(num);
    featureDim_ = static_cast<int>(dim);
    for (uint32_t i = 0; i < num; i++) {
        bank_[i].features.resize(dim);
        f.read(reinterpret_cast<char*>(bank_[i].features.data()), dim * sizeof(float));
        f.read(reinterpret_cast<char*>(&bank_[i].layerIdx), sizeof(bank_[i].layerIdx));
        f.read(reinterpret_cast<char*>(&bank_[i].patchRow), sizeof(bank_[i].patchRow));
        f.read(reinterpret_cast<char*>(&bank_[i].patchCol), sizeof(bank_[i].patchCol));
    }
    if (!f.good()) {
        return Status{StatusCode::ErrorModelLoad, "read failed (truncated?): " + path};
    }
    return Status{};
}

// -------------------------------------------------------
// 暴力最近邻搜索（Brute-force k=1 NN）
// 算法原理：
//   PatchCore 使用特征空间中的最近邻距离作为异常得分。
//   核心假设：正常样本的特征在特征空间中聚集，异常样本（如划痕、污渍）
//   的特征会偏离正常聚类中心，因此到最近邻的距离越大，异常概率越高。
//   
//   距离度量：L2 欧氏距离 ||q - p||₂
//   选择 L2 的原因：
//     - 在 CNN 特征空间中 L2 距离与余弦相似度单调相关（特征已归一化时等价）
//     - 计算开销低于 Mahalanobis 距离（无需协方差矩阵求逆）
//     - 对特征各维度权重平等，无需额外学习
//
//   时间复杂度：O(N·D)，N=记忆库大小，D=特征维度
//   适用场景：记忆库规模 ≤ 10⁵ 时暴力搜索足够快
//   大规模场景可替换为 FAISS 库的近似最近邻搜索（IVF/HNSW）
// -------------------------------------------------------
size_t MemoryBank::NearestNeighbor(const std::vector<float>& query, float& distOut) const {
    if (bank_.empty()) { distOut = 0; return 0; }
    size_t bestIdx = 0;
    float bestDist = std::numeric_limits<float>::max();

    // OpenMP 并行：每个 bank entry 的 L2 距离计算独立，可并行归约
    // #pragma omp parallel for reduction(min:bestDist) schedule(dynamic, 256)
    //   - reduction(min:bestDist) 各线程局部 bestDist 取最小值后合并
    //   - schedule(dynamic, 256) 动态分块，避免长尾效应
    // NOTE: 当前注释掉，编译时需确认 OpenMP 可用后再启用
    // #pragma omp parallel for reduction(min:bestDist) schedule(dynamic, 256)
    for (size_t i = 0; i < bank_.size(); i++) {
        float d = 0;
        for (int j = 0; j < featureDim_; j++) {
            float diff = query[j] - bank_[i].features[j];
            d += diff * diff;
        }
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    distOut = std::sqrt(bestDist);
    return bestIdx;
}

// -------------------------------------------------------
// 计算整张图像的异常热力图
// 流程：
//   1. 遍历所有查询 PatchFeature，确定特征图最大行列数
//   2. 对每个 Patch 位置执行最近邻搜索，得该点异常得分
//   3. 将低分辨率的得分图上采样到原图尺寸（双线性插值）
//
// 异常热力图计算流程详解：
//
// Step 1: 空间索引重建
//   从 PatchFeature 的 (patchRow, patchCol) 字段重建特征图的空间布局。
//   每个 PatchFeature 对应特征图上的一个空间位置，其最近邻距离即为该位置
//   的异常得分（score map），尺寸为 H_feat × W_feat（如 28×28）。
//
// Step 2: 异常得分映射
//   对每个空间位置 (r, c)，用该位置的 Patch 特征向量在记忆库中执行
//   最近邻搜索，得到 L2 距离作为该位置的异常得分 s(r,c)。
//   得分越高，说明该位置越不像记忆库中的"正常"特征。
//
// Step 3: 双线性上采样到原图尺寸
//   特征图得分 s(r,c) 分辨率远低于原图，通过双线性插值上采样到
//   原图尺寸 imgH×imgW，得到像素级的异常热力图。
//   双线性插值公式：
//     f(x,y) = (1-α)(1-β)f(i,j) + α(1-β)f(i+1,j) + (1-α)βf(i,j+1) + αβf(i+1,j+1)
//   其中 i=floor(x), j=floor(y), α=x-i, β=y-j
// -------------------------------------------------------
std::vector<float> MemoryBank::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {
    if (queries.empty()) return {};

    // 按 layerIdx 分组: 不同 layer 的 Patch 空间分辨率不同,
    // 需分别计算得分图再融合
    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

    if (numLayers <= 1) {
        // 单层: 构建低分辨率得分图 → 双线性上采样
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) return {};

        cv::Mat scoreMap(maxRow, maxCol, CV_32F);
        for (auto& q : queries) {
            float dist = 0;
            NearestNeighbor(q.features, dist);
            scoreMap.at<float>(q.patchRow, q.patchCol) = dist;
        }
        cv::Mat upsampled;
        cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        std::vector<float> result(imgW * imgH);
        std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
        return result;
    }

    // 多层融合: 每层独立计算得分图 → 上采样到原图 → 逐像素取最大值
    cv::Mat fused(imgH, imgW, CV_32F, cv::Scalar(0));

    for (int li = 0; li < numLayers; li++) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) continue;

        cv::Mat layerMap(maxRow, maxCol, CV_32F);
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            float dist = 0;
            NearestNeighbor(q.features, dist);
            layerMap.at<float>(q.patchRow, q.patchCol) = dist;
        }

        cv::Mat upsampled;
        cv::resize(layerMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

        // 逐像素 max-fusion，用 cv::max 替代标量循环（SIMD 加速）
        cv::max(fused, upsampled, fused);
    }

    std::vector<float> result(imgW * imgH);
    std::copy(fused.begin<float>(), fused.end<float>(), result.begin());
    return result;
}

} // namespace aicore
