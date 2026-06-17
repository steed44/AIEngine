// ============================================================
// coreset_sampler.cpp — Coreset（核心集）采样算法实现
// 功能：基于最远点采样（Farthest Point Sampling, FPS）策略，
//       从大规模特征池中贪婪选取最具多样性的子集
// 算法原理：
//   1. 选取离原点最远的点作为第一个种子
//   2. 每次选择距已选集合最远的点加入结果
//   3. 重复直到达到目标数量
// ============================================================
#include "patchcore/coreset_sampler.h"
#include <limits>
#include <cmath>

namespace aicore {

// 计算两个等长浮点向量之间的 L2 欧氏距离
// L2 欧氏距离：||a-b||₂ = sqrt(Σ(a_i - b_i)²)
// 在 PatchCore 特征空间中，L2 距离近似等于特征之间的语义差异程度。
// CNN 深层特征的高维空间具有"语义线性"特性，即语义相似的图像在
// 特征空间中也靠近，因此 L2 距离是合理的相似性度量。
static float L2Dist(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}

// -------------------------------------------------------
// 执行最远点采样（FPS）核心集选择
//
// 算法步骤：
//   1. 如果池为空或目标 ≥ 总数，直接返回全部索引
//   2. 选离原点最远的特征作为第一个种子点
//   3. 更新所有点到已选集合的最小距离
//   4. 贪婪迭代：每次选距离已选集合最远的点
//      并维护 minDist 数组（各点到已选集合的最小距离）
//
// 最远点采样（Farthest Point Sampling, FPS）算法详解：
//
// 数学问题定义：
//   给定 n 个点的集合 P，选择 k 个点构成核心集 S ⊂ P，
//   使得 P 中任意点到 S 的最近距离的最大值最小化。
//   即：min_{S,|S|=k} max_{p∈P} min_{s∈S} ||p-s||₂
//   这是一个标准的 k-center 问题（NP-hard），FPS 提供 (1+ε) 近似。
//
// 为什么 FPS 适合 PatchCore：
//   PatchCore 训练时从正常图像提取了大量 Patch 特征（通常 10⁵-10⁶ 级）。
//   直接存储所有特征会导致记忆库过大且推理时 NN 搜索变慢。
//   FPS 选择出的特征子集能保持原始特征空间的覆盖率（diversity），
//   即任意未选中的特征都在已选特征附近，不会漏掉任何"正常模式"。
//
// 算法各步骤作用：
//   1. 选择离原点最远的特征：确保初始种子在特征空间边缘，避免"偏科"
//   2. 维护 minDist[i] = min_{s∈selected} ||p_i - s||₂
//      记录每个未选点到已选集合的最小距离
//   3. 每次选择使 minDist 最大化的点：这等于挑选当前"缺口"最大的位置
//   4. 新点加入后更新 minDist：如果新点离 p_i 更近，就缩小 minDist[i]
//
// 时间复杂度：O(nk)，其中 n=池大小，k=目标采样数
// FPS 后 MemoryBank 大小从 n 压缩到 k = n * coresetFraction
// -------------------------------------------------------
std::vector<size_t> CoresetSampler::Sample(
    const std::vector<PatchFeature>& pool, size_t targetSize) {
    if (pool.empty() || targetSize >= pool.size()) {
        std::vector<size_t> all(pool.size());
        for (size_t i = 0; i < pool.size(); i++) all[i] = i;
        return all;
    }

    size_t n = pool.size();
    std::vector<float> minDist(n, std::numeric_limits<float>::max());
    std::vector<bool> selected(n, false);
    std::vector<size_t> result;
    result.reserve(targetSize);

    // Step 1: 选离原点（零向量）最远的点作为初始种子
    // 选择离原点最远的特征作为第一个种子（而非随机选择）
    // 原因：随机选择可能导致中心聚集，远离中心的 "边缘模式" 可能被遗漏。
    // 选择最远点保证了初始覆盖的广度。
    size_t first = 0;
    float maxNorm = 0;
    for (size_t i = 0; i < n; i++) {
        float d = L2Dist(pool[i].features, std::vector<float>(pool[i].features.size(), 0));
        if (d > maxNorm) { maxNorm = d; first = i; }
    }
    result.push_back(first);
    selected[first] = true;

    // Step 2: 更新所有点到第一个种子的距离
    for (size_t i = 0; i < n; i++) {
        if (!selected[i]) {
            minDist[i] = L2Dist(pool[i].features, pool[first].features);
        }
    }

    // Step 3~N: 迭代选取最远点
    // 核心循环：每次选择使当前覆盖范围"漏洞"最大的点
    // 这等价于：当前已选集合的 Voronoi 图中，面积最大的 cell 的中心
    // 每一步都在补上当前覆盖最差的位置
    for (size_t k = 1; k < targetSize; k++) {
        // 找距离已选集合最远的未选点
        size_t best = 0;
        float bestVal = -1;
        for (size_t i = 0; i < n; i++) {
            if (!selected[i] && minDist[i] > bestVal) {
                bestVal = minDist[i];
                best = i;
            }
        }
        result.push_back(best);
        selected[best] = true;

        // 用新选中的点更新其余未选点的最小距离
        for (size_t i = 0; i < n; i++) {
            if (!selected[i]) {
                float d = L2Dist(pool[i].features, pool[best].features);
                if (d < minDist[i]) minDist[i] = d;
            }
        }
    }
    return result;
}

} // namespace aicore
