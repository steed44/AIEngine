// ============================================================
// coreset_sampler.h — Coreset（核心集）采样器
// 功能：对大规模 Patch 特征池进行贪婪子集采样，选择最具
//       代表性的特征子集，显著减少 MemoryBank 的存储和计算开销
// 算法：基于最远点采样（Farthest Point Sampling）策略
// ============================================================
#pragma once
#include "patchcore/memory_bank.h"
#include <vector>

namespace aicore {

// -------------------------------------------------------
// CoresetSampler — 核心集采样器
// 职责：从大量 PatchFeature 中选出一个大小受控的子集，
//       使得子集最大程度覆盖原始分布，同时压缩至 targetSize
// 典型使用场景：训练 PatchCore 时，提取到的特征数量可达
//       数十万，通过 Coreset 采样缩减至 1%-10% 再存入 MemoryBank
// -------------------------------------------------------
class CoresetSampler {
public:
    // 执行核心集采样 (FPS, O(n*k))
    std::vector<size_t> Sample(const std::vector<PatchFeature>& pool,
                                size_t targetSize);

    // 快速采样: 从 pool 中随机取 maxCandidates 个候选, 再 FPS
    // 适合 pool 极大(n>50000)时使用, 大幅减少 O(nk) 开销
    // @param pool          原始特征池
    // @param targetSize    目标采样数量
    // @param maxCandidates 随机候选上限 (默认 20000)
    std::vector<size_t> FastSample(const std::vector<PatchFeature>& pool,
                                    size_t targetSize,
                                    size_t maxCandidates = 20000);
};

} // namespace aicore
