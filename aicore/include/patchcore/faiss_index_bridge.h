#pragma once
#include "core/types.h"
#include "patchcore/faiss_index.h"
#include "patchcore/memory_bank.h"
#include <string>
#include <vector>
#include <memory>

namespace aicore {

/**
 * FaissIndexBridge — MemoryBank .bin 与 FaissIndex 的桥接层
 *
 * 职责：
 *   1. 从 MemoryBank .bin 文件读取特征数据
 *   2. 构建 FaissIndex 索引
 *   3. 提供与 TieredMemoryBank::ComputeAnomalyMap 兼容的异常热力图接口
 *   4. 自动管理 .faiss 索引文件的加载/保存/缓存
 *
 * 降级策略：
 *   - .faiss 文件不存在或损坏 → 自动从 .bin 重新训练
 *   - 训练失败 → 返回错误，由上层降级到暴力搜索
 */
class FaissIndexBridge {
public:
    // 从 .bin 文件读取特征并训练 FAISS 索引
    // 同时将索引保存到 .{algorithm}.faiss 文件（下次加速加载）
    Status TrainFromMemoryBank(const std::string& memoryBankPath,
                               const FaissIndexConfig& cfg);

    // 直接加载已训练的 .faiss 索引文件
    Status LoadIndex(const std::string& indexPath);

    // 将当前索引保存到文件
    Status SaveIndex(const std::string& indexPath) const;

    // 计算异常热力图（与 TieredMemoryBank::ComputeAnomalyMap 接口兼容）
    // queries: PatchFeature 列表
    // imgH, imgW: 原图尺寸（用于上采样）
    // 返回 CV_32F 热力图（扁平化，行优先）
    cv::Mat ComputeAnomalyMap(const std::vector<PatchFeature>& queries,
                              int imgH, int imgW) const;

    // 访问器
    const FaissIndex& GetIndex() const { return index_; }
    bool IsReady() const { return index_.IsTrained(); }

private:
    FaissIndex index_;
    std::string binPath_;
};

} // namespace aicore
