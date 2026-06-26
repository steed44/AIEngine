// ============================================================
// tiered_memory_bank.h — 分级 MemoryBank
// 功能：三级存储架构（GPU/CPU/磁盘）的特征库，支持特征升级/
//       降级迁移和最近邻搜索。
//
// 为什么需要三级存储？
//   PatchCore 的 memory bank 包含从正常样本提取的数万到数百万
//   个 patch 特征，大小可达数 GB。全部放在显存不现实（GPU OOM），
//   全部在磁盘又太慢。三级存储在不同成本/速度间做权衡：
//
//   kGPU (Hot)  — 显存，GPU 并行 NN 搜索，速度最快，容量有限
//   kCPU (Warm) — mmap 系统内存，CPU 逐条搜索，速度中等
//   kDisk (Cold)— mmap 磁盘，按缺页中断加载，最慢但容量无限
//
// 升级/降级策略：
//   PromoteToGPU(): 从磁盘加载到显存，触发 LRU 驱逐
//   DemoteToCPU():  释放显存保留 mmap，重新访问触发 page fault
//   DemoteToDisk(): 释放显存+通知 OS 回收 mmap 页面
//
// 异常得分图（Anomaly Map）算法：
//   输入：backbone 提取的 patch 特征列表
//   过程：对每个 patch，在 memory bank 中做暴力最近邻搜索（L2）
//   输出：得分图（特征网格大小）→ 双线性上采样 → 原图大小
//   多层融合：每层独立得分 → 上采样 → 逐像素取最大值
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include <string>
#include <vector>
#include <cstdint>

namespace aicore {

// 存储层级枚举，按访问速度降序
// 自动选择：推理时如果显存够就 PromoteToGPU，不够就用 CPU/磁盘
enum class BankTier { kGPU, kCPU, kDisk };

// 三级分级 MemoryBank 类
// 继承自 binary memory bank 格式（.bin 文件），支持 mmap 懒加载
class TieredMemoryBank {
public:
    TieredMemoryBank() = default;
    ~TieredMemoryBank();
    TieredMemoryBank(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank& operator=(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank(const TieredMemoryBank&) = delete;
    TieredMemoryBank& operator=(const TieredMemoryBank&) = delete;

    // 从 .bin 文件加载特征库，通过 mmap 映射到磁盘层级
    // 支持新旧两种文件格式（自动识别 magic number）
    Status Load(const std::string& path);

    // 清理：释放 GPU 显存、取消 mmap、重置状态
    void Clear();

    size_t Size() const { return num_; }          // 特征数量
    int FeatureDim() const { return dim_; }       // 特征向量维度
    BankTier GetTier() const { return tier_; }    // 当前存储层级

    // 提升到 GPU 显存：从磁盘读取全部特征并通过 cudaMemcpy H2D 传输
    // 触发 MemoryManager 的 LRU 驱逐策略
    Status PromoteToGPU();
    // 降级到 CPU 内存：释放 GPU 显存，数据保留在 mmap 映射中
    void DemoteToCPU();
    // 降级到磁盘：释放 GPU 显存 + 通知 OS 回收 mmap 页面
    void DemoteToDisk();

    // 计算异常热力图 — 自动选择 CPU/GPU 计算路径
    // 单层：NN 搜索 → L2 距离 → sqrt → 得分图 → 上采样到原图尺寸
    // 多层：每层独立计算 → 上采样 → 逐像素 max-fusion
    std::vector<float> ComputeAnomalyMap(
        const std::vector<PatchFeature>& queries, int imgH, int imgW) const;

    static constexpr uint32_t kMagic = 0x50434F52;

private:
    // CPU 计算路径：循环遍历每个 patch × 每个 bank 特征，暴力 L2 距离
    // 复杂度 O(M × N × D)，M=patch 数，N=bank 大小，D=特征维度
    std::vector<float> ComputeOnCPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;
    // GPU 计算路径：批量 CUDA kernel 并行计算距离矩阵
    // 将 M 个 query 打包 → H2D → BatchL2DistanceGPU kernel → D2H
    std::vector<float> ComputeOnGPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;

    BankTier tier_ = BankTier::kDisk;   // 当前存储层级
    std::string path_;                   // 磁盘 .bin 文件路径
    int num_ = 0, dim_ = 0;             // 特征数量与特征维度

    void* mmapPtr_ = nullptr;           // 磁盘文件 mmap 虚拟地址
    size_t mmapSize_ = 0;               // 文件 mmap 大小（字节）

    float* gpuData_ = nullptr;          // GPU 显存指针（PromoteToGPU 后有效）
    uint64_t gpuAllocId_ = 0;           // MemoryManager 中的分配 ID（用于 LRU 追踪）
};

} // namespace aicore
