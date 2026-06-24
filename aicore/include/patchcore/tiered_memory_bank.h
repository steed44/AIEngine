// 分级 MemoryBank — 支持 GPU/CPU/磁盘三级存储
// 根据可用资源自动选择计算设备，大记忆中先 mmap 到磁盘再按需升级
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include <string>
#include <vector>
#include <cstdint>

namespace aicore {

// 分级层级枚举
// kGPU: 显存中（最快，搜索时 GPU 并行计算）
// kCPU: 系统内存中（中等速度）
// kDisk: 磁盘 mmap（最慢，按需加载）
enum class BankTier { kGPU, kCPU, kDisk };

// 分级 MemoryBank
// 对长 MemoryBank 提供三级存储支持，自动选择最佳计算设备
class TieredMemoryBank {
public:
    TieredMemoryBank() = default;
    ~TieredMemoryBank();
    TieredMemoryBank(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank& operator=(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank(const TieredMemoryBank&) = delete;
    TieredMemoryBank& operator=(const TieredMemoryBank&) = delete;

    // 从二进制文件加载特征库，优先 mmap 到磁盘层级
    Status Load(const std::string& path);

    void Clear();
    size_t Size() const { return num_; }
    int FeatureDim() const { return dim_; }
    BankTier GetTier() const { return tier_; }

    // 将特征库升级到 GPU 显存（大幅加速 NN 搜索）
    Status PromoteToGPU();
    // 降级到 CPU 内存
    void DemoteToCPU();
    // 降级到磁盘 mmap（释放内存/显存）
    void DemoteToDisk();

    // 计算异常热力图（自动选择当前层级的计算路径）
    std::vector<float> ComputeAnomalyMap(
        const std::vector<PatchFeature>& queries, int imgH, int imgW) const;

    static constexpr uint32_t kMagic = 0x50434F52;

private:
    // CPU 路径：暴力 NN 搜索
    std::vector<float> ComputeOnCPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;
    // GPU 路径：使用 CUDA 内核并行 NN 搜索（待实现）
    std::vector<float> ComputeOnGPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;

    BankTier tier_ = BankTier::kDisk;   // 当前存储层级
    std::string path_;                   // 磁盘文件路径
    int num_ = 0, dim_ = 0;             // 特征数量与维度

    void* mmapPtr_ = nullptr;           // 磁盘 mmap 指针
    size_t mmapSize_ = 0;               // mmap 大小

    float* gpuData_ = nullptr;          // GPU 显存指针（升级后有效）
    uint64_t gpuAllocId_ = 0;           // GPU 显存分配 ID
};

} // namespace aicore
