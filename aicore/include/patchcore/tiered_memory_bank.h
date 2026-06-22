#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include <string>
#include <vector>
#include <cstdint>

namespace aicore {

enum class BankTier { kGPU, kCPU, kDisk };

class TieredMemoryBank {
public:
    TieredMemoryBank() = default;
    ~TieredMemoryBank();
    TieredMemoryBank(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank& operator=(TieredMemoryBank&& other) noexcept;
    TieredMemoryBank(const TieredMemoryBank&) = delete;
    TieredMemoryBank& operator=(const TieredMemoryBank&) = delete;

    Status Load(const std::string& path);

    void Clear();
    size_t Size() const { return num_; }
    int FeatureDim() const { return dim_; }
    BankTier GetTier() const { return tier_; }

    Status PromoteToGPU();
    void DemoteToCPU();
    void DemoteToDisk();

    std::vector<float> ComputeAnomalyMap(
        const std::vector<PatchFeature>& queries, int imgH, int imgW) const;

    static constexpr uint32_t kMagic = 0x50434F52;

private:
    std::vector<float> ComputeOnCPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;
    std::vector<float> ComputeOnGPU(const std::vector<PatchFeature>& queries,
                                    int imgH, int imgW) const;

    BankTier tier_ = BankTier::kDisk;
    std::string path_;
    int num_ = 0, dim_ = 0;

    void* mmapPtr_ = nullptr;
    size_t mmapSize_ = 0;

    float* gpuData_ = nullptr;
    uint64_t gpuAllocId_ = 0;
};

} // namespace aicore
