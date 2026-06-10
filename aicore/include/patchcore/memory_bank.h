#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace aicore {

struct PatchFeature {
    std::vector<float> features;
    int layerIdx = 0;
    int patchRow = 0, patchCol = 0;
};

class MemoryBank {
public:
    MemoryBank() = default;

    bool Load(const std::string& path);
    bool Save(const std::string& path) const;

    void Build(const std::vector<PatchFeature>& features);
    void Clear();

    size_t NearestNeighbor(const std::vector<float>& query, float& distOut) const;
    std::vector<float> ComputeAnomalyMap(const std::vector<PatchFeature>& queries,
                                          int imgH, int imgW) const;

    size_t Size() const { return bank_.size(); }
    int FeatureDim() const { return featureDim_; }

    static constexpr uint32_t kMagic = 0x50434F52;

private:
    std::vector<PatchFeature> bank_;
    int featureDim_ = 0;
};

} // namespace aicore
