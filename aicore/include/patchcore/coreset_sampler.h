#pragma once
#include "patchcore/memory_bank.h"
#include <vector>

namespace aicore {

class CoresetSampler {
public:
    std::vector<size_t> Sample(const std::vector<PatchFeature>& pool,
                                size_t targetSize);
};

} // namespace aicore
