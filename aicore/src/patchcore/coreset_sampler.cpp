#include "patchcore/coreset_sampler.h"
#include <limits>
#include <cmath>

namespace aicore {

static float L2Dist(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}

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

    size_t first = 0;
    float maxNorm = 0;
    for (size_t i = 0; i < n; i++) {
        float d = L2Dist(pool[i].features, std::vector<float>(pool[i].features.size(), 0));
        if (d > maxNorm) { maxNorm = d; first = i; }
    }
    result.push_back(first);
    selected[first] = true;

    for (size_t i = 0; i < n; i++) {
        if (!selected[i]) {
            minDist[i] = L2Dist(pool[i].features, pool[first].features);
        }
    }

    for (size_t k = 1; k < targetSize; k++) {
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
