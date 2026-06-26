#include "postprocess/nms_common.h"
#include <algorithm>

namespace aicore {

void NMSCommon(std::vector<NodeResult>& candidates, float iouThreshold) {
    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.confidence > b.confidence; });

    std::vector<bool> keep(candidates.size(), true);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (!keep[j] || candidates[i].label != candidates[j].label)
                continue;
            if (IouBox(candidates[i].bbox, candidates[j].bbox) > iouThreshold)
                keep[j] = false;
        }
    }

    size_t writeIdx = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (keep[i])
            candidates[writeIdx++] = std::move(candidates[i]);
    }
    candidates.resize(writeIdx);
}

} // namespace aicore
