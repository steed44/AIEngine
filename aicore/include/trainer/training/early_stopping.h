#pragma once
#include "core/types.h"

namespace aicore {

class EarlyStopping {
public:
    EarlyStopping(int patience = 10, float minDelta = 0.001f);
    bool ShouldStop(float currentMetric);
    void Reset();
    int GetBestEpoch() const { return bestEpoch_; }
    float GetBestMetric() const { return bestMetric_; }

private:
    int patience_;
    float minDelta_;
    int bestEpoch_ = 0;
    float bestMetric_ = -1e9f;
    int noImprove_ = 0;
};

} // namespace aicore
