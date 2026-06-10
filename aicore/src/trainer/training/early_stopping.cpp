#include "trainer/training/early_stopping.h"

namespace aicore {

EarlyStopping::EarlyStopping(int patience, float minDelta)
    : patience_(patience), minDelta_(minDelta) {}

bool EarlyStopping::ShouldStop(float currentMetric) {
    if (currentMetric > bestMetric_ + minDelta_) {
        bestMetric_ = currentMetric;
        bestEpoch_++;
        noImprove_ = 0;
    } else {
        noImprove_++;
    }
    return noImprove_ > patience_;
}

void EarlyStopping::Reset() {
    bestEpoch_ = 0;
    bestMetric_ = -1e9f;
    noImprove_ = 0;
}

} // namespace aicore
