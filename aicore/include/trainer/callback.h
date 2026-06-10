#pragma once
#include "core/types.h"

namespace aicore {

class ITrainCallback {
public:
    virtual ~ITrainCallback() = default;
    virtual void OnEpochBegin(int epoch) = 0;
    virtual void OnEpochEnd(int epoch, float loss, float metric) = 0;
    virtual void OnBatchEnd(int batch, float loss) = 0;
    virtual void OnTrainEnd(float bestMetric) = 0;
};

} // namespace aicore
