#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include "trainer/data/dataset.h"
#include <string>

namespace aicore {

struct AICORE_API PatchCoreTrainConfig {
    int inputSize = 224;
    std::string backboneLayers = "layer2,layer3";
    std::string backboneType = "opencv_dnn";
    double coresetFraction = 0.1;
    size_t maxFeatures = 100000;
};

class AICORE_API PatchCoreTrainer {
public:
    Status Train(IDataset& dataset, const std::string& modelPath,
                 const std::string& outputPath,
                 const PatchCoreTrainConfig& cfg);
    Status TrainFromFolder(const std::string& folderPath,
                           const std::string& modelPath,
                           const std::string& outputPath,
                           const PatchCoreTrainConfig& cfg);
    std::string GetLastError() const { return lastError_; }

private:
    std::string lastError_;
};

} // namespace aicore
