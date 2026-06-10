#pragma once
#include "core/processor.h"
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include "patchcore/memory_bank.h"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace aicore {

class PatchCoreNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override { return name_; }
    std::string GetType() const override { return "patchcore"; }

private:
    std::vector<PatchFeature> ForwardOpenCVDnn(const cv::Mat& blob);
    std::vector<PatchFeature> ForwardModelBackend(const cv::Mat& img);

    std::string name_;
    bool useOpenCVDnn_ = true;
    cv::dnn::Net net_;
    std::unique_ptr<IModelBackend> backend_;
    MemoryBank memoryBank_;
    std::vector<std::string> outputLayerNames_;
    int inputSize_ = 224;
    float anomalyThreshold_ = 0.5f;
};

} // namespace aicore
