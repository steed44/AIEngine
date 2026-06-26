// YOLO 解码节点 — 将模型原始输出张量解码为检测框 NodeResult
#pragma once
#include "core/processor.h"
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

class YoloDecodeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    void DecodeScale(const float* data, int numBoxes, int stride,
                     int gridW, int gridH, int numClasses,
                     float scale, int padX, int padY,
                     std::vector<NodeResult>& candidates) const;

    std::string versionStr_ = "v8";
    float confidenceThreshold_ = 0.5f;
    float iouThreshold_ = 0.45f;
    int numClasses_ = 80;
    int modelInputSize_ = 640;
    std::string name_ = "yolo_decode";
};

} // namespace aicore
