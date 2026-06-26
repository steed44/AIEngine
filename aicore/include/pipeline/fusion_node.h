#pragma once
#include "core/processor.h"
#include "patchcore/backbone.h"
#include "patchcore/memory_bank.h"
#include <memory>

namespace aicore {

// FusionNode — YOLO 检测结果 + PatchCore 异常评分融合
//
// 输入:
//   inputs[0] — 带 detections 的帧（来自 NmsNode）
//   inputs[1] — 原始图像帧（用于 ROI 裁剪）
// 输出:
//   每项 detection.measurements 追加:
//     "anomaly_score" — PatchCore 异常得分 [0, ∞)
//     "is_anomaly"    — 1.0（异常）或 0.0（正常）
//
// Config 参数:
//   backbone_type    — opencv_dnn / libtorch / model_backend
//   model_path       — backbone 模型路径
//   input_size       — backbone 输入尺寸
//   backbone_layers  — 特征提取层（逗号分隔）
//   memory_bank_path — MemoryBank .bin 文件路径
//   anomaly_threshold — 异常判定阈值
class FusionNode : public IProcessor {
public:
    FusionNode();
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

    void SetThreadPool(ThreadPool* pool) override;

private:
    Status ProcessOneRoi(const cv::Mat& fullImage, const BBox& bbox,
                         float& outScore, cv::Mat& outHeatmap);

    std::unique_ptr<IBackbone> backbone_;
    MemoryBank memoryBank_;
    std::string nodeId_;
    float anomalyThreshold_ = 0.5f;
    int inputSize_ = 224;
    bool initialized_ = false;
    ThreadPool* threadPool_ = nullptr;
};

} // namespace aicore
