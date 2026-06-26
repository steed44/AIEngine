// ============================================================
// yolo_decode_node.h — YOLO 解码节点声明
//
// 功能：将 YOLO 模型的多尺度输出张量（ONNX/TensorRT raw output）
// 解码为检测框列表，支持 YOLOv8（anchor-free，DFL 解码）和
// YOLOv5/v3（anchor-based，直接回归）两种架构。
//
// 在流水线中的位置：
//   [输入图像] → [Letterbox/Resize] → [Normalize] → [YOLO 模型推理]
//   → [YoloDecodeNode] → [NmsNode] → [输出检测结果]
//
// YOLOv8 输出张量规格（3 个检测头，各输出一个 4 维张量）：
//   尺度 1 (P3): 1 × 144 × 80 × 80   (小目标检测头，stride=8)
//   尺度 2 (P4): 1 × 144 × 40 × 40   (中目标检测头，stride=16)
//   尺度 3 (P5): 1 × 144 × 20 × 20   (大目标检测头，stride=32)
//   其中 144 = 4 × 16 (DFL 回归) + 80 (COCO 类别)
// ============================================================
#pragma once
#include "core/processor.h"
#include "core/types.h"
#include <string>
#include <vector>

namespace aicore {

// YOLO 解码节点：将模型原始输出张量解码为 NodeResult 格式的检测框
// 支持 YOLOv8 anchor-free 和 YOLOv5/v3 anchor-based
class YoloDecodeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    // 单尺度解码：将某一尺度的输出张量解码为候选框列表
    // data     — 张量数据指针，形状 [1, C, H, W]
    // numBoxes — H × W，该尺度的总 grid cell 数
    // stride   — C，每个 cell 的 float 数 = 4×regMax + numClasses
    // gridW/H  — 特征图宽高
    // numClasses — 类别数
    // scale/padX/padY — letterbox 逆变换参数
    // candidates — [out] 解码后的候选框列表
    void DecodeScale(const float* data, int numBoxes, int stride,
                     int gridW, int gridH, int numClasses,
                     float scale, int padX, int padY,
                     std::vector<NodeResult>& candidates) const;

    std::string versionStr_ = "v8";               // YOLO 版本 ("v8" 或 "")
    float confidenceThreshold_ = 0.5f;             // 置信度阈值
    float iouThreshold_ = 0.45f;                   // NMS IoU 阈值
    int numClasses_ = 80;                          // 类别数（COCO=80）
    int modelInputSize_ = 640;                     // 模型输入尺寸
    std::string name_ = "yolo_decode";             // 节点名称
};

} // namespace aicore
