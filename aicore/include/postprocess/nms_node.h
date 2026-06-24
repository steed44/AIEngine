// NMS 后处理节点 — 对模型输出的检测框执行非极大值抑制
#pragma once
#include "core/processor.h"

namespace aicore {

// 非极大值抑制（NMS）处理器
// 对检测结果按置信度排序，移除与高置信度框 IoU 过大的冗余框
class NmsNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    float iouThreshold_ = 0.45f;           // NMS IoU 阈值（高于此值的重叠框被抑制）
    float confidenceThreshold_ = 0.5f;     // 置信度阈值（低于此值的框被过滤）
};

} // namespace aicore
