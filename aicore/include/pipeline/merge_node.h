// 合并节点 — 汇聚多路输入流到一路输出
//
// 在 DAG 中作为汇聚点使用：
//   多个上游节点的输出帧在此合并为 vector，
//   后续节点可同时处理来自不同分支的数据。
//   支持 maxInputs 参数限制输入路数。
#pragma once
#include "core/processor.h"
#include <vector>
#include <string>

namespace aicore {

// 多路合并节点
// 将上游多个输入源的帧汇聚到一路输出中，支持输入数量上限检查
class MergeNode : public IProcessor {
public:
    MergeNode();
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

private:
    std::string mergeMode_ = "concat";  // 合并模式（concat / sum / mean 等）
    int maxInputs_ = 0;                 // 最大输入路数限制（0 表示不限）
};

} // namespace aicore
