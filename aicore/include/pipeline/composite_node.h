// 复合节点 — 将子流水线嵌入到父流水线中作为一个节点
#pragma once
#include "core/processor.h"
#include "core/pipeline.h"
#include <memory>

namespace aicore {

// 复合节点包装器
// 将一条完整的流水线（IPipeline）封装为 IProcessor 节点，
// 在父流水线的 DAG 中作为一个独立处理步骤参与调度
class CompositeNode : public IProcessor {
public:
    Status Init(const NodeConfig& config) override;
    Status Process(const std::vector<Frame>& inputs,
                   std::vector<Frame>& outputs) override;
    std::string GetName() const override;
    std::string GetType() const override;

    // 设置内部子流水线实例（所有权移交）
    void SetInnerPipeline(std::unique_ptr<IPipeline> pipeline);

private:
    std::unique_ptr<IPipeline> innerPipeline_;  // 内部子流水线
    std::string name_;                          // 节点名称
};

} // namespace aicore
