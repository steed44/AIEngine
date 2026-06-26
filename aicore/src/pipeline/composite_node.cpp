// ============================================================
// composite_node.cpp — 复合节点
// 将子流水线作为一个节点嵌入到父流水线 DAG 中
// ============================================================
#include "pipeline/composite_node.h"

namespace aicore {

/**
 * 初始化复合节点：读取节点名称
 * @param config 节点配置键值对
 */
Status CompositeNode::Init(const NodeConfig& config) {
    auto it = config.find("name");
    name_ = (it != config.end()) ? it->second : "composite";
    return Status{};
}

/**
 * 执行子流水线：逐帧送入内部流水线，汇总检测结果
 * @param inputs  输入帧列表
 * @param outputs [out] 输出帧列表（当前透传空）
 */
Status CompositeNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    if (!innerPipeline_)
        return Status{StatusCode::ErrorInternal, "inner pipeline not set"};
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "no input frames"};

    // 逐帧执行内部流水线，将子结果合并到总结果中
    // 子流水线内部可能包含多层 DAG 节点（递归嵌套）
    Result result;
    for (const auto& frame : inputs) {
        Result r;
        // 逐帧推理：子流水线内部有自己的 DAG 拓扑和执行器
        auto s = innerPipeline_->Execute(frame, r);
        if (!s) return s;
        // 合并检测结果和延迟指标
        for (auto& d : r.detections)
            result.detections.push_back(std::move(d));
        result.totalLatencyMs += r.totalLatencyMs;
    }
    return Status{};
}

/** 返回配置中设定的节点名称 */
std::string CompositeNode::GetName() const { return name_; }
/** 返回节点类型标识 */
std::string CompositeNode::GetType() const { return "composite"; }

/**
 * 设置内部子流水线
 * @param pipeline 子流水线实例的所有权
 */
void CompositeNode::SetInnerPipeline(std::unique_ptr<IPipeline> pipeline) {
    innerPipeline_ = std::move(pipeline);
}

} // namespace aicore
