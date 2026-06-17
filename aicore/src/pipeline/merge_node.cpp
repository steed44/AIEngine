// ============================================================
// merge_node.cpp — 合并节点
// 汇聚多路输入流，透传所有输入帧到输出
// ============================================================
#include "pipeline/merge_node.h"

namespace aicore {

/** 默认构造函数 */
MergeNode::MergeNode() {}

/**
 * 初始化合并节点：读取合并模式及最大输入数量限制
 * @param config 节点配置键值对
 */
Status MergeNode::Init(const NodeConfig& config) {
    auto it = config.find("mode");
    if (it != config.end()) mergeMode_ = it->second;
    it = config.find("max_inputs");
    if (it != config.end()) maxInputs_ = std::stoi(it->second);
    return Status{};
}

/**
 * 执行合并操作：校验输入数量后直接透传
 * @param inputs  上游所有输入帧的汇集
 * @param outputs [out] 与 inputs 相同的帧列表
 */
Status MergeNode::Process(const std::vector<Frame>& inputs,
                          std::vector<Frame>& outputs) {
    if (maxInputs_ > 0 && static_cast<int>(inputs.size()) > maxInputs_)
        return Status{StatusCode::ErrorInvalidInput, "too many inputs"};
    outputs = inputs;
    return Status{};
}

/** 返回节点名称 */
std::string MergeNode::GetName() const { return "merge"; }
/** 返回节点类型标识 */
std::string MergeNode::GetType() const { return "merge"; }

} // namespace aicore
