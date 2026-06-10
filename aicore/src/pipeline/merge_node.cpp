#include "pipeline/merge_node.h"

namespace aicore {

MergeNode::MergeNode() {}

Status MergeNode::Init(const NodeConfig& config) {
    auto it = config.find("mode");
    if (it != config.end()) mergeMode_ = it->second;
    it = config.find("max_inputs");
    if (it != config.end()) maxInputs_ = std::stoi(it->second);
    return Status{};
}

Status MergeNode::Process(const std::vector<Frame>& inputs,
                          std::vector<Frame>& outputs) {
    if (maxInputs_ > 0 && static_cast<int>(inputs.size()) > maxInputs_)
        return Status{StatusCode::ErrorInvalidInput, "too many inputs"};
    outputs = inputs;
    return Status{};
}

std::string MergeNode::GetName() const { return "merge"; }
std::string MergeNode::GetType() const { return "merge"; }

} // namespace aicore
