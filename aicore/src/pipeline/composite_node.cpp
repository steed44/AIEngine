#include "pipeline/composite_node.h"

namespace aicore {

Status CompositeNode::Init(const NodeConfig& config) {
    auto it = config.find("name");
    name_ = (it != config.end()) ? it->second : "composite";
    return Status{};
}

Status CompositeNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    if (!innerPipeline_)
        return Status{StatusCode::ErrorInternal, "inner pipeline not set"};
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "no input frames"};

    Result result;
    for (const auto& frame : inputs) {
        Result r;
        auto s = innerPipeline_->Execute(frame, r);
        if (!s) return s;
        for (auto& d : r.detections)
            result.detections.push_back(std::move(d));
        result.totalLatencyMs += r.totalLatencyMs;
    }
    return Status{};
}

std::string CompositeNode::GetName() const { return name_; }
std::string CompositeNode::GetType() const { return "composite"; }

void CompositeNode::SetInnerPipeline(std::unique_ptr<IPipeline> pipeline) {
    innerPipeline_ = std::move(pipeline);
}

} // namespace aicore
