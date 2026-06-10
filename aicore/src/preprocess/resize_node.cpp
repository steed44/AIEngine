#include "preprocess/resize_node.h"

namespace aicore {

Status ResizeNode::Init(const NodeConfig& config) {
    auto it = config.find("width");
    if (it != config.end()) targetWidth_ = std::stoi(it->second);
    it = config.find("height");
    if (it != config.end()) targetHeight_ = std::stoi(it->second);
    return Status{};
}

Status ResizeNode::Process(const std::vector<Frame>& inputs,
                           std::vector<Frame>& outputs) {
    for (const auto& frame : inputs) {
        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;
        cv::resize(frame.image, out.image,
                   cv::Size(targetWidth_, targetHeight_),
                   0, 0, interpolation_);
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string ResizeNode::GetName() const { return "resize"; }
std::string ResizeNode::GetType() const { return "resize"; }

} // namespace aicore
