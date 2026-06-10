#include "preprocess/normalize_node.h"

namespace aicore {

Status NormalizeNode::Init(const NodeConfig& config) {
    auto it = config.find("bgr_to_rgb");
    if (it != config.end()) bgrToRgb_ = (it->second == "true");
    return Status{};
}

Status NormalizeNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    for (const auto& frame : inputs) {
        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;

        cv::Mat img = frame.image.clone();
        if (bgrToRgb_ && img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        img.convertTo(out.image, CV_32FC3, 1.0 / 255.0);

        for (int i = 0; i < img.channels(); ++i) {
            cv::Mat channel(out.image.size(), CV_32FC1,
                            out.image.ptr<float>(0) + i * out.image.cols * out.image.rows);
            cv::subtract(channel, cv::Scalar(mean_[i]), channel);
            cv::divide(channel, cv::Scalar(std_[i]), channel);
        }

        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string NormalizeNode::GetName() const { return "normalize"; }
std::string NormalizeNode::GetType() const { return "normalize"; }

} // namespace aicore
