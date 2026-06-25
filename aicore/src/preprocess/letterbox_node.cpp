#include "preprocess/letterbox_node.h"

namespace aicore {

Status LetterboxNode::Init(const NodeConfig& config) {
    auto it = config.find("width");
    if (it != config.end()) targetWidth_ = std::stoi(it->second);
    it = config.find("height");
    if (it != config.end()) targetHeight_ = std::stoi(it->second);
    it = config.find("pad_color");
    if (it != config.end()) {
        // pad_color 格式: "114,114,114" (BGR)
        int r, g, b;
        sscanf_s(it->second.c_str(), "%d,%d,%d", &b, &g, &r);
        padColor_ = cv::Scalar(b, g, r);
    }
    return Status{};
}

Status LetterboxNode::Process(const std::vector<Frame>& inputs,
                               std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "letterbox: no input"};

    for (const auto& frame : inputs) {
        const auto& img = frame.image;
        int h = img.rows, w = img.cols;
        if (h == 0 || w == 0)
            return Status{StatusCode::ErrorPreprocess, "letterbox: empty image"};

        // 计算等比例缩放因子
        float scale = std::min((float)targetWidth_ / w, (float)targetHeight_ / h);
        int newW = (int)std::round(w * scale);
        int newH = (int)std::round(h * scale);

        // 等比例 resize
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // 计算 padding
        int padX = (targetWidth_ - newW) / 2;
        int padY = (targetHeight_ - newH) / 2;

        // 灰边填充
        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, padY, padY, padX, padX,
                           cv::BORDER_CONSTANT, padColor_);

        Frame out(std::move(padded), frame.frameId);
        // 传递 letterbox 参数给下游解码节点
        out.roiMap["letterbox_scale"] = scale;
        out.roiMap["letterbox_pad_x"] = (float)padX;
        out.roiMap["letterbox_pad_y"] = (float)padY;
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string LetterboxNode::GetName() const { return "letterbox"; }
std::string LetterboxNode::GetType() const { return "letterbox"; }

} // namespace aicore
