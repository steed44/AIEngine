#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <string>

namespace aicore {

struct Frame {
    cv::Mat image;
    uint64_t frameId = 0;
    uint64_t timestamp = 0;
    std::string sourceId;
    std::map<std::string, float> roiMap;

    Frame() = default;

    Frame(cv::Mat img, uint64_t id = 0)
        : image(std::move(img)), frameId(id) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool empty() const { return image.empty(); }
    int width() const { return image.cols; }
    int height() const { return image.rows; }
};

} // namespace aicore
