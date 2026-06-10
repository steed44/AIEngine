#include "trainer/data/dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>

namespace aicore {

Status COCODataset::Load(const std::string& path) {
    (void)path;
    return Status{};
}

size_t COCODataset::Size() const {
    return samples_.size();
}

Sample COCODataset::Get(size_t index) {
    if (index >= samples_.size()) return {};
    auto& s = samples_[index];
    s.image = cv::imread(s.imagePath);
    return s;
}

int COCODataset::NumClasses() const {
    return 80;
}

} // namespace aicore
