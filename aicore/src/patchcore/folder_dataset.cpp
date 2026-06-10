#include "patchcore/folder_dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <algorithm>

namespace aicore {

Status FolderDataset::Load(const std::string& folderPath) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(folderPath)) {
        return Status{StatusCode::ErrorInvalidInput, "not a directory: " + folderPath};
    }

    std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    for (auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

        Sample s;
        s.imagePath = entry.path().string();
        s.image = cv::imread(s.imagePath);
        s.label = 0;
        if (!s.image.empty()) {
            samples_.push_back(std::move(s));
        }
    }
    return Status{};
}

size_t FolderDataset::Size() const { return samples_.size(); }

Sample FolderDataset::Get(size_t index) { return samples_.at(index); }

} // namespace aicore
