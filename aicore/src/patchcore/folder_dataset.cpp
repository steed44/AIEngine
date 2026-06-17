// ============================================================
// folder_dataset.cpp — 文件夹数据集实现
// 功能：遍历文件夹，自动识别常见图像格式并读取为 Sample
// ============================================================
#include "patchcore/folder_dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <algorithm>

namespace aicore {

// -------------------------------------------------------
// 加载文件夹中所有图像文件
// 支持的扩展名：.jpg .jpeg .png .bmp .tiff .tif
// 忽略非图像文件和无法读取的损坏图像
// -------------------------------------------------------
Status FolderDataset::Load(const std::string& folderPath) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(folderPath)) {
        return Status{StatusCode::ErrorInvalidInput, "not a directory: " + folderPath};
    }

    std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    for (auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        // 检查文件扩展名是否为支持的图像格式（不区分大小写）
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

std::vector<std::string> FolderDataset::ListImageFiles(const std::string& folderPath) {
    namespace fs = std::filesystem;
    std::vector<std::string> result;
    if (!fs::is_directory(folderPath)) return result;

    std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    for (auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
        result.push_back(entry.path().string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

size_t FolderDataset::Size() const { return samples_.size(); }

Sample FolderDataset::Get(size_t index) { return samples_.at(index); }

} // namespace aicore
