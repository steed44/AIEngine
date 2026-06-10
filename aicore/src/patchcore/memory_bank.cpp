#include "patchcore/memory_bank.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace aicore {

void MemoryBank::Build(const std::vector<PatchFeature>& features) {
    bank_ = features;
    if (!features.empty()) {
        featureDim_ = static_cast<int>(features[0].features.size());
    }
}

void MemoryBank::Clear() {
    bank_.clear();
    featureDim_ = 0;
}

bool MemoryBank::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t num = static_cast<uint32_t>(bank_.size());
    uint32_t dim = static_cast<uint32_t>(featureDim_);
    f.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    f.write(reinterpret_cast<const char*>(&num), sizeof(num));
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    for (auto& pf : bank_) {
        f.write(reinterpret_cast<const char*>(pf.features.data()), dim * sizeof(float));
        f.write(reinterpret_cast<const char*>(&pf.layerIdx), sizeof(pf.layerIdx));
        f.write(reinterpret_cast<const char*>(&pf.patchRow), sizeof(pf.patchRow));
        f.write(reinterpret_cast<const char*>(&pf.patchCol), sizeof(pf.patchCol));
    }
    return f.good();
}

bool MemoryBank::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, num = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kMagic) return false;
    f.read(reinterpret_cast<char*>(&num), sizeof(num));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    bank_.resize(num);
    featureDim_ = static_cast<int>(dim);
    for (uint32_t i = 0; i < num; i++) {
        bank_[i].features.resize(dim);
        f.read(reinterpret_cast<char*>(bank_[i].features.data()), dim * sizeof(float));
        f.read(reinterpret_cast<char*>(&bank_[i].layerIdx), sizeof(bank_[i].layerIdx));
        f.read(reinterpret_cast<char*>(&bank_[i].patchRow), sizeof(bank_[i].patchRow));
        f.read(reinterpret_cast<char*>(&bank_[i].patchCol), sizeof(bank_[i].patchCol));
    }
    return f.good();
}

size_t MemoryBank::NearestNeighbor(const std::vector<float>& query, float& distOut) const {
    if (bank_.empty()) return 0;
    size_t bestIdx = 0;
    float bestDist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < bank_.size(); i++) {
        float d = 0;
        for (int j = 0; j < featureDim_; j++) {
            float diff = query[j] - bank_[i].features[j];
            d += diff * diff;
        }
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    distOut = std::sqrt(bestDist);
    return bestIdx;
}

std::vector<float> MemoryBank::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries, int imgH, int imgW) const {
    int maxRow = 0, maxCol = 0;
    for (auto& q : queries) {
        maxRow = std::max(maxRow, q.patchRow + 1);
        maxCol = std::max(maxCol, q.patchCol + 1);
    }
    if (maxRow == 0 || maxCol == 0) return {};

    cv::Mat scoreMap(maxRow, maxCol, CV_32F);
    for (auto& q : queries) {
        float dist = 0;
        NearestNeighbor(q.features, dist);
        scoreMap.at<float>(q.patchRow, q.patchCol) = dist;
    }

    cv::Mat upsampled;
    cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);

    std::vector<float> result(imgW * imgH);
    std::copy(upsampled.begin<float>(), upsampled.end<float>(), result.begin());
    return result;
}

} // namespace aicore
