#include "patchcore/faiss_index_bridge.h"
#include <fstream>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace aicore {

static std::string AlgorithmSuffix(FaissSearchAlgorithm algo) {
    switch (algo) {
    case FaissSearchAlgorithm::BruteForce: return ".flat.faiss";
    case FaissSearchAlgorithm::IVF:        return ".ivf.faiss";
    case FaissSearchAlgorithm::HNSW:       return ".hnsw.faiss";
    }
    return ".faiss";
}

static bool ReadFeatureRecord(std::ifstream& f, PatchFeature& pf, int dim) {
    pf.features.resize(dim);
    f.read(reinterpret_cast<char*>(pf.features.data()), dim * sizeof(float));
    f.read(reinterpret_cast<char*>(&pf.layerIdx), sizeof(pf.layerIdx));
    f.read(reinterpret_cast<char*>(&pf.patchRow), sizeof(pf.patchRow));
    f.read(reinterpret_cast<char*>(&pf.patchCol), sizeof(pf.patchCol));
    return f.good();
}

Status FaissIndexBridge::TrainFromMemoryBank(
    const std::string& memoryBankPath, const FaissIndexConfig& cfg) {
    binPath_ = memoryBankPath;

    std::ifstream f(memoryBankPath, std::ios::binary);
    if (!f) {
        return {StatusCode::ErrorModelLoad,
                "cannot open memory bank: " + memoryBankPath};
    }

    uint32_t magic = 0, num = 0, dim = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic == MemoryBank::kMagicNew) {
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != MemoryBank::kCurrentVersion) {
            return {StatusCode::ErrorModelLoad,
                    "unsupported version: " + std::to_string(version)};
        }
        f.read(reinterpret_cast<char*>(&num), sizeof(num));
        f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        uint8_t hasNorm = 0;
        f.read(reinterpret_cast<char*>(&hasNorm), sizeof(hasNorm));
        if (hasNorm) {
            f.seekg(dim * 2 * sizeof(float), std::ios::cur);
        }
    } else if (magic == MemoryBank::kMagic) {
        f.read(reinterpret_cast<char*>(&num), sizeof(num));
        f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    } else {
        return {StatusCode::ErrorModelLoad,
                "invalid magic in memory bank: " + memoryBankPath};
    }

    int featureDim = static_cast<int>(dim);
    std::vector<PatchFeature> features(num);
    for (uint32_t i = 0; i < num; ++i) {
        if (!ReadFeatureRecord(f, features[i], featureDim)) {
            return {StatusCode::ErrorModelLoad,
                    "truncated memory bank: " + memoryBankPath};
        }
    }

    Status st = index_.Train(features, cfg);
    if (!st) return st;

    // 保存索引供后续加速加载，失败非致命
    Status saveSt = index_.Save(memoryBankPath + AlgorithmSuffix(cfg.algorithm));
    (void)saveSt;

    return Status{};
}

Status FaissIndexBridge::LoadIndex(const std::string& indexPath) {
    return index_.Load(indexPath);
}

Status FaissIndexBridge::SaveIndex(const std::string& indexPath) const {
    return index_.Save(indexPath);
}

cv::Mat FaissIndexBridge::ComputeAnomalyMap(
    const std::vector<PatchFeature>& queries,
    int imgH, int imgW) const {
    if (!index_.IsTrained() || queries.empty()) {
        return cv::Mat(imgH, imgW, CV_32F, cv::Scalar(0));
    }

    int d = index_.Dimension();
    int numLayers = 0;
    for (auto& q : queries) {
        if (q.layerIdx + 1 > numLayers) numLayers = q.layerIdx + 1;
    }

    if (numLayers <= 1) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) {
            return cv::Mat(imgH, imgW, CV_32F, cv::Scalar(0));
        }

        int nq = static_cast<int>(queries.size());
        std::vector<float> flatQ(nq * d);
        for (int i = 0; i < nq; ++i) {
            std::memcpy(&flatQ[i * d], queries[i].features.data(),
                        d * sizeof(float));
        }

        std::vector<float> dists = index_.SearchBatch(flatQ, nq);

        cv::Mat scoreMap(maxRow, maxCol, CV_32F, cv::Scalar(0));
        for (int i = 0; i < nq; ++i) {
            scoreMap.at<float>(queries[i].patchRow, queries[i].patchCol) = dists[i];
        }

        cv::Mat upsampled;
        cv::resize(scoreMap, upsampled, cv::Size(imgW, imgH),
                   0, 0, cv::INTER_LINEAR);
        return upsampled;
    }

    // 多层融合：每层独立得分图 → 上采样 → 逐像素 max
    cv::Mat fused(imgH, imgW, CV_32F, cv::Scalar(0));

    for (int li = 0; li < numLayers; ++li) {
        int maxRow = 0, maxCol = 0;
        for (auto& q : queries) {
            if (q.layerIdx != li) continue;
            if (q.patchRow + 1 > maxRow) maxRow = q.patchRow + 1;
            if (q.patchCol + 1 > maxCol) maxCol = q.patchCol + 1;
        }
        if (maxRow == 0 || maxCol == 0) continue;

        std::vector<int> idx;
        for (int i = 0; i < static_cast<int>(queries.size()); ++i) {
            if (queries[i].layerIdx == li) idx.push_back(i);
        }

        int nq = static_cast<int>(idx.size());
        std::vector<float> flatQ(nq * d);
        for (int i = 0; i < nq; ++i) {
            std::memcpy(&flatQ[i * d], queries[idx[i]].features.data(),
                        d * sizeof(float));
        }

        std::vector<float> dists = index_.SearchBatch(flatQ, nq);

        cv::Mat layerMap(maxRow, maxCol, CV_32F, cv::Scalar(0));
        for (int i = 0; i < nq; ++i) {
            const auto& q = queries[idx[i]];
            layerMap.at<float>(q.patchRow, q.patchCol) = dists[i];
        }

        cv::Mat up;
        cv::resize(layerMap, up, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);
        cv::max(fused, up, fused);
    }

    return fused;
}

} // namespace aicore
