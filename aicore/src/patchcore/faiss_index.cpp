// ============================================================
// faiss_index.cpp — FAISS 索引封装实现
// ============================================================
#include "patchcore/faiss_index.h"
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/IDSelector.h>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace aicore {

// -------------------------------------------------------
// FaissIndex 析构 — unique_ptr 自动释放 index_
// -------------------------------------------------------
FaissIndex::~FaissIndex() = default;

// -------------------------------------------------------
// Train — 根据配置构建 FAISS 索引
// -------------------------------------------------------
Status FaissIndex::Train(const std::vector<PatchFeature>& features,
                         const FaissIndexConfig& cfg) {
    if (features.empty()) {
        return {StatusCode::ErrorInvalidInput, "FaissIndex: empty feature pool"};
    }

    // 检查维度一致性
    d_ = static_cast<int>(features[0].features.size());
    if (d_ == 0) {
        return {StatusCode::ErrorInvalidInput,
                "FaissIndex: zero-dimensional feature"};
    }
    // 确保 d_ 与 config 一致（如果 cfg.d 已设置）
    if (cfg.d != 0 && cfg.d != d_) {
        return {StatusCode::ErrorInvalidInput,
                "FaissIndex: config dimension mismatch"};
    }

    cfg_ = cfg;
    cfg_.d = d_;

    // 拼接特征矩阵: ntotal × d
    ntotal_ = features.size();
    std::vector<float> data(ntotal_ * d_);
    for (size_t i = 0; i < ntotal_; ++i) {
        const auto& f = features[i].features;
        if (static_cast<int>(f.size()) != d_) {
            return {StatusCode::ErrorInvalidInput,
                    "FaissIndex: inconsistent feature dimension in pool"};
        }
        std::memcpy(&data[i * d_], f.data(), d_ * sizeof(float));
    }

    try {
        switch (cfg_.algorithm) {
        case FaissSearchAlgorithm::BruteForce:
            index_ = std::make_unique<faiss::IndexFlatL2>(d_);
            index_->add(ntotal_, data.data());
            break;

        case FaissSearchAlgorithm::IVF: {
            auto quantizer = new faiss::IndexFlatL2(d_);
            auto ivf = new faiss::IndexIVFFlat(quantizer, d_, cfg_.nlist,
                                                faiss::METRIC_L2);
            ivf->train(ntotal_, data.data());
            ivf->add(ntotal_, data.data());
            ivf->nprobe = cfg_.nprobe;
            index_.reset(ivf);
            break;
        }

        case FaissSearchAlgorithm::HNSW: {
            auto hnsw = new faiss::IndexHNSWFlat(d_, cfg_.M);
            hnsw->hnsw.efConstruction = cfg_.efConstruction;
            hnsw->hnsw.efSearch = cfg_.efSearch;
            hnsw->train(ntotal_, data.data());
            hnsw->add(ntotal_, data.data());
            index_.reset(hnsw);
            break;
        }

        default:
            return {StatusCode::ErrorConfigParse,
                    "FaissIndex: unknown search algorithm"};
        }
    } catch (const std::exception& e) {
        std::string msg = "FaissIndex::Train failed: ";
        msg += e.what();
        return {StatusCode::ErrorInternal, msg};
    }

    is_trained_ = true;
    return Status{};
}

// -------------------------------------------------------
// Save — 序列化索引到文件
// -------------------------------------------------------
Status FaissIndex::Save(const std::string& path) const {
    if (!is_trained_ || !index_) {
        return {StatusCode::ErrorInternal,
                "FaissIndex::Save: index not trained"};
    }
    try {
        faiss::write_index(index_.get(), path.c_str());
    } catch (const std::exception& e) {
        std::string msg = "FaissIndex::Save failed: ";
        msg += e.what();
        return {StatusCode::ErrorInternal, msg};
    }
    return Status{};
}

// -------------------------------------------------------
// Load — 从文件加载索引
// -------------------------------------------------------
Status FaissIndex::Load(const std::string& path) {
    try {
        auto loaded = std::unique_ptr<faiss::Index>(
            faiss::read_index(path.c_str()));
        index_ = std::move(loaded);
        d_ = index_->d;
        ntotal_ = index_->ntotal;
        is_trained_ = true;

        // 尝试推断算法类型（从 index 类型）
        if (dynamic_cast<faiss::IndexFlat*>(index_.get())) {
            cfg_.algorithm = FaissSearchAlgorithm::BruteForce;
        } else if (dynamic_cast<faiss::IndexIVFFlat*>(index_.get())) {
            cfg_.algorithm = FaissSearchAlgorithm::IVF;
        } else if (dynamic_cast<faiss::IndexHNSWFlat*>(index_.get())) {
            cfg_.algorithm = FaissSearchAlgorithm::HNSW;
        }
    } catch (const std::exception& e) {
        std::string msg = "FaissIndex::Load failed: ";
        msg += e.what();
        return {StatusCode::ErrorInternal, msg};
    }
    return Status{};
}

// -------------------------------------------------------
// NearestNeighbor — 单查询向量 k=1 搜索
// -------------------------------------------------------
std::pair<size_t, float> FaissIndex::NearestNeighbor(
    const std::vector<float>& query) const {
    if (!is_trained_ || !index_) {
        return {0, std::numeric_limits<float>::max()};
    }

    idx_t idxOut = -1;
    float distOut = 0.0f;
    index_->search(1, query.data(), 1, &distOut, &idxOut);

    return {static_cast<size_t>(idxOut), distOut};
}

// -------------------------------------------------------
// SearchBatch — 批量搜索
// queries: M×d 扁平矩阵, numQueries: M
// 返回 M 个最近邻距离
// -------------------------------------------------------
std::vector<float> FaissIndex::SearchBatch(
    const std::vector<float>& queries, int numQueries) const {
    if (!is_trained_ || !index_ || numQueries <= 0) {
        return {};
    }

    std::vector<float> distances(numQueries);
    std::vector<idx_t> indices(numQueries);
    index_->search(numQueries, queries.data(), 1,
                   distances.data(), indices.data());
    return distances;
}

} // namespace aicore
