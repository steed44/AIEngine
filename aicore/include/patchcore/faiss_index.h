// ============================================================
// faiss_index.h — FAISS 近似最近邻搜索索引封装
// 功能：封装 FAISS 的 IndexFlatL2 / IndexIVFFlat / IndexHNSWFlat
//       提供统一的 Train / Save / Load / Search 接口
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"  // PatchFeature
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace aicore {

// -------------------------------------------------------
// FaissSearchAlgorithm — FAISS 索引算法枚举
// -------------------------------------------------------
enum class FaissSearchAlgorithm {
    BruteForce = 0,  // IndexFlatL2，精确搜索
    IVF,             // IndexIVFFlat，倒排索引近似搜索
    HNSW             // IndexHNSWFlat，图索引近似搜索
};

// -------------------------------------------------------
// FaissIndexConfig — FAISS 索引构建与搜索参数
// -------------------------------------------------------
struct FaissIndexConfig {
    FaissSearchAlgorithm algorithm = FaissSearchAlgorithm::BruteForce;
    int d = 0;       // 特征向量维度（必填，Train 前设置）

    // IVF 参数
    int nlist = 100;   // 聚类中心数（构建参数）
    int nprobe = 16;   // 搜索时探查簇数（精度↔速度权衡）

    // HNSW 参数
    int M = 16;               // 每层最大连接数
    int efConstruction = 200; // 构建时动态列表大小
    int efSearch = 64;        // 搜索时动态列表大小

    // 通用
    int gpuDevice = -1;   // -1=CPU, >=0=GPU 设备号
    uint32_t seed = 42;   // 随机种子（IVF k-means 可重现）
};

// -------------------------------------------------------
// FaissIndex — FAISS 索引封装
// 职责：管理 FAISS 索引对象的生命周期，提供训练/序列化/搜索接口
// -------------------------------------------------------
class FaissIndex {
public:
    ~FaissIndex();

    // 从 PatchFeature 列表训练并构建索引
    //   BruteForce → IndexFlatL2（无需 train，直接 add）
    //   IVF        → IndexIVFFlat（train + add）
    //   HNSW       → IndexHNSWFlat（无需 train，直接 add）
    Status Train(const std::vector<PatchFeature>& features,
                 const FaissIndexConfig& cfg);

    // 序列化保存索引到文件（FAISS 内置 binary 格式）
    Status Save(const std::string& path) const;

    // 从文件加载索引
    Status Load(const std::string& path);

    // k=1 最近邻搜索
    // 返回 {bestIdx, bestDist}
    std::pair<size_t, float> NearestNeighbor(
        const std::vector<float>& query) const;

    // 批量最近邻搜索
    // queries: M×d 扁平矩阵，numQueries: M
    // 返回 M 个最近邻距离
    std::vector<float> SearchBatch(
        const std::vector<float>& queries, int numQueries) const;

    // 访问器
    size_t Size() const { return ntotal_; }
    int Dimension() const { return d_; }
    FaissSearchAlgorithm Algorithm() const { return cfg_.algorithm; }
    bool IsTrained() const { return is_trained_; }

private:
    std::unique_ptr<faiss::Index> index_;  // FAISS 索引（多态）
    FaissIndexConfig cfg_;
    bool is_trained_ = false;
    size_t ntotal_ = 0;
    int d_ = 0;
};

} // namespace aicore
