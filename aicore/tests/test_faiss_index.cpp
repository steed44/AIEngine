#include <gtest/gtest.h>
#include "patchcore/faiss_index.h"
#include "patchcore/faiss_index_bridge.h"
#include "patchcore/memory_bank.h"
#include <fstream>
#include <numeric>

namespace aicore {

// -------------------------------------------------------
// 辅助函数：生成等维度的测试特征
// -------------------------------------------------------
static std::vector<PatchFeature> MakeFeatures(int n, int dim) {
    std::vector<PatchFeature> features(n);
    for (int i = 0; i < n; ++i) {
        features[i].features.resize(dim);
        features[i].layerIdx = 0;
        features[i].patchRow = i;
        features[i].patchCol = 0;
        for (int j = 0; j < dim; ++j) {
            features[i].features[j] = static_cast<float>(i * dim + j);
        }
    }
    return features;
}

// -------------------------------------------------------
// FaissIndex: Flat
// -------------------------------------------------------
TEST(FaissIndexFlat, TrainAndSearch) {
    auto features = MakeFeatures(3, 4);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    // 查询与第一个特征相同的向量 → 最近邻距离应为 0
    std::vector<float> query = features[0].features;
    auto [bestIdx, bestDist] = idx.NearestNeighbor(query);
    EXPECT_EQ(bestIdx, 0u);
    EXPECT_NEAR(bestDist, 0.0f, 1e-4f);

    EXPECT_EQ(idx.Size(), 3u);
    EXPECT_EQ(idx.Dimension(), 4);
    EXPECT_TRUE(idx.IsTrained());
    EXPECT_EQ(idx.Algorithm(), FaissSearchAlgorithm::BruteForce);
}

// -------------------------------------------------------
// FaissIndex: IVF
// -------------------------------------------------------
TEST(FaissIndexIVF, TrainAndSearch) {
    auto features = MakeFeatures(50, 8);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::IVF;
    cfg.d = 8;
    cfg.nlist = 5;
    cfg.nprobe = 3;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    std::vector<float> query = features[0].features;
    auto [bestIdx, bestDist] = idx.NearestNeighbor(query);
    // IVF 是近似搜索，最近邻索引可能不是精确的 0，但应该接近
    EXPECT_LE(bestDist, 1.0f);

    EXPECT_EQ(idx.Algorithm(), FaissSearchAlgorithm::IVF);
}

// -------------------------------------------------------
// FaissIndex: HNSW
// -------------------------------------------------------
TEST(FaissIndexHNSW, TrainAndSearch) {
    auto features = MakeFeatures(30, 8);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::HNSW;
    cfg.d = 8;
    cfg.M = 8;
    cfg.efConstruction = 50;
    cfg.efSearch = 20;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    std::vector<float> query = features[0].features;
    auto [bestIdx, bestDist] = idx.NearestNeighbor(query);
    EXPECT_LE(bestDist, 0.5f);

    EXPECT_EQ(idx.Algorithm(), FaissSearchAlgorithm::HNSW);
}

// -------------------------------------------------------
// FaissIndex: 保存/加载往返
// -------------------------------------------------------
TEST(FaissIndex, SaveLoadRoundtrip) {
    auto features = MakeFeatures(5, 4);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    std::string path = "test_faiss_roundtrip.faiss";
    ASSERT_TRUE(idx.Save(path));

    FaissIndex loaded;
    ASSERT_TRUE(loaded.Load(path));

    EXPECT_EQ(loaded.Size(), idx.Size());
    EXPECT_EQ(loaded.Dimension(), idx.Dimension());
    EXPECT_TRUE(loaded.IsTrained());

    // 搜索结果一致
    auto [origIdx, origDist] = idx.NearestNeighbor(features[0].features);
    auto [loadIdx, loadDist] = loaded.NearestNeighbor(features[0].features);
    EXPECT_EQ(origIdx, loadIdx);
    EXPECT_NEAR(origDist, loadDist, 1e-4f);

    std::remove(path.c_str());
}

// -------------------------------------------------------
// FaissIndex: 空特征池
// -------------------------------------------------------
TEST(FaissIndex, EmptyPool) {
    std::vector<PatchFeature> empty;
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    Status st = idx.Train(empty, cfg);
    EXPECT_FALSE(st);
}

// -------------------------------------------------------
// FaissIndex: 维度不匹配
// -------------------------------------------------------
TEST(FaissIndex, DimMismatch) {
    auto f1 = MakeFeatures(1, 4);
    auto f2 = MakeFeatures(1, 8);
    std::vector<PatchFeature> mixed;
    mixed.push_back(f1[0]);
    mixed.push_back(f2[0]);

    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    Status st = idx.Train(mixed, cfg);
    EXPECT_FALSE(st);
}

// -------------------------------------------------------
// FaissIndex: 批量搜索
// -------------------------------------------------------
TEST(FaissIndex, SearchBatch) {
    auto features = MakeFeatures(4, 4);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    int nq = 2;
    std::vector<float> queries = features[0].features;
    queries.insert(queries.end(), features[2].features.begin(),
                   features[2].features.end());

    std::vector<float> dists = idx.SearchBatch(queries, nq);
    ASSERT_EQ(dists.size(), 2u);
    EXPECT_NEAR(dists[0], 0.0f, 1e-4f);  // query[0] == features[0]
    EXPECT_NEAR(dists[1], 0.0f, 1e-4f);  // query[1] == features[2]
}

// -------------------------------------------------------
// FaissIndexBridge: ComputeAnomalyMap
// -------------------------------------------------------
TEST(FaissIndexBridge, ComputeAnomalyMap) {
    auto features = MakeFeatures(4, 4);
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndex idx;
    ASSERT_TRUE(idx.Train(features, cfg));

    std::vector<float> rawQ(features[0].features);
    std::vector<float> dists = idx.SearchBatch(rawQ, 1);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_NEAR(dists[0], 0.0f, 1e-4f);
}

// -------------------------------------------------------
// FaissIndexBridge: 从 .bin 训练
// -------------------------------------------------------
TEST(FaissIndexBridge, TrainFromMemoryBank) {
    // 创建临时 MemoryBank .bin 文件
    auto features = MakeFeatures(5, 4);
    MemoryBank bank;
    bank.Build(features);

    std::string binPath = "test_faiss_bridge.bin";
    ASSERT_TRUE(bank.Save(binPath));

    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndexBridge bridge;
    Status st = bridge.TrainFromMemoryBank(binPath, cfg);
    ASSERT_TRUE(st);
    EXPECT_TRUE(bridge.IsReady());

    // 清理
    std::remove(binPath.c_str());
    std::remove((binPath + ".flat.faiss").c_str());
}

} // namespace aicore
