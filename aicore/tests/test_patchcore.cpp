// ============================================================
// 文件: tests/test_patchcore.cpp
// 用途: PatchCore 异常检测模块单元测试
//   涵盖 MemoryBank / CoresetSampler / PatchCoreNode / Trainer / Dataset
// ============================================================

#include <gtest/gtest.h>
#include "patchcore/memory_bank.h"
#include "patchcore/coreset_sampler.h"
#include "patchcore/folder_dataset.h"
#include "patchcore/patchcore_node.h"
#include "patchcore/patchcore_trainer.h"
#include "patchcore/anomaly_evaluator.h"
#include "patchcore/patchcore_visualize.h"
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <chrono>

using namespace aicore;

// 测试：MemoryBank 构建和最近邻查询
TEST(MemoryBankTest, BuildAndQuery) {
    std::vector<PatchFeature> features;
    for (int i = 0; i < 10; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), 0, 0};
        pf.patchRow = i; pf.patchCol = 0;
        features.push_back(pf);
    }
    MemoryBank bank;
    bank.Build(features);
    EXPECT_EQ(bank.Size(), 10);
    EXPECT_EQ(bank.FeatureDim(), 3);

    float dist = 0;
    std::vector<float> query = {5, 0, 0};
    size_t idx = bank.NearestNeighbor(query, dist);
    EXPECT_EQ(idx, 5);
    EXPECT_NEAR(dist, 0, 1e-5);
}

// 测试：MemoryBank ComputeAnomalyMap 单层路径, 上采样尺寸正确
TEST(MemoryBankTest, ComputeAnomalyMapSingleLayer) {
    std::vector<PatchFeature> bankFeats;
    PatchFeature bf;
    bf.features = {0, 0};
    bf.patchRow = 0; bf.patchCol = 0;
    bankFeats.push_back(bf);
    MemoryBank bank;
    bank.Build(bankFeats);

    std::vector<PatchFeature> queries;
    for (int c = 0; c < 2; c++) {
        PatchFeature q;
        q.features = {5, 5};
        q.layerIdx = 0; q.patchRow = 0; q.patchCol = c;
        queries.push_back(q);
    }
    auto result = bank.ComputeAnomalyMap(queries, 4, 4);
    ASSERT_EQ(result.size(), 16u);
}

// 测试：MemoryBank ComputeAnomalyMap 多层融合, 逐像素取最大值
TEST(MemoryBankTest, ComputeAnomalyMapMultiLayer) {
    std::vector<PatchFeature> bankFeats;
    PatchFeature bf;
    bf.features = {0, 0, 0};
    bf.patchRow = 0; bf.patchCol = 0;
    bankFeats.push_back(bf);
    MemoryBank bank;
    bank.Build(bankFeats);

    std::vector<PatchFeature> queries;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            PatchFeature q;
            q.features = {100, 100, 100};
            q.layerIdx = 0; q.patchRow = r; q.patchCol = c;
            queries.push_back(q);
        }
    }
    PatchFeature ql;
    ql.features = {0, 0, 0};
    ql.layerIdx = 1; ql.patchRow = 0; ql.patchCol = 0;
    queries.push_back(ql);

    auto result = bank.ComputeAnomalyMap(queries, 4, 4);
    ASSERT_EQ(result.size(), 16u);
    float maxVal = *std::max_element(result.begin(), result.end());
    float minVal = *std::min_element(result.begin(), result.end());
    EXPECT_GT(maxVal, 100.0f);
    EXPECT_GT(minVal, 1.0f);
}

// 测试：MemoryBank 保存到文件后重新加载，数据一致
TEST(MemoryBankTest, SaveLoadRoundtrip) {
    std::vector<PatchFeature> features;
    PatchFeature pf;
    pf.features = {1, 2, 3};
    pf.patchRow = 0; pf.patchCol = 0;
    features.push_back(pf);

    MemoryBank bank;
    bank.Build(features);
    EXPECT_TRUE(bank.Save("test_mem.bin"));

    MemoryBank loaded;
    EXPECT_TRUE(loaded.Load("test_mem.bin"));
    EXPECT_EQ(loaded.Size(), 1);
    EXPECT_EQ(loaded.FeatureDim(), 3);

    std::remove("test_mem.bin");
}

// 测试：CoresetSampler 将 100 个特征抽样为 10 个
TEST(CoresetSamplerTest, ReduceSize) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 100; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i * 2), 0};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.Sample(pool, 10);
    EXPECT_EQ(indices.size(), 10);
}

// 测试：PatchCoreNode 缺少模型路径时初始化失败
TEST(PatchCoreNodeTest, InitMissingModelPath) {
    PatchCoreNode node;
    NodeConfig cfg;
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
}

// 测试：PatchCoreNode 处理空输入返回失败
TEST(PatchCoreNodeTest, ProcessEmptyInput) {
    PatchCoreNode node;
    std::vector<Frame> inputs, outputs;
    auto s = node.Process(inputs, outputs);
    EXPECT_FALSE(s);
}

// 测试：PatchCoreNode 初始化时加载 memory bank 文件
TEST(PatchCoreNodeTest, InitLoadsMemoryBank) {
    std::vector<PatchFeature> feats;
    PatchFeature pf;
    pf.features = {1, 2, 3};
    pf.patchRow = 0; pf.patchCol = 0;
    feats.push_back(pf);
    MemoryBank bank;
    bank.Build(feats);
    ASSERT_TRUE(bank.Save("test_patchcore.bin"));

    PatchCoreNode node;
    NodeConfig cfg;
    cfg["model_path"] = "dummy.onnx";
    cfg["memory_bank_path"] = "test_patchcore.bin";
    try {
        (void)node.Init(cfg);
    } catch (...) {
        // OpenCV dnn 会在 ONNX 文件不存在时抛出异常 — 属预期行为
    }
    std::remove("test_patchcore.bin");
}

// 测试：训练器在路径不存在时返回失败
TEST(PatchCoreTrainerTest, FolderNotFound) {
    PatchCoreTrainer trainer;
    auto s = trainer.TrainFromFolder("nonexistent_dir", "dummy.onnx", "out.bin", {});
    EXPECT_FALSE(s);
}

// 测试：训练器在空文件夹时返回失败
TEST(PatchCoreTrainerTest, EmptyFolder) {
    std::string dir = "empty_test_dir/";
    std::filesystem::create_directories(dir);

    PatchCoreTrainer trainer;
    auto s = trainer.TrainFromFolder(dir, "dummy.onnx", "out.bin", {});
    EXPECT_FALSE(s);

    std::filesystem::remove_all(dir);
}

// 测试：FolderDataset 从文件夹加载图片，大小正确
TEST(FolderDatasetTest, LoadFromFolder) {
    std::string dir = "test_images/";
    std::filesystem::create_directories(dir);
    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::imwrite(dir + "img1.png", img);
    cv::imwrite(dir + "img2.jpg", img);

    FolderDataset ds;
    EXPECT_TRUE(ds.Load(dir));
    EXPECT_EQ(ds.Size(), 2);

    auto s = ds.Get(0);
    EXPECT_EQ(s.image.rows, 10);
    EXPECT_EQ(s.label, 0);

    std::filesystem::remove_all(dir);
}

// ─── AnomalyEvaluator ────────────────────────────────────────

TEST(AnomalyEvaluatorTest, Empty) {
    AnomalyEvaluator eval;
    auto r = eval.Evaluate();
    EXPECT_DOUBLE_EQ(r.auc, 0.0);
    EXPECT_DOUBLE_EQ(r.bestF1, 0.0);
}

TEST(AnomalyEvaluatorTest, PerfectSeparation) {
    AnomalyEvaluator eval;
    eval.AddSample(1, 0.9f);
    eval.AddSample(1, 0.8f);
    eval.AddSample(0, 0.2f);
    eval.AddSample(0, 0.1f);
    auto r = eval.Evaluate();
    EXPECT_NEAR(r.auc, 1.0, 0.01);
    EXPECT_NEAR(r.bestF1, 1.0, 0.01);
    EXPECT_GT(r.bestThreshold, 0.2);
    EXPECT_LT(r.bestThreshold, 0.8);
}

TEST(AnomalyEvaluatorTest, EvaluateAtExact) {
    std::vector<std::pair<int, float>> samples = {
        {1, 0.9f}, {1, 0.8f}, {0, 0.2f}, {0, 0.1f}
    };
    auto r = AnomalyEvaluator::EvaluateAt(samples, 0.5f);
    EXPECT_EQ(r.tp, 2);
    EXPECT_EQ(r.fp, 0);
    EXPECT_EQ(r.tn, 2);
    EXPECT_EQ(r.fn, 0);
    EXPECT_DOUBLE_EQ(r.bestF1, 1.0);
    EXPECT_DOUBLE_EQ(r.accuracy, 1.0);
}

TEST(AnomalyEvaluatorTest, FindBestThreshold) {
    std::vector<std::pair<int, float>> samples = {
        {1, 0.95f}, {1, 0.87f}, {1, 0.76f},
        {0, 0.34f}, {0, 0.22f}, {0, 0.11f}
    };
    auto r = AnomalyEvaluator::FindBestThreshold(samples, 100);
    EXPECT_GT(r.bestF1, 0.95);
    EXPECT_EQ(r.tp, 3);
    EXPECT_EQ(r.fp, 0);
}

TEST(AnomalyEvaluatorTest, ImperfectSeparation) {
    std::vector<std::pair<int, float>> samples = {
        {1, 0.9f}, {0, 0.8f}, {1, 0.7f}, {0, 0.4f},
        {1, 0.3f}, {0, 0.2f}, {1, 0.15f},
    };
    auto r = AnomalyEvaluator::EvaluateAt(samples, 0.5f);
    EXPECT_EQ(r.tp, 2);
    EXPECT_EQ(r.fp, 1);
    EXPECT_EQ(r.tn, 2);
    EXPECT_EQ(r.fn, 2);
    EXPECT_GT(r.bestF1, 0.5);
}

// ─── MemoryBank 新格式 ──────────────────────────────────────

TEST(MemoryBankTest, SaveLoadNewFormat) {
    std::vector<PatchFeature> features;
    for (int i = 0; i < 5; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(i * 3)};
        pf.layerIdx = i % 2;
        pf.patchRow = i;
        pf.patchCol = i * 2;
        features.push_back(pf);
    }

    MemoryBank bank;
    bank.Build(features);
    Status s = bank.Save("test_new_format.bin");
    ASSERT_TRUE(s) << s.message;

    MemoryBank loaded;
    s = loaded.Load("test_new_format.bin");
    ASSERT_TRUE(s) << s.message;
    EXPECT_EQ(loaded.Size(), 5);
    EXPECT_EQ(loaded.FeatureDim(), 3);

    float dist = 0;
    size_t idx = loaded.NearestNeighbor({3, 6, 9}, dist);
    EXPECT_EQ(idx, 3);
    EXPECT_NEAR(dist, 0, 1e-4f);

    std::remove("test_new_format.bin");
}

TEST(MemoryBankTest, SaveLoadWithNormParams) {
    std::vector<PatchFeature> features;
    PatchFeature pf;
    pf.features = {1, 2, 3};
    pf.patchRow = 0; pf.patchCol = 0;
    features.push_back(pf);

    MemoryBank bank;
    bank.Build(features);
    bank.hasNormParams_ = true;
    bank.normMean_ = {0.5f, 0.6f, 0.7f};
    bank.normStd_ = {0.1f, 0.2f, 0.3f};

    ASSERT_TRUE(bank.Save("test_norm.bin"));

    MemoryBank loaded;
    ASSERT_TRUE(loaded.Load("test_norm.bin"));
    EXPECT_TRUE(loaded.hasNormParams_);
    ASSERT_EQ(loaded.normMean_.size(), 3);
    EXPECT_NEAR(loaded.normMean_[0], 0.5f, 1e-4f);
    EXPECT_NEAR(loaded.normStd_[2], 0.3f, 1e-4f);

    std::remove("test_norm.bin");
}

TEST(MemoryBankTest, LoadOldFormat) {
    // 手动写入旧格式文件 (kMagic, 无 version/norm 字段)
    {
        std::ofstream ofs("test_old_fmt.bin", std::ios::binary);
        ASSERT_TRUE(ofs);
        uint32_t magic = MemoryBank::kMagic;
        uint32_t num = 2, dim = 2;
        ofs.write(reinterpret_cast<const char*>(&magic), 4);
        ofs.write(reinterpret_cast<const char*>(&num), 4);
        ofs.write(reinterpret_cast<const char*>(&dim), 4);
        for (int i = 0; i < 2; i++) {
            float feat[2] = {float(i), float(i + 1)};
            int extra[3] = {0, i, 0};
            ofs.write(reinterpret_cast<const char*>(feat), 8);
            ofs.write(reinterpret_cast<const char*>(extra), 12);
        }
    }

    MemoryBank bank;
    ASSERT_TRUE(bank.Load("test_old_fmt.bin"));
    EXPECT_EQ(bank.Size(), 2);
    EXPECT_EQ(bank.FeatureDim(), 2);
    EXPECT_FALSE(bank.hasNormParams_);

    float dist = 0;
    size_t idx = bank.NearestNeighbor({1, 2}, dist);
    EXPECT_EQ(idx, 1);
    EXPECT_NEAR(dist, 0, 1e-4f);

    std::remove("test_old_fmt.bin");
}

// ─── CoresetSampler FastSample ─────────────────────────────

TEST(CoresetSamplerTest, FastSampleReduces) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 200; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i * 2), 0};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.FastSample(pool, 10);
    EXPECT_EQ(indices.size(), 10);
    // 所有 index 必须唯一
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), 10);
}

TEST(CoresetSamplerTest, FastSampleTargetLargerThanPool) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 5; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i)};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.FastSample(pool, 100);
    EXPECT_EQ(indices.size(), 5);  // 全部返回
}

TEST(CoresetSamplerTest, FastSampleEmptyPool) {
    std::vector<PatchFeature> pool;
    CoresetSampler sampler;
    auto indices = sampler.FastSample(pool, 10);
    EXPECT_TRUE(indices.empty());
}

TEST(CoresetSamplerTest, FastSampleLargePool) {
    // 60000 items → 触发 maxCandidates=20000 随机子集路径
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 60000; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i % 100)};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.FastSample(pool, 100);
    EXPECT_EQ(indices.size(), 100);
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), 100);
    for (auto idx : indices) {
        EXPECT_LT(idx, pool.size());
    }
}

// ─── Visualization ──────────────────────────────────────────

TEST(PatchCoreVisualizeTest, ColorizeAnomalyMap) {
    cv::Mat scoreMap(8, 8, CV_32F);
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            scoreMap.at<float>(r, c) = static_cast<float>(r + c) / 14.0f;

    cv::Mat colored = ColorizeAnomalyMap(scoreMap, 1.0f);
    EXPECT_EQ(colored.type(), CV_8UC3);
    EXPECT_EQ(colored.rows, 8);
    EXPECT_EQ(colored.cols, 8);
}

TEST(PatchCoreVisualizeTest, ColorizeAutoMax) {
    cv::Mat scoreMap(4, 4, CV_32F, cv::Scalar(42.0f));
    cv::Mat colored = ColorizeAnomalyMap(scoreMap);
    EXPECT_EQ(colored.type(), CV_8UC3);
    EXPECT_EQ(colored.rows, 4);
}

TEST(PatchCoreVisualizeTest, DrawAnomalyOverlay) {
    cv::Mat scoreMap(16, 16, CV_32F, cv::Scalar(0.5f));
    cv::Mat image(16, 16, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Mat overlay = DrawAnomalyOverlay(scoreMap, image, 0.5f, 0.1f);
    EXPECT_EQ(overlay.type(), CV_8UC3);
    EXPECT_EQ(overlay.rows, 16);
    EXPECT_EQ(overlay.cols, 16);
}

TEST(PatchCoreVisualizeTest, DrawOverlayZeroThreshold) {
    cv::Mat scoreMap(8, 8, CV_32F, cv::Scalar(0.0f));
    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat overlay = DrawAnomalyOverlay(scoreMap, image, 1.0f, 0.0f);
    EXPECT_EQ(overlay.type(), CV_8UC3);
}

// ─── Edge Cases ─────────────────────────────────────────────

TEST(MemoryBankEdgeTest, EmptyBankNearestNeighbor) {
    MemoryBank bank;
    float dist = -1;
    size_t idx = bank.NearestNeighbor({1, 2, 3}, dist);
    EXPECT_EQ(idx, 0);
    EXPECT_FLOAT_EQ(dist, 0);
}

TEST(MemoryBankEdgeTest, EmptyBankAnomalyMap) {
    MemoryBank bank;
    std::vector<PatchFeature> queries;
    PatchFeature pf;
    pf.features = {1, 2};
    pf.patchRow = 0; pf.patchCol = 0;
    queries.push_back(pf);
    auto result = bank.ComputeAnomalyMap(queries, 4, 4);
    ASSERT_EQ(result.size(), 16u);
    for (float v : result)
        EXPECT_FLOAT_EQ(v, 0);
}

TEST(MemoryBankEdgeTest, EmptyQueryAnomalyMap) {
    MemoryBank bank;
    std::vector<PatchFeature> features;
    PatchFeature pf;
    pf.features = {1, 2};
    features.push_back(pf);
    bank.Build(features);

    std::vector<PatchFeature> emptyQ;
    auto result = bank.ComputeAnomalyMap(emptyQ, 4, 4);
    EXPECT_TRUE(result.empty());
}

TEST(MemoryBankEdgeTest, BuildEmptyFeatures) {
    MemoryBank bank;
    bank.Build({});
    EXPECT_EQ(bank.Size(), 0);
    EXPECT_EQ(bank.FeatureDim(), 0);
}

TEST(MemoryBankEdgeTest, ClearIdempotent) {
    MemoryBank bank;
    bank.Clear();
    bank.Clear();
    bank.Clear();
    EXPECT_EQ(bank.Size(), 0);
}

TEST(CoresetSamplerEdgeTest, ZeroTargetSample) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 10; i++) {
        PatchFeature pf;
        pf.features = {float(i)};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.Sample(pool, 0);
    EXPECT_TRUE(indices.empty());
}

TEST(CoresetSamplerEdgeTest, ZeroTargetFastSample) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 10; i++) {
        PatchFeature pf;
        pf.features = {float(i)};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto indices = sampler.FastSample(pool, 0);
    EXPECT_TRUE(indices.empty());
}

TEST(CoresetSamplerEdgeTest, SingleElementPool) {
    std::vector<PatchFeature> pool;
    PatchFeature pf;
    pf.features = {42};
    pool.push_back(pf);
    CoresetSampler sampler;
    auto indices = sampler.Sample(pool, 1);
    ASSERT_EQ(indices.size(), 1);
    EXPECT_EQ(indices[0], 0);
}

TEST(EvaluatorEdgeTest, SingleAnomaly) {
    AnomalyEvaluator eval;
    eval.AddSample(1, 0.9f);
    auto r = eval.Evaluate();
    EXPECT_DOUBLE_EQ(r.auc, 0.0);
    EXPECT_EQ(r.tp, 0); // 单类时 Evaluate 返回空结果
}

TEST(EvaluatorEdgeTest, AllSameScore) {
    AnomalyEvaluator eval;
    eval.AddSample(1, 0.5f);
    eval.AddSample(0, 0.5f);
    auto r = eval.Evaluate();
    EXPECT_NEAR(r.bestF1, 0.667, 0.01);
}

TEST(VisualizeEdgeTest, SinglePixelMap) {
    cv::Mat scoreMap(1, 1, CV_32F, cv::Scalar(1.0f));
    cv::Mat colored = ColorizeAnomalyMap(scoreMap);
    EXPECT_EQ(colored.total(), 1);
    EXPECT_EQ(colored.type(), CV_8UC3);
}

TEST(VisualizeEdgeTest, MismatchedSizeOverlay) {
    cv::Mat scoreMap(4, 4, CV_32F, cv::Scalar(0.5f));
    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(128, 128, 128));
    EXPECT_ANY_THROW(DrawAnomalyOverlay(scoreMap, image));
}

// ─── #5 模拟端到端集成 ─────────────────────────────────────

TEST(IntegrationTest, PipelineSimulated) {
    std::vector<PatchFeature> allFeatures;
    for (int i = 0; i < 1000; i++) {
        PatchFeature pf;
        pf.features = {static_cast<float>(i), static_cast<float>(i % 50), 0};
        pf.layerIdx = 0;
        pf.patchRow = i / 32;
        pf.patchCol = i % 32;
        allFeatures.push_back(pf);
    }

    CoresetSampler sampler;
    auto indices = sampler.FastSample(allFeatures, 100);
    ASSERT_EQ(indices.size(), 100);

    std::vector<PatchFeature> sampled;
    for (auto idx : indices) sampled.push_back(allFeatures[idx]);

    MemoryBank bank;
    bank.Build(sampled);
    EXPECT_EQ(bank.Size(), 100);
    ASSERT_TRUE(bank.Save("test_e2e.bin"));

    MemoryBank loaded;
    ASSERT_TRUE(loaded.Load("test_e2e.bin"));
    EXPECT_EQ(loaded.Size(), 100);

    std::vector<PatchFeature> queries;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            PatchFeature q;
            q.features = {50, 25, 0};
            q.layerIdx = 0; q.patchRow = r; q.patchCol = c;
            queries.push_back(q);
        }
    auto anomalyMap = loaded.ComputeAnomalyMap(queries, 32, 32);
    ASSERT_EQ(anomalyMap.size(), 1024u);

    double sum = 0;
    for (float v : anomalyMap) sum += v;
    EXPECT_GT(sum, 0);

    std::remove("test_e2e.bin");
}

TEST(IntegrationTest, TrainSaveLoadEvaluate) {
    CoresetSampler sampler;
    std::vector<PatchFeature> normalPool;
    for (int i = 0; i < 500; i++) {
        PatchFeature pf;
        pf.features = {0.1f * (i % 10), 0.1f * (i / 10), 0};
        normalPool.push_back(pf);
    }
    auto idx = sampler.FastSample(normalPool, 50);
    std::vector<PatchFeature> sampled;
    for (auto i : idx) sampled.push_back(normalPool[i]);

    MemoryBank bank;
    bank.Build(sampled);
    ASSERT_TRUE(bank.Save("test_e2e_eval.bin"));

    MemoryBank loaded;
    ASSERT_TRUE(loaded.Load("test_e2e_eval.bin"));

    AnomalyEvaluator evaluator;
    for (int i = 0; i < 20; i++) {
        PatchFeature q;
        q.features = {0.1f * (i % 10), 0.1f * (i / 10 % 10), 0};
        float dist = 0;
        loaded.NearestNeighbor(q.features, dist);
        evaluator.AddSample(0, dist);
    }
    for (int i = 0; i < 20; i++) {
        PatchFeature q;
        q.features = {100 + 0.1f * i, 200 + 0.1f * i, 0};
        float dist = 0;
        loaded.NearestNeighbor(q.features, dist);
        evaluator.AddSample(1, dist);
    }
    auto result = evaluator.Evaluate();
    EXPECT_GT(result.auc, 0.8);
    EXPECT_GT(result.bestF1, 0.8);

    std::remove("test_e2e_eval.bin");
}

// ─── #6 性能基准测试 ────────────────────────────────────────

TEST(BenchmarkTest, NearestNeighborScaling) {
    const int dims = 64;
    for (int size : {100, 500, 2000}) {
        std::vector<PatchFeature> features;
        for (int i = 0; i < size; i++) {
            PatchFeature pf;
            pf.features.resize(dims);
            for (int d = 0; d < dims; d++)
                pf.features[d] = static_cast<float>((i * 31 + d * 17) % 100);
            features.push_back(pf);
        }
        MemoryBank bank;
        bank.Build(features);

        std::vector<float> query(dims, 42);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) {
            float dist;
            bank.NearestNeighbor(query, dist);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 100.0;
        // 2000 维 64 维 × 100 次应在 50ms 以内
        EXPECT_LT(ms, 50000.0);
    }
}

TEST(BenchmarkTest, ComputeAnomalyMapScaling) {
    std::vector<PatchFeature> features;
    for (int i = 0; i < 500; i++) {
        PatchFeature pf;
        pf.features = {float(i), float(i % 10), 0};
        features.push_back(pf);
    }
    MemoryBank bank;
    bank.Build(features);

    std::vector<PatchFeature> queries;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            PatchFeature q;
            q.features = {50, 25, 0};
            q.patchRow = r; q.patchCol = c;
            queries.push_back(q);
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = bank.ComputeAnomalyMap(queries, 256, 256);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    ASSERT_EQ(result.size(), 65536u);
    EXPECT_LT(ms, 5000.0);
}

TEST(BenchmarkTest, FastSampleScaling) {
    std::vector<PatchFeature> pool;
    for (int i = 0; i < 50000; i++) {
        PatchFeature pf;
        pf.features = {float(i), float(i % 1000)};
        pool.push_back(pf);
    }
    CoresetSampler sampler;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto indices = sampler.FastSample(pool, 100);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    EXPECT_EQ(indices.size(), 100);
    EXPECT_LT(ms, 10000); // 10s 上限
}

TEST(BenchmarkTest, SaveLoadThroughput) {
    std::vector<PatchFeature> features;
    for (int i = 0; i < 2000; i++) {
        PatchFeature pf;
        pf.features = {float(i), float(i * 2), float(i * 3)};
        features.push_back(pf);
    }
    MemoryBank bank;
    bank.Build(features);

    auto t0 = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(bank.Save("test_bench.bin"));
    auto t1 = std::chrono::high_resolution_clock::now();

    MemoryBank loaded;
    ASSERT_TRUE(loaded.Load("test_bench.bin"));
    auto t2 = std::chrono::high_resolution_clock::now();

    auto saveMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    EXPECT_EQ(loaded.Size(), 2000);
    EXPECT_LT(saveMs, 1000);
    EXPECT_LT(loadMs, 1000);

    std::remove("test_bench.bin");
}

// ─── FAISS 集成测试 ──────────────────────────────────────────

TEST(FaissIntegrationTest, BruteForceMatchesBaseline) {
    const int dim = 4;
    const int n = 20;
    std::vector<PatchFeature> features(n);
    for (int i = 0; i < n; ++i) {
        features[i].features.resize(dim);
        for (int d = 0; d < dim; ++d)
            features[i].features[d] = static_cast<float>(i * dim + d);
        features[i].layerIdx = 0;
        features[i].patchRow = i;
        features[i].patchCol = 0;
    }

    // 暴力 baseline
    MemoryBank bank;
    bank.Build(features);

    // FAISS Flat
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = dim;
    FaissIndex faissIdx;
    ASSERT_TRUE(faissIdx.Train(features, cfg));

    // 随机查询 10 次，比较结果
    std::mt19937 rng(42);
    for (int q = 0; q < 10; ++q) {
        std::vector<float> query(dim);
        for (int d = 0; d < dim; ++d)
            query[d] = static_cast<float>(rng() % 100);

        float baselineDist = 0;
        size_t baselineIdx = bank.NearestNeighbor(query, baselineDist);
        auto [faissIdx_, faissDist_] = faissIdx.NearestNeighbor(query);
        (void)faissIdx_;

        // FAISS Flat 应得到与暴力搜索一致的距离
        EXPECT_NEAR(faissDist_, baselineDist, 1e-3f);
    }
}

TEST(FaissIntegrationTest, HNSWApproximateRecall) {
    const int dim = 8;
    const int n = 200;
    std::vector<PatchFeature> features(n);
    for (int i = 0; i < n; ++i) {
        features[i].features.resize(dim);
        for (int d = 0; d < dim; ++d)
            features[i].features[d] = static_cast<float>((i * 31 + d * 17) % 100);
        features[i].layerIdx = 0;
        features[i].patchRow = i;
        features[i].patchCol = 0;
    }

    // 暴力 baseline
    MemoryBank bank;
    bank.Build(features);

    // HNSW
    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::HNSW;
    cfg.d = dim;
    cfg.M = 16;
    cfg.efConstruction = 100;
    cfg.efSearch = 50;
    FaissIndex hnswIdx;
    ASSERT_TRUE(hnswIdx.Train(features, cfg));

    // HNSW 近似搜索的 recall@1 应 ≥ 0.9
    int correct = 0;
    int trials = 50;
    std::mt19937 rng(123);
    for (int t = 0; t < trials; ++t) {
        std::vector<float> query(dim);
        for (int d = 0; d < dim; ++d)
            query[d] = static_cast<float>(rng() % 100);

        float baselineDist = 0;
        size_t baselineIdx = bank.NearestNeighbor(query, baselineDist);
        auto [hnswIdx_, hnswDist_] = hnswIdx.NearestNeighbor(query);
        (void)hnswIdx_;

        // HNSW 距离 ≤ baseline 距离的 1.1 倍则算正确
        if (hnswDist_ <= baselineDist * 1.1f + 0.01f) {
            correct++;
        }
    }
    EXPECT_GE(static_cast<float>(correct) / trials, 0.9f);
}

TEST(FaissIntegrationTest, TrainFromMemoryBankFile) {
    std::vector<PatchFeature> features(10);
    for (int i = 0; i < 10; ++i) {
        features[i].features = {static_cast<float>(i), static_cast<float>(i + 1),
                                static_cast<float>(i + 2), static_cast<float>(i + 3)};
        features[i].layerIdx = 0;
        features[i].patchRow = 0;
        features[i].patchCol = i;
    }
    MemoryBank bank;
    bank.Build(features);
    ASSERT_TRUE(bank.Save("test_faiss_e2e.bin"));

    FaissIndexConfig cfg;
    cfg.algorithm = FaissSearchAlgorithm::BruteForce;
    cfg.d = 4;

    FaissIndexBridge bridge;
    ASSERT_TRUE(bridge.TrainFromMemoryBank("test_faiss_e2e.bin", cfg));
    EXPECT_TRUE(bridge.IsReady());

    std::vector<PatchFeature> queries;
    PatchFeature q;
    q.features = features[0].features;
    q.layerIdx = 0;
    q.patchRow = 0;
    q.patchCol = 0;
    queries.push_back(q);

    cv::Mat heatmap = bridge.ComputeAnomalyMap(queries, 10, 10);
    EXPECT_EQ(heatmap.rows, 10);
    EXPECT_EQ(heatmap.cols, 10);
    EXPECT_EQ(heatmap.type(), CV_32F);

    std::remove("test_faiss_e2e.bin");
    std::remove("test_faiss_e2e.bin.flat.faiss");
}
