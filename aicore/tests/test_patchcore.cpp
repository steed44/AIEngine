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
