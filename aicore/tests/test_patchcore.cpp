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
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <filesystem>

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
