// ============================================================
// 文件: tests/test_roi.cpp
// 用途: 多 ROI 配置解析和数据结构单元测试
// ============================================================
#include <gtest/gtest.h>
#include "patchcore/roi_def.h"
#include <cstdio>

using namespace aicore;

// 测试：MultiRoiConfig 从 JSON 文件解析，字段正确
TEST(RoiConfigTest, FromJson) {
    // 准备临时 JSON 文件
    const char* jsonContent = R"({
        "backbone": {
            "type": "libtorch",
            "model_path": "model.pt",
            "input_size": 224,
            "layers": "layer2,layer3"
        },
        "rois": [
            {"id": "roi_1", "x": 10, "y": 20, "width": 100, "height": 80},
            {"id": "roi_2", "x": 200, "y": 150, "width": 50, "height": 50}
        ],
        "train": {
            "coreset_fraction": 0.2,
            "max_features": 50000
        },
        "inference": {
            "anomaly_threshold": 0.6
        },
        "model_dir": "./my_models"
    })";

    FILE* f = fopen("test_roi_config.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    MultiRoiConfig cfg;
    auto s = cfg.FromJson("test_roi_config.json");
    EXPECT_TRUE(s) << s.message;

    // 验证 backbone 配置
    EXPECT_EQ(cfg.backboneType, "libtorch");
    EXPECT_EQ(cfg.backboneModelPath, "model.pt");
    EXPECT_EQ(cfg.inputSize, 224);
    EXPECT_EQ(cfg.backboneLayers, "layer2,layer3");

    // 验证 ROI 列表
    EXPECT_EQ(cfg.rois.size(), 2);
    EXPECT_EQ(cfg.rois[0].id, "roi_1");
    EXPECT_EQ(cfg.rois[0].x, 10);
    EXPECT_EQ(cfg.rois[0].y, 20);
    EXPECT_EQ(cfg.rois[0].w, 100);
    EXPECT_EQ(cfg.rois[0].h, 80);
    EXPECT_EQ(cfg.rois[1].id, "roi_2");
    EXPECT_EQ(cfg.rois[1].x, 200);
    EXPECT_EQ(cfg.rois[1].w, 50);

    // 验证训练参数
    EXPECT_DOUBLE_EQ(cfg.coresetFraction, 0.2);
    EXPECT_EQ(cfg.maxFeatures, 50000u);

    // 验证推理参数
    EXPECT_FLOAT_EQ(cfg.anomalyThreshold, 0.6f);

    // 验证模型目录
    EXPECT_EQ(cfg.modelDir, "./my_models");

    std::remove("test_roi_config.json");
}

// 测试：空 ROI 列表应返回错误
TEST(RoiConfigTest, EmptyRois) {
    const char* jsonContent = R"({
        "backbone": {"type": "opencv_dnn", "model_path": "m.onnx"},
        "rois": []
    })";

    FILE* f = fopen("test_empty_rois.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    MultiRoiConfig cfg;
    auto s = cfg.FromJson("test_empty_rois.json");
    EXPECT_FALSE(s);

    std::remove("test_empty_rois.json");
}

// 测试：JSON 序列化往返（ToJson → FromJson 后字段一致）
TEST(RoiConfigTest, ToJsonRoundtrip) {
    MultiRoiConfig cfg;
    cfg.backboneType = "libtorch";
    cfg.backboneModelPath = "test.pt";
    cfg.inputSize = 224;
    cfg.backboneLayers = "layer2,layer3";

    RoiDef r1; r1.id = "roi_a"; r1.x = 0; r1.y = 0; r1.w = 10; r1.h = 10;
    RoiDef r2; r2.id = "roi_b"; r2.x = 100; r2.y = 50; r2.w = 200; r2.h = 150;
    cfg.rois = {r1, r2};

    cfg.coresetFraction = 0.15;
    cfg.maxFeatures = 20000;
    cfg.anomalyThreshold = 0.7f;
    cfg.modelDir = "./test_models";

    // 序列化为 JSON 字符串，再解析回新配置
    std::string json = cfg.ToJson();

    FILE* f = fopen("test_roundtrip.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(json.c_str(), f);
    fclose(f);

    MultiRoiConfig loaded;
    auto s = loaded.FromJson("test_roundtrip.json");
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(loaded.backboneType, cfg.backboneType);
    EXPECT_EQ(loaded.backboneModelPath, cfg.backboneModelPath);
    EXPECT_EQ(loaded.inputSize, cfg.inputSize);
    EXPECT_EQ(loaded.rois.size(), cfg.rois.size());
    EXPECT_EQ(loaded.rois[0].id, cfg.rois[0].id);
    EXPECT_EQ(loaded.rois[1].w, cfg.rois[1].w);
    EXPECT_DOUBLE_EQ(loaded.coresetFraction, cfg.coresetFraction);
    EXPECT_EQ(loaded.maxFeatures, cfg.maxFeatures);
    EXPECT_FLOAT_EQ(loaded.anomalyThreshold, cfg.anomalyThreshold);
    EXPECT_EQ(loaded.modelDir, cfg.modelDir);

    std::remove("test_roundtrip.json");
}
