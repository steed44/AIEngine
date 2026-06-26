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

// ============================================================
// PerImageRoiConfig 测试
// ============================================================

// 测试：PerImageRoiConfig 从 JSON 解析
TEST(PerImageRoiTest, FromJson) {
    const char* jsonContent = R"({
        "image": "test_img.png",
        "rois": [
            {"id": "1", "x": 10, "y": 20, "width": 100, "height": 80},
            {"id": "2", "x": 200, "y": 150, "width": 50, "height": 50},
            {"id": "3", "x": 50, "y": 300, "width": 120, "height": 90}
        ]
    })";

    FILE* f = fopen("test_per_img.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    PerImageRoiConfig cfg;
    auto s = PerImageRoiConfig::FromJson("test_per_img.json", cfg);
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(cfg.imagePath, "test_img.png");
    EXPECT_EQ(cfg.rois.size(), 3u);
    EXPECT_EQ(cfg.rois[0].id, "1");
    EXPECT_EQ(cfg.rois[0].x, 10);
    EXPECT_EQ(cfg.rois[0].y, 20);
    EXPECT_EQ(cfg.rois[0].w, 100);
    EXPECT_EQ(cfg.rois[0].h, 80);
    EXPECT_EQ(cfg.rois[1].id, "2");
    EXPECT_EQ(cfg.rois[2].id, "3");

    std::remove("test_per_img.json");
}

// 测试：PerImageRoiConfig 缺少必填字段应报错
TEST(PerImageRoiTest, MissingFields) {
    const char* jsonContent = R"({
        "image": "test.png",
        "rois": [
            {"id": "1", "x": 0, "y": 0, "width": 100},
            {"id": "", "x": 10, "y": 20, "width": 50, "height": 30}
        ]
    })";

    FILE* f = fopen("test_missing.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    PerImageRoiConfig cfg;
    auto s = PerImageRoiConfig::FromJson("test_missing.json", cfg);
    EXPECT_FALSE(s);

    std::remove("test_missing.json");
}

// 测试：PerImageRoiConfig ToJson 往返
TEST(PerImageRoiTest, ToJsonRoundtrip) {
    PerImageRoiConfig cfg;
    cfg.imagePath = "roundtrip_test.png";
    RoiDef r; r.id = "5"; r.x = 10; r.y = 20; r.w = 100; r.h = 80;
    cfg.rois.push_back(r);

    std::string json = cfg.ToJson();

    FILE* f = fopen("test_rt_pi.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(json.c_str(), f);
    fclose(f);

    PerImageRoiConfig loaded;
    auto s = PerImageRoiConfig::FromJson("test_rt_pi.json", loaded);
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(loaded.imagePath, cfg.imagePath);
    EXPECT_EQ(loaded.rois.size(), cfg.rois.size());
    EXPECT_EQ(loaded.rois[0].id, cfg.rois[0].id);

    std::remove("test_rt_pi.json");
}

// ============================================================
// MultiRoiConfig 每图模式测试
// ============================================================

// 测试：解析 per_image_file 模式
TEST(MultiRoiConfigPerImageTest, PerImageFileMode) {
    const char* jsonContent = R"({
        "backbone": {
            "type": "libtorch",
            "model_path": "backbone.pt",
            "input_size": 224,
            "layers": "layer2,layer3"
        },
        "roi_source": "per_image_file",
        "per_image_rois_dir": "./roi_annotations/",
        "train": {
            "coreset_fraction": 0.1,
            "max_features": 100000
        },
        "inference": {
            "anomaly_threshold": 0.5
        },
        "model_dir": "./models"
    })";

    FILE* f = fopen("test_per_image_mode.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    MultiRoiConfig cfg;
    auto s = cfg.FromJson("test_per_image_mode.json");
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(cfg.roiSource, MultiRoiConfig::RoiSourceType::PerImageFile);
    EXPECT_EQ(cfg.perImageRoisDir, "./roi_annotations/");
    EXPECT_EQ(cfg.backboneType, "libtorch");
    EXPECT_EQ(cfg.backboneModelPath, "backbone.pt");
    EXPECT_EQ(cfg.inputSize, 224);
    EXPECT_EQ(cfg.modelDir, "./models");
    EXPECT_EQ(cfg.rois.size(), 0u); // 每图模式下不应有固定 rois

    std::remove("test_per_image_mode.json");
}

// 测试：解析 per_image_list 模式
TEST(MultiRoiConfigPerImageTest, PerImageListMode) {
    const char* jsonContent = R"({
        "backbone": {
            "type": "opencv_dnn",
            "model_path": "model.onnx",
            "input_size": 224,
            "layers": "layer2,layer3"
        },
        "roi_source": "per_image_list",
        "per_image_rois_file": "./all_rois.json",
        "model_dir": "./models"
    })";

    FILE* f = fopen("test_per_image_list.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    MultiRoiConfig cfg;
    auto s = cfg.FromJson("test_per_image_list.json");
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(cfg.roiSource, MultiRoiConfig::RoiSourceType::PerImageList);
    EXPECT_EQ(cfg.perImageRoisFile, "./all_rois.json");

    std::remove("test_per_image_list.json");
}

// 测试：固定模式向后兼容（没有 roi_source 字段时默认 fixed）
TEST(MultiRoiConfigPerImageTest, BackwardCompatibleFixed) {
    const char* jsonContent = R"({
        "backbone": {
            "type": "opencv_dnn",
            "model_path": "m.onnx"
        },
        "rois": [
            {"id": "a", "x": 0, "y": 0, "width": 100, "height": 100},
            {"id": "b", "x": 200, "y": 200, "width": 50, "height": 50}
        ]
    })";

    FILE* f = fopen("test_backward_compat.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(jsonContent, f);
    fclose(f);

    MultiRoiConfig cfg;
    auto s = cfg.FromJson("test_backward_compat.json");
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(cfg.roiSource, MultiRoiConfig::RoiSourceType::Fixed);
    EXPECT_EQ(cfg.rois.size(), 2u);
    EXPECT_EQ(cfg.rois[0].id, "a");
    EXPECT_EQ(cfg.rois[1].id, "b");

    std::remove("test_backward_compat.json");
}

// 测试：ToJson 序列化 per_image_file 模式
TEST(MultiRoiConfigPerImageTest, ToJsonPerImageFile) {
    MultiRoiConfig cfg;
    cfg.roiSource = MultiRoiConfig::RoiSourceType::PerImageFile;
    cfg.backboneType = "libtorch";
    cfg.backboneModelPath = "bp.pt";
    cfg.perImageRoisDir = "./annotations/";
    cfg.modelDir = "./models";

    std::string json = cfg.ToJson();

    FILE* f = fopen("test_rt_pif.json", "w");
    ASSERT_NE(f, nullptr);
    fputs(json.c_str(), f);
    fclose(f);

    MultiRoiConfig loaded;
    auto s = loaded.FromJson("test_rt_pif.json");
    EXPECT_TRUE(s) << s.message;

    EXPECT_EQ(loaded.roiSource, MultiRoiConfig::RoiSourceType::PerImageFile);
    EXPECT_EQ(loaded.perImageRoisDir, cfg.perImageRoisDir);
    EXPECT_EQ(loaded.backboneType, cfg.backboneType);

    std::remove("test_rt_pif.json");
}
