// ============================================================
// 文件: tests/test_backbone.cpp
// 用途: PatchCore Backbone 工厂和各实现的单元测试
//   涵盖: CreateBackbone 工厂 / OpenCVDnnBackbone / ModelBackendBackbone
// ============================================================

#include <gtest/gtest.h>
#include "patchcore/backbone.h"
#include "patchcore/backbone_opencv.h"
#include "patchcore/backbone_model.h"

using namespace aicore;

// ========== 工厂测试 ==========

TEST(BackboneFactoryTest, CreateOpenCV) {
    NodeConfig emptyCfg;
    auto bb = CreateBackbone("opencv_dnn", emptyCfg);
    ASSERT_NE(bb, nullptr);
    EXPECT_EQ(bb->GetType(), "opencv_dnn");
}

TEST(BackboneFactoryTest, CreateModelBackend) {
    NodeConfig emptyCfg;
    auto bb = CreateBackbone("model_backend", emptyCfg);
    ASSERT_NE(bb, nullptr);
    EXPECT_EQ(bb->GetType(), "model_backend");
}

TEST(BackboneFactoryTest, CreateUnknownType) {
    NodeConfig emptyCfg;
    auto bb = CreateBackbone("nonexistent", emptyCfg);
    EXPECT_EQ(bb, nullptr);
}

// ========== OpenCVDnnBackbone 测试 ==========

TEST(OpenCVDnnBackboneTest, InitMissingModelPath) {
    OpenCVDnnBackbone bb;
    NodeConfig cfg;
    auto s = bb.Init(cfg);
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorConfigParse);
}

TEST(OpenCVDnnBackboneTest, InitNonExistentModel) {
    OpenCVDnnBackbone bb;
    NodeConfig cfg;
    cfg["model_path"] = "nonexistent_model.onnx";
    // readNetFromONNX 在文件不存在时抛出 cv::Exception，当前实现未捕获
    EXPECT_THROW(bb.Init(cfg), cv::Exception);
}

// ========== ModelBackendBackbone 测试 ==========

TEST(ModelBackendBackboneTest, InitMissingModelPath) {
    ModelBackendBackbone bb;
    NodeConfig cfg;
    cfg["backend_type"] = "onnxruntime";
    auto s = bb.Init(cfg);
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorConfigParse);
}

TEST(ModelBackendBackboneTest, InitUnknownBackendType) {
    ModelBackendBackbone bb;
    NodeConfig cfg;
    cfg["model_path"] = "dummy.onnx";
    cfg["backend_type"] = "nonexistent";
    // 未知 backend_type 降级为默认 ONNX Runtime，stub 创建成功
    // Init 不会返回 ErrorConfigParse，但 backend_ 会是 ONNX Runtime 后端
    auto s = bb.Init(cfg);
    // stub 后端 Load 成功，所以 Init 也成功
    EXPECT_TRUE(s);
}
