// ============================================================
// 文件: tests/test_optimizer.cpp
// 用途: 模型优化模块单元测试
//   涵盖 OnnxExporter / TensorRtBuilder / Int8Calibrator /
//   ModelOptimizer / OptimizerApi
// ============================================================

#include <gtest/gtest.h>
#include "optimizer/onnx_exporter.h"
#include "optimizer/tensorrt_builder.h"
#include "optimizer/int8_calibrator.h"
#include "optimizer/model_optimizer.h"
#include "optimizer/optimizer_api.h"

using namespace aicore;

// 测试：创建 OnnxExporter 后错误信息为空
TEST(OnnxExporterTest, CreateAndConfigure) {
    OnnxExporter exporter;
    EXPECT_EQ(exporter.GetLastError(), "");
}

// 测试：TensorRtBuilder 在文件不存在时构建返回错误
TEST(TensorRtBuilderTest, BuildStubReturnsError) {
    TensorRtBuilder builder;
    BuildConfig cfg;
    cfg.onnxPath = "dummy.onnx";
    cfg.enginePath = "dummy.engine";
    auto s = builder.Build(cfg);
    EXPECT_FALSE(s);
}

// 测试：Int8Calibrator 加载标定数据
TEST(Int8CalibratorTest, LoadData) {
    Int8Calibrator cal;
    EXPECT_TRUE(cal.LoadCalibrationData("calib_dir", 100));
    EXPECT_EQ(cal.GetNumSamples(), 100);
    EXPECT_EQ(cal.GetBatch(), nullptr);
}

// 测试：空配置导致优化失败并有错误信息
TEST(ModelOptimizerTest, EmptyConfig) {
    ModelOptimizer opt;
    auto s = opt.Optimize("{}");
    EXPECT_FALSE(s);
    EXPECT_FALSE(opt.GetLastError().empty());
}

// 测试：优化器 API 版本号非空
TEST(ModelOptimizerTest, OptimizerApiVersion) {
    auto ver = aicore_optimizer_version();
    EXPECT_NE(ver, nullptr);
    EXPECT_NE(std::string(ver), "");
}

// 优化器测试用配置
const char* kOptimizerTestConfig = R"({
    "model_path": "test.pt",
    "onnx_path": "test.onnx",
    "engine_path": "test.engine",
    "precision": "fp16"
})";

// 测试：优化器 API 全流程（预期文件不存在导致失败）
TEST(ModelOptimizerTest, OptimizerApiFullFlow) {
    const char* err = nullptr;
    int ret = aicore_optimize(kOptimizerTestConfig, &err);
    EXPECT_NE(ret, 0);
}
