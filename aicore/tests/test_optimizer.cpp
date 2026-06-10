#include <gtest/gtest.h>
#include "optimizer/onnx_exporter.h"
#include "optimizer/tensorrt_builder.h"
#include "optimizer/int8_calibrator.h"
#include "optimizer/model_optimizer.h"
#include "optimizer/optimizer_api.h"

using namespace aicore;

TEST(OnnxExporterTest, CreateAndConfigure) {
    OnnxExporter exporter;
    EXPECT_EQ(exporter.GetLastError(), "");
}

TEST(TensorRtBuilderTest, BuildStubReturnsError) {
    TensorRtBuilder builder;
    BuildConfig cfg;
    cfg.onnxPath = "dummy.onnx";
    cfg.enginePath = "dummy.engine";
    auto s = builder.Build(cfg);
    EXPECT_FALSE(s);
}

TEST(Int8CalibratorTest, LoadData) {
    Int8Calibrator cal;
    EXPECT_TRUE(cal.LoadCalibrationData("calib_dir", 100));
    EXPECT_EQ(cal.GetNumSamples(), 100);
    EXPECT_EQ(cal.GetBatch(), nullptr);
}

TEST(ModelOptimizerTest, EmptyConfig) {
    ModelOptimizer opt;
    auto s = opt.Optimize("{}");
    EXPECT_FALSE(s);
    EXPECT_FALSE(opt.GetLastError().empty());
}

TEST(ModelOptimizerTest, OptimizerApiVersion) {
    auto ver = aicore_optimizer_version();
    EXPECT_NE(ver, nullptr);
    EXPECT_NE(std::string(ver), "");
}

const char* kOptimizerTestConfig = R"({
    "model_path": "test.pt",
    "onnx_path": "test.onnx",
    "engine_path": "test.engine",
    "precision": "fp16"
})";

TEST(ModelOptimizerTest, OptimizerApiFullFlow) {
    const char* err = nullptr;
    int ret = aicore_optimize(kOptimizerTestConfig, &err);
    EXPECT_NE(ret, 0);
}
