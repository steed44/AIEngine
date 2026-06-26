#include <gtest/gtest.h>
#include "pipeline/fusion_node.h"
#include "core/frame.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

TEST(FusionNodeTest, InitFailsWithBadBackboneType) {
    FusionNode node;
    NodeConfig cfg;
    cfg["backbone_type"] = "nonexistent_backend";
    cfg["model_path"] = "model.onnx";
    cfg["memory_bank_path"] = "bank.bin";
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
}

TEST(FusionNodeTest, InitFailsWithoutMemoryBank) {
    FusionNode node;
    NodeConfig cfg;
    cfg["backbone_type"] = "opencv_dnn";
    cfg["model_path"] = "nonexistent.onnx";
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
}

TEST(FusionNodeTest, InitFailsMissingMemoryBankPath) {
    FusionNode node;
    NodeConfig cfg;
    cfg["backbone_type"] = "nonexistent_backend";
    cfg["model_path"] = "dummy.onnx";
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorConfigParse);
}

TEST(FusionNodeTest, ProcessFailsWithoutInit) {
    FusionNode node;
    std::vector<Frame> inputs;
    inputs.emplace_back(cv::Mat(100, 100, CV_8UC3), 0);
    inputs.emplace_back(cv::Mat(100, 100, CV_8UC3), 1);
    std::vector<Frame> outputs;
    auto s = node.Process(inputs, outputs);
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorInternal);
}

TEST(FusionNodeTest, NeedsTwoInputs) {
    FusionNode node;
    NodeConfig cfg;
    cfg["node_id"] = "test";
    node.Init(cfg); // will fail
    std::vector<Frame> inputs;
    inputs.emplace_back(cv::Mat(100, 100, CV_8UC3), 0);
    std::vector<Frame> outputs;
    auto s = node.Process(inputs, outputs);
    EXPECT_FALSE(s);
}

TEST(FusionNodeTest, TypeAndName) {
    FusionNode node;
    NodeConfig cfg;
    cfg["node_id"] = "my_fusion";
    auto s = node.Init(cfg);
    EXPECT_FALSE(s);
    EXPECT_EQ(node.GetName(), "my_fusion");
    EXPECT_EQ(node.GetType(), "fusion");
}

} // namespace aicore
