#include <gtest/gtest.h>
#include "core/pipeline.h"
#include "config/pipeline_builder.h"
#include "config/config_parser.h"
#include "pipeline/pipeline_impl.h"
#include "pipeline/model_node.h"
#include "pipeline/composite_node.h"
#include "pipeline/merge_node.h"
#include "preprocess/resize_node.h"
#include "preprocess/normalize_node.h"
#include "postprocess/nms_node.h"
#include "engine/engine_pool.h"

using namespace aicore;

TEST(ResizeNodeTest, ResizeToTarget) {
    ResizeNode node;
    NodeConfig cfg{{"width", "320"}, {"height", "240"}};
    ASSERT_TRUE(node.Init(cfg));

    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    Frame input(img, 1);
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({input}, outputs));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].width(), 320);
    EXPECT_EQ(outputs[0].height(), 240);
}

TEST(NormalizeNodeTest, NormalizeValues) {
    NormalizeNode node;
    ASSERT_TRUE(node.Init({}));

    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(255, 128, 0));
    Frame input(img, 1);
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({input}, outputs));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].image.type(), CV_32FC3);
}

TEST(NmsNodeTest, InitSucceeds) {
    NmsNode node;
    NodeConfig cfg{{"iou_threshold", "0.5"}, {"confidence_threshold", "0.3"}};
    EXPECT_TRUE(node.Init(cfg));
    EXPECT_EQ(node.GetType(), "nms");
}

TEST(ModelNodeTest, InitAndGetName) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    auto sharedBackend = std::shared_ptr<IModelBackend>(std::move(backend));

    ModelNode node(sharedBackend);
    EXPECT_TRUE(node.Init({}));
    EXPECT_EQ(node.GetType(), "model");
    EXPECT_EQ(node.GetBackend().get(), sharedBackend.get());
}

TEST(MergeNodeTest, PassThrough) {
    MergeNode node;
    ASSERT_TRUE(node.Init({}));

    cv::Mat img(10, 10, CV_8UC3);
    Frame f1(img.clone(), 1);
    Frame f2(img.clone(), 2);

    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({f1, f2}, outputs));
    ASSERT_EQ(outputs.size(), 2);
    EXPECT_EQ(outputs[0].frameId, 1);
    EXPECT_EQ(outputs[1].frameId, 2);
}

TEST(CompositeNodeTest, InnerPipeline) {
    CompositeNode node;
    ASSERT_TRUE(node.Init({}));

    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    auto sharedBackend = std::shared_ptr<IModelBackend>(std::move(backend));

    auto pipeline = std::make_unique<PipelineImpl>();
    auto modelNode = std::make_shared<ModelNode>(sharedBackend);
    pipeline->AddNode("model", modelNode, {});

    node.SetInnerPipeline(std::move(pipeline));
    EXPECT_EQ(node.GetType(), "composite");
}

TEST(PipelineBuilderTest, BuildSimplePipeline) {
    ConfigParser parser;
    PipelineConfig config;
    std::string json = R"({
        "pipeline": {
            "name": "test",
            "nodes": [
                {"id": "resize_1", "type": "resize", "params": {"width": "640", "height": "640"}},
                {"id": "norm_1", "type": "normalize"},
                {"id": "model_1", "type": "model", "backend": "tensorrt", "model_path": "dummy"}
            ],
            "edges": [
                {"from": "input", "to": "resize_1"},
                {"from": "resize_1", "to": "norm_1"},
                {"from": "norm_1", "to": "model_1"}
            ]
        }
    })";
    ASSERT_TRUE(parser.Parse(json, config));

    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    Status s = builder.Build(config, pipeline);
    ASSERT_TRUE(s);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->GetState(), PipelineState::kReady);
}

TEST(PipelineBuilderTest, InvalidNodeType) {
    PipelineBuilder builder;
    PipelineConfig config;
    config.nodes.push_back({"bad", "nonexistent", BackendType::kUnknown, "", 0, 1, {}});
    std::unique_ptr<IPipeline> pipeline;
    Status s = builder.Build(config, pipeline);
    EXPECT_FALSE(s);
}

TEST(PipelineTest, ExecuteSimple) {
    auto pipeline = std::make_unique<PipelineImpl>();
    auto resizeNode = std::make_shared<ResizeNode>();
    resizeNode->Init({{"width", "32"}, {"height", "32"}});
    auto normNode = std::make_shared<NormalizeNode>();
    normNode->Init({});

    pipeline->AddNode("resize", resizeNode, {"input"});
    pipeline->AddNode("norm", normNode, {"resize"});
    pipeline->AddEdge("resize", "norm");
    pipeline->MarkReady();

    EXPECT_EQ(pipeline->GetState(), PipelineState::kReady);

    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(128, 128, 128));
    Frame input(img, 1);
    Result result;
    Status s = pipeline->Execute(input, result);
    EXPECT_TRUE(s) << result.errorMsg;
    EXPECT_GT(result.totalLatencyMs, 0);
}
