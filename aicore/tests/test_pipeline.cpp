// ============================================================
// 文件: tests/test_pipeline.cpp
// 用途: 流水线各节点 (Resize/Normalize/NMS/Model/Merge/Composite)
//       及 PipelineBuilder/PipelineImpl 集成测试
// ============================================================

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

// 测试：ResizeNode 将图片缩放到指定尺寸
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

// 测试：NormalizeNode 将 uint8 转为 float32 类型
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

// 测试：NMS 节点初始化成功
TEST(NmsNodeTest, InitSucceeds) {
    NmsNode node;
    NodeConfig cfg{{"iou_threshold", "0.5"}, {"confidence_threshold", "0.3"}};
    EXPECT_TRUE(node.Init(cfg));
    EXPECT_EQ(node.GetType(), "nms");
}

// 测试：ModelNode 初始化和类型名称
TEST(ModelNodeTest, InitAndGetName) {
    auto backend = BackendFactory::Create(BackendType::kTensorRT);
    ASSERT_NE(backend, nullptr);
    auto sharedBackend = std::shared_ptr<IModelBackend>(std::move(backend));

    ModelNode node(sharedBackend);
    EXPECT_TRUE(node.Init({}));
    EXPECT_EQ(node.GetType(), "model");
    EXPECT_EQ(node.GetBackend().get(), sharedBackend.get());
}

// 测试：MergeNode 简单透传多个输入帧
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

// 测试：CompositeNode 组合节点设置内部流水线
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

// 测试：PipelineBuilder 根据 JSON 配置构建完整流水线
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

// 测试：无效节点类型导致构建失败
TEST(PipelineBuilderTest, InvalidNodeType) {
    PipelineBuilder builder;
    PipelineConfig config;
    config.nodes.push_back({"bad", "nonexistent", BackendType::kUnknown, "", 0, 1, {}});
    std::unique_ptr<IPipeline> pipeline;
    Status s = builder.Build(config, pipeline);
    EXPECT_FALSE(s);
}

// 测试：PipelineImpl 执行 resize → normalize 流水线，验证延迟 > 0
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
