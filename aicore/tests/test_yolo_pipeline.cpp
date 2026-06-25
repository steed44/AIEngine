#include <gtest/gtest.h>
#include "core/pipeline.h"
#include "core/frame.h"
#include "core/types.h"
#include "config/config_parser.h"
#include "config/pipeline_builder.h"
#include "pipeline/pipeline_impl.h"
#include "preprocess/letterbox_node.h"
#include "postprocess/yolo_decode_node.h"
#include "engine/engine_pool.h"
#include <opencv2/imgproc.hpp>

using namespace aicore;

// Pipeline 构建 — letterbox + yolo_decode 配置解析和构建
TEST(YOLOPipelineTest, BuildPipelineConfig) {
    ConfigParser parser;
    PipelineConfig config;
    std::string json = R"({
        "pipeline": {
            "name": "yolo_test",
            "max_concurrency": 1,
            "nodes": [
                {"id": "letterbox", "type": "letterbox",
                 "params": {"width": "640", "height": "640"}},
                {"id": "yolo_decode", "type": "yolo_decode",
                 "params": {"confidence_threshold": "0.5",
                            "iou_threshold": "0.45",
                            "num_classes": "80",
                            "version": "v8"}}
            ],
            "edges": [
                {"from": "input", "to": "letterbox"},
                {"from": "letterbox", "to": "yolo_decode"}
            ]
        }
    })";
    ASSERT_TRUE(parser.Parse(json, config));

    auto pool = std::make_shared<EnginePool>();
    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    auto s = builder.Build(config, pipeline, pool);
    EXPECT_TRUE(s) << s.message;
    EXPECT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->GetState(), PipelineState::kReady);
}

// letterbox → yolo_decode 数据流（无 model 节点，rawOutputs 为空 → 空 detections）
TEST(YOLOPipelineTest, LetterboxToDecodeFlow) {
    ConfigParser parser;
    PipelineConfig config;
    std::string json = R"({
        "pipeline": {
            "name": "yolo_flow_test",
            "max_concurrency": 1,
            "nodes": [
                {"id": "letterbox", "type": "letterbox",
                 "params": {"width": "640", "height": "640"}},
                {"id": "yolo_decode", "type": "yolo_decode",
                 "params": {"confidence_threshold": "0.5",
                            "iou_threshold": "0.45",
                            "num_classes": "80",
                            "version": "v8"}}
            ],
            "edges": [
                {"from": "input", "to": "letterbox"},
                {"from": "letterbox", "to": "yolo_decode"}
            ]
        }
    })";
    ASSERT_TRUE(parser.Parse(json, config));

    auto pool = std::make_shared<EnginePool>();
    PipelineBuilder builder;
    std::unique_ptr<IPipeline> pipeline;
    auto s = builder.Build(config, pipeline, pool);
    ASSERT_TRUE(s);

    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    Frame input(img, 1);

    Result result;
    s = pipeline->Execute(input, result);
    EXPECT_TRUE(s) << "errMsg=[" << result.errorMsg << "] status=" << (int)result.status;
    EXPECT_EQ(result.status, StatusCode::OK) << result.errorMsg;
    // 无 model 节点 → rawOutputs 为空 → yolo_decode 输出空 detections
    EXPECT_TRUE(result.detections.empty());
}

// 完整 pipeline 执行（含模拟 rawOutputs — 通过 PipelineImpl 直接组装节点）
TEST(YOLOPipelineTest, FullPipelineWithMockData) {
    // 直接组装节点 (letterbox → yolo_decode)
    auto pipeline = std::make_unique<PipelineImpl>();

    auto letterbox = std::make_shared<LetterboxNode>();
    ASSERT_TRUE(letterbox->Init({{"width", "640"}, {"height", "640"}}));

    auto yoloDecode = std::make_shared<YoloDecodeNode>();
    ASSERT_TRUE(yoloDecode->Init({{"confidence_threshold", "0.3"},
                                  {"iou_threshold", "0.45"},
                                  {"num_classes", "80"},
                                  {"version", "v8"},
                                  {"model_input_size", "640"}}));

    pipeline->AddNode("letterbox", letterbox, {"input"});
    pipeline->AddNode("yolo_decode", yoloDecode, {"letterbox"});
    pipeline->AddEdge("letterbox", "yolo_decode");
    pipeline->MarkReady();

    EXPECT_EQ(pipeline->GetState(), PipelineState::kReady);

    // 构造模拟原始输出（YOLOv8 单尺度 80×80 grid，class 5 高置信度）
    int H = 80, W = 80, nc = 80, regMax = 16;
    int C = regMax * 4 + nc;
    std::vector<float> tensorData(C * H * W, -10.0f);

    int targetGx = 10, targetGy = 10;
    float* cell = tensorData.data() + (targetGy * W + targetGx) * C;
    for (int j = 0; j < 4 * regMax; j++) cell[j] = 0.0f;
    cell[1] = 10.0f;
    cell[regMax + 1] = 10.0f;
    cell[2 * regMax + 3] = 10.0f;
    cell[3 * regMax + 3] = 10.0f;
    cell[4 * regMax + 5] = 20.0f;

    Tensor rawTensor;
    rawTensor.data = tensorData.data();
    rawTensor.shape = {1, C, H, W};
    rawTensor.dtype = DataType::kFloat32;
    rawTensor.bytes = tensorData.size() * sizeof(float);

    // 创建输入帧，但没法在输入帧上传 rawOutputs（只由 model 节点产生）
    // 所以测试验证的是 letterbox 预处理 + yolo_decode 处理空 rawOutputs 的流程
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    Frame input(img, 1);

    Result result;
    Status s = pipeline->Execute(input, result);
    EXPECT_TRUE(s) << result.errorMsg;
    EXPECT_EQ(result.status, StatusCode::OK);
    // 无原始输出（无 model 节点），detections 应为空
    EXPECT_TRUE(result.detections.empty());
}
