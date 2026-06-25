#include <gtest/gtest.h>
#include "postprocess/yolo_decode_node.h"
#include "core/types.h"

using namespace aicore;

// 构造 YOLOv8 单尺度输出: [1, C, H, W], C = regMax*4 + nc
// 在 (gx, gy) 处放一个高置信度目标，其余背景类别为负值（sigmoid ≈ 0）
static std::vector<float> MakeFakeYOLOv8Output(int H, int W, int nc,
                                                int targetGx, int targetGy,
                                                float conf) {
    int regMax = 16;
    int C = regMax * 4 + nc;
    std::vector<float> data(C * H * W, -10.0f);  // 背景 cls 负值 → sigmoid ~0

    auto setCell = [&](int gx, int gy) {
        int idx = gy * W + gx;
        float* cell = data.data() + idx * C;
        // one-hot DFL: l=1, t=1, r=3, b=3
        cell[1] = 10.0f;
        cell[regMax + 1] = 10.0f;
        cell[2 * regMax + 3] = 10.0f;
        cell[3 * regMax + 3] = 10.0f;
        // class 5 高分
        cell[4 * regMax + 5] = conf;
        // 重置背景 DFL 部分为 0（避免负值干扰 softmax）
        for (int j = 0; j < 4 * regMax; j++) {
            if (cell[j] < 0) cell[j] = 0.0f;
        }
    };

    setCell(targetGx, targetGy);
    return data;
}

TEST(YoloDecodeNodeTest, DecodeV8SingleObject) {
    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.3";
    cfg["iou_threshold"] = "0.45";
    cfg["num_classes"] = "80";
    cfg["version"] = "v8";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    int H = 80, W = 80, nc = 80;
    int targetGx = 5, targetGy = 4;
    auto rawData = MakeFakeYOLOv8Output(H, W, nc, targetGx, targetGy, 20.0f);

    Tensor tensor;
    tensor.data = rawData.data();
    tensor.shape = {1, 4 * 16 + nc, H, W};
    tensor.dtype = DataType::kFloat32;
    tensor.bytes = rawData.size() * sizeof(float);

    Frame frame;
    frame.rawOutputs.push_back(tensor);
    frame.roiMap["letterbox_scale"] = 1.0f;
    frame.roiMap["letterbox_pad_x"] = 0.0f;
    frame.roiMap["letterbox_pad_y"] = 0.0f;

    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1);

    EXPECT_EQ(outputs[0].detections.size(), 1);
    if (!outputs[0].detections.empty()) {
        EXPECT_EQ(outputs[0].detections[0].label, "5");
        EXPECT_GT(outputs[0].detections[0].confidence, 0.5f);
        EXPECT_GT(outputs[0].detections[0].bbox.w, 0);
        EXPECT_GT(outputs[0].detections[0].bbox.h, 0);
    }
}

TEST(YoloDecodeNodeTest, FilterByConfidence) {
    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.9";
    cfg["iou_threshold"] = "0.45";
    cfg["num_classes"] = "80";
    cfg["version"] = "v8";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    int H = 20, W = 20, nc = 80;
    auto rawData = MakeFakeYOLOv8Output(H, W, nc, 5, 4, 0.5f);

    Tensor tensor;
    tensor.data = rawData.data();
    tensor.shape = {1, 4 * 16 + nc, H, W};
    tensor.dtype = DataType::kFloat32;
    tensor.bytes = rawData.size() * sizeof(float);

    Frame frame;
    frame.rawOutputs.push_back(tensor);
    frame.roiMap["letterbox_scale"] = 1.0f;
    frame.roiMap["letterbox_pad_x"] = 0.0f;
    frame.roiMap["letterbox_pad_y"] = 0.0f;

    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));

    EXPECT_EQ(outputs[0].detections.size(), 0);
}

TEST(YoloDecodeNodeTest, NMSRemovesOverlap) {
    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.1";
    cfg["iou_threshold"] = "0.5";
    cfg["num_classes"] = "80";
    cfg["version"] = "v8";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    int H = 20, W = 20, nc = 80;
    int regMax = 16;
    int C = regMax * 4 + nc;
    std::vector<float> data(C * H * W, -10.0f);  // 背景 cls 负值 → sigmoid ~0

    auto setCell = [&](int gx, int gy, float conf) {
        int idx = gy * W + gx;
        float* cell = data.data() + idx * C;
        cell[1] = 10.0f;
        cell[regMax + 1] = 10.0f;
        cell[2 * regMax + 3] = 10.0f;
        cell[3 * regMax + 3] = 10.0f;
        cell[4 * regMax + 5] = conf;
        for (int j = 0; j < 4 * regMax; j++) {
            if (cell[j] < 0) cell[j] = 0.0f;
        }
    };
    setCell(5, 5, 20.0f);
    setCell(6, 5, 15.0f);
    setCell(15, 15, 18.0f);

    Tensor tensor;
    tensor.data = data.data();
    tensor.shape = {1, C, H, W};
    tensor.dtype = DataType::kFloat32;
    tensor.bytes = data.size() * sizeof(float);

    Frame frame;
    frame.rawOutputs.push_back(tensor);
    frame.roiMap["letterbox_scale"] = 1.0f;
    frame.roiMap["letterbox_pad_x"] = 0.0f;
    frame.roiMap["letterbox_pad_y"] = 0.0f;

    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));

    EXPECT_EQ(outputs[0].detections.size(), 2);
}

// YOLOv5 格式: [cx, cy, w, h, cls0..clsn-1]
static std::vector<float> MakeFakeYOLOv5Output(int numBoxes, int nc,
                                                int targetIdx, float conf) {
    int stride = 4 + nc;
    std::vector<float> data(numBoxes * stride, -10.0f);  // 背景 cls 负值
    float* cell = data.data() + targetIdx * stride;
    cell[0] = 320.0f;
    cell[1] = 320.0f;
    cell[2] = 100.0f;
    cell[3] = 80.0f;
    cell[4 + 3] = conf;
    return data;
}

TEST(YoloDecodeNodeTest, DecodeV5SingleObject) {
    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.3";
    cfg["iou_threshold"] = "0.45";
    cfg["num_classes"] = "80";
    cfg["version"] = "v5";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    int numBoxes = 100;
    int nc = 80;
    auto rawData = MakeFakeYOLOv5Output(numBoxes, nc, 42, 20.0f);

    Tensor tensor;
    tensor.data = rawData.data();
    tensor.shape = {1, 4 + nc, 10, 10};
    tensor.dtype = DataType::kFloat32;
    tensor.bytes = rawData.size() * sizeof(float);

    Frame frame;
    frame.rawOutputs.push_back(tensor);
    frame.roiMap["letterbox_scale"] = 1.0f;
    frame.roiMap["letterbox_pad_x"] = 0.0f;
    frame.roiMap["letterbox_pad_y"] = 0.0f;

    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));

    EXPECT_EQ(outputs[0].detections.size(), 1);
    if (!outputs[0].detections.empty()) {
        EXPECT_EQ(outputs[0].detections[0].label, "3");
        EXPECT_GT(outputs[0].detections[0].confidence, 0.5f);
        EXPECT_GT(outputs[0].detections[0].bbox.x, 0);
        EXPECT_GT(outputs[0].detections[0].bbox.y, 0);
    }
}

TEST(YoloDecodeNodeTest, LetterboxCoordinateRoundtrip) {
    float scale = 0.5f;
    int padX = 0;
    int padY = 80;

    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.1";
    cfg["iou_threshold"] = "0.45";
    cfg["num_classes"] = "80";
    cfg["version"] = "v8";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    int H = 80, W = 80, nc = 80, regMax = 16;
    int C = regMax * 4 + nc;
    std::vector<float> data(C * H * W, -10.0f);  // 背景 cls 负值

    int gx = 20, gy = 15;
    // 重置目标 cell 的 DFL 部分为 0（负值干扰 softmax）
    float* targetCell = data.data() + (gy * W + gx) * C;
    for (int j = 0; j < 4 * regMax; j++) targetCell[j] = 0.0f;
    float* cell = data.data() + (gy * W + gx) * C;
    cell[1] = 10.0f;
    cell[regMax + 1] = 10.0f;
    cell[2 * regMax + 3] = 10.0f;
    cell[3 * regMax + 3] = 10.0f;
    cell[4 * regMax + 5] = 20.0f;

    // 预期:
    // x1 = 20.5-1=19.5, y1=15.5-1=14.5, x2=20.5+3=23.5, y2=15.5+3=18.5
    // cx=21.5(grid), cy=16.5(grid), w=4(grid), h=4(grid)
    // stride=640/80=8 => cx=172, cy=132, w=32, h=32 (model coords)
    // un-letterbox: cx=(172-0)/0.5=344, cy=(132-80)/0.5=104, w=64, h=64
    float expectedCx = 344.0f;
    float expectedCy = 104.0f;
    float expectedW = 64.0f;
    float expectedH = 64.0f;

    Tensor tensor;
    tensor.data = data.data();
    tensor.shape = {1, C, H, W};
    tensor.dtype = DataType::kFloat32;
    tensor.bytes = data.size() * sizeof(float);

    Frame frame;
    frame.rawOutputs.push_back(tensor);
    frame.roiMap["letterbox_scale"] = scale;
    frame.roiMap["letterbox_pad_x"] = (float)padX;
    frame.roiMap["letterbox_pad_y"] = (float)padY;

    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));

    ASSERT_EQ(outputs[0].detections.size(), 1);
    EXPECT_NEAR(outputs[0].detections[0].bbox.x, expectedCx, 1.0f);
    EXPECT_NEAR(outputs[0].detections[0].bbox.y, expectedCy, 1.0f);
    EXPECT_NEAR(outputs[0].detections[0].bbox.w, expectedW, 1.0f);
    EXPECT_NEAR(outputs[0].detections[0].bbox.h, expectedH, 1.0f);
}

TEST(YoloDecodeNodeTest, EmptyRawOutputs) {
    YoloDecodeNode node;
    NodeConfig cfg;
    cfg["confidence_threshold"] = "0.5";
    cfg["iou_threshold"] = "0.45";
    cfg["num_classes"] = "80";
    cfg["version"] = "v8";
    cfg["model_input_size"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    Frame frame;
    std::vector<Frame> inputs = {frame};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));

    EXPECT_EQ(outputs[0].detections.size(), 0);
}
