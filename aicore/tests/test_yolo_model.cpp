#include <gtest/gtest.h>
#include "trainer/model/yolo_model.h"

using namespace aicore;

// YOLOv8Model 前向传播形状验证
TEST(YOLOModelTest, ForwardReturnsCorrectShapes) {
    YOLOv8Model model(3);
    ModelConfig cfg;
    cfg.numClasses = 3;
    ASSERT_TRUE(model.Build(cfg));

    auto input = torch::randn({1, 3, 128, 128});
    auto outputs = model.Forward(input);
    ASSERT_EQ(outputs.size(), 3);

    EXPECT_EQ(outputs[0].size(0), 1);
    EXPECT_EQ(outputs[0].size(1), 4 * 16 + 3); // no = 4*regMax + nc
    EXPECT_EQ(outputs[0].size(2), 16);  // 128/8 = 16
    EXPECT_EQ(outputs[0].size(3), 16);

    EXPECT_EQ(outputs[1].size(2), 8);  // 128/16 = 8
    EXPECT_EQ(outputs[1].size(3), 8);

    EXPECT_EQ(outputs[2].size(2), 4);  // 128/32 = 4
    EXPECT_EQ(outputs[2].size(3), 4);

    for (auto& o : outputs) {
        EXPECT_TRUE(o.isfinite().all().item<bool>());
    }
}

TEST(YOLOModelTest, PredictReturnsCorrectShape) {
    YOLOv8Model model(3);
    ModelConfig cfg;
    cfg.numClasses = 3;
    ASSERT_TRUE(model.Build(cfg));

    auto input = torch::randn({2, 3, 128, 128});
    auto out = model.predict(input);

    EXPECT_EQ(out.size(0), 2);
    EXPECT_EQ(out.size(1), 4 * 16 + 3); // 4*regMax + nc

    int64_t totalGrids = 16*16 + 8*8 + 4*4; // 256 + 64 + 16 = 336
    EXPECT_EQ(out.size(2), totalGrids);
    EXPECT_TRUE(out.isfinite().all().item<bool>());
}

TEST(YOLOModelTest, PredictMultiBatch) {
    YOLOv8Model model(3);
    ModelConfig cfg;
    cfg.numClasses = 3;
    ASSERT_TRUE(model.Build(cfg));

    auto input = torch::randn({4, 3, 128, 128});
    auto out = model.predict(input);
    EXPECT_EQ(out.size(0), 4);
    EXPECT_TRUE(out.isfinite().all().item<bool>());
}
