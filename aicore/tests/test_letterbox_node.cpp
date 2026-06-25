#include <gtest/gtest.h>
#include "preprocess/letterbox_node.h"
#include <opencv2/imgproc.hpp>

using namespace aicore;

TEST(LetterboxNodeTest, BasicLetterbox) {
    // 输入: 1280×720 图像 → 目标: 640×640
    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), -1);

    LetterboxNode node;
    NodeConfig cfg;
    cfg["width"] = "640";
    cfg["height"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    Frame input(img, 1);
    std::vector<Frame> inputs = {input};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1);

    // 输出是 640×640
    EXPECT_EQ(outputs[0].width(), 640);
    EXPECT_EQ(outputs[0].height(), 640);
    // roiMap 携带 letterbox 参数
    EXPECT_GT(outputs[0].roiMap.count("letterbox_scale"), 0);
    EXPECT_GT(outputs[0].roiMap.count("letterbox_pad_x"), 0);
    EXPECT_GT(outputs[0].roiMap.count("letterbox_pad_y"), 0);

    // 验证 scale: 640/1280=0.5 (w 方向约束), 640/720≈0.889 (h 方向约束)
    // scale = min(0.5, 0.889) = 0.5
    EXPECT_FLOAT_EQ(outputs[0].roiMap.at("letterbox_scale"), 0.5f);
}

TEST(LetterboxNodeTest, SmallerImage) {
    // 输入图像小于目标尺寸，应放大 + 填充
    cv::Mat img(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));

    LetterboxNode node;
    NodeConfig cfg;
    cfg["width"] = "640";
    cfg["height"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    Frame input(img, 1);
    std::vector<Frame> inputs = {input};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1);

    EXPECT_EQ(outputs[0].width(), 640);
    EXPECT_EQ(outputs[0].height(), 640);
    // scale 应 > 1
    EXPECT_GT(outputs[0].roiMap.at("letterbox_scale"), 1.0f);
}

TEST(LetterboxNodeTest, SquareImage) {
    // 正方形图像应刚好填满
    cv::Mat img(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));

    LetterboxNode node;
    NodeConfig cfg;
    cfg["width"] = "640";
    cfg["height"] = "640";
    ASSERT_TRUE(node.Init(cfg));

    Frame input(img, 1);
    std::vector<Frame> inputs = {input};
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1);

    EXPECT_EQ(outputs[0].width(), 640);
    EXPECT_EQ(outputs[0].height(), 640);
    EXPECT_FLOAT_EQ(outputs[0].roiMap.at("letterbox_scale"), 1.0f);
    // 无 padding（scale=1, newW=640, newH=640）
    EXPECT_FLOAT_EQ(outputs[0].roiMap.at("letterbox_pad_x"), 0.0f);
    EXPECT_FLOAT_EQ(outputs[0].roiMap.at("letterbox_pad_y"), 0.0f);
}
