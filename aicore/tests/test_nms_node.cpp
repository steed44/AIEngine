#include <gtest/gtest.h>
#include "postprocess/nms_node.h"
#include "postprocess/nms_common.h"

using namespace aicore;

TEST(NmsCommonTest, IouBoxIdentical) {
    BBox a{10, 10, 20, 20};  // [0,0,20,20]
    BBox b{10, 10, 20, 20};
    EXPECT_NEAR(IouBox(a, b), 1.0f, 1e-6f);
}

TEST(NmsCommonTest, IouBoxNoOverlap) {
    BBox a{10, 10, 20, 20};
    BBox b{40, 10, 20, 20};
    EXPECT_NEAR(IouBox(a, b), 0.0f, 1e-6f);
}

TEST(NmsCommonTest, IouBoxPartialOverlap) {
    BBox a{10, 10, 20, 20};  // [0,0,20,20]
    BBox b{15, 10, 20, 20};  // [5,0,25,20]
    // inter: [5,0,20,20] = 15*20 = 300
    // areaA=400, areaB=400, union=500
    // IoU=300/500=0.6
    EXPECT_NEAR(IouBox(a, b), 0.6f, 1e-6f);
}

TEST(NmsCommonTest, FiltersByClass) {
    std::vector<NodeResult> dets;
    dets.push_back({"", "cat", 0.9f, {10, 10, 20, 20}});
    dets.push_back({"", "cat", 0.8f, {12, 12, 20, 20}});  // overlaps cat
    dets.push_back({"", "dog", 0.85f, {10, 10, 20, 20}}); // same coords, diff class
    NMSCommon(dets, 0.5f);
    ASSERT_EQ(dets.size(), 2);  // 1 cat + 1 dog
}

TEST(NmsCommonTest, EmptyInput) {
    std::vector<NodeResult> dets;
    NMSCommon(dets, 0.5f);
    EXPECT_TRUE(dets.empty());
}

TEST(NmsCommonTest, SingleDetection) {
    std::vector<NodeResult> dets;
    dets.push_back({"", "cat", 0.9f, {10, 10, 20, 20}});
    NMSCommon(dets, 0.5f);
    ASSERT_EQ(dets.size(), 1);
}

TEST(NmsNodeTest, FiltersByConfidence) {
    NmsNode node;
    ASSERT_TRUE(node.Init({{"confidence_threshold", "0.5"},
                           {"iou_threshold", "0.45"}}));
    Frame input;
    input.detections = {
        {"", "cat", 0.9f, {10, 10, 20, 20}},
        {"", "cat", 0.3f, {15, 15, 20, 20}},
    };
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({input}, outputs));
    ASSERT_EQ(outputs[0].detections.size(), 1);
    EXPECT_NEAR(outputs[0].detections[0].confidence, 0.9f, 0.01f);
}

TEST(NmsNodeTest, EmptyInput) {
    NmsNode node;
    ASSERT_TRUE(node.Init({}));
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({Frame{}}, outputs));
    EXPECT_TRUE(outputs[0].detections.empty());
}

TEST(NmsNodeTest, NMSFiltersOverlap) {
    NmsNode node;
    ASSERT_TRUE(node.Init({{"confidence_threshold", "0.0"},
                           {"iou_threshold", "0.5"}}));
    Frame input;
    input.detections = {
        {"", "cat", 0.9f, {10, 10, 20, 20}},
        {"", "cat", 0.8f, {12, 12, 20, 20}},   // high IoU with first
        {"", "dog", 0.7f, {10, 10, 20, 20}},   // diff class, kept
    };
    std::vector<Frame> outputs;
    ASSERT_TRUE(node.Process({input}, outputs));
    // 1 cat (high conf) + 1 dog, 1 cat suppressed
    EXPECT_EQ(outputs[0].detections.size(), 2);
}
