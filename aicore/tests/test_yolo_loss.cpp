#include <gtest/gtest.h>
#include "trainer/model/yolo_loss.h"

using namespace aicore;

// ─── boxCxwhToXyxy ────────────────────────────────────────────

TEST(YOLOLossTest, BoxCxwhToXyxyIdentity) {
    auto cx = torch::tensor({0.5, 0.3, 0.7});
    auto cy = torch::tensor({0.5, 0.4, 0.6});
    auto w  = torch::tensor({0.2, 0.1, 0.3});
    auto h  = torch::tensor({0.4, 0.2, 0.5});
    auto boxes = torch::stack({cx, cy, w, h}, 1);
    auto xyxy = boxCxwhToXyxy(boxes);
    EXPECT_NEAR(xyxy[0][0].item<float>(), 0.4f, 1e-6);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 0.3f, 1e-6);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 0.6f, 1e-6);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 0.7f, 1e-6);
}

TEST(YOLOLossTest, BoxCxwhToXyxyZeroSize) {
    auto cx = torch::tensor({});
    auto cy = torch::tensor({});
    auto w  = torch::tensor({});
    auto h  = torch::tensor({});
    auto boxes = torch::stack({cx, cy, w, h}, 1);
    auto xyxy = boxCxwhToXyxy(boxes);
    EXPECT_EQ(xyxy.size(0), 0);
}

// ─── computeIoU ────────────────────────────────────────────────

TEST(YOLOLossTest, IoUIdenticalBoxes) {
    auto b1 = torch::tensor({{0.2, 0.2, 0.8, 0.8}});
    auto b2 = torch::tensor({{0.2, 0.2, 0.8, 0.8}});
    auto iou = computeIoU(b1, b2);
    EXPECT_NEAR(iou.item<float>(), 1.0f, 1e-6);
}

TEST(YOLOLossTest, IoUNoOverlap) {
    auto b1 = torch::tensor({{0.0, 0.0, 0.1, 0.1}});
    auto b2 = torch::tensor({{0.9, 0.9, 1.0, 1.0}});
    auto iou = computeIoU(b1, b2);
    EXPECT_NEAR(iou.item<float>(), 0.0f, 1e-6);
}

TEST(YOLOLossTest, IoUHalfOverlap) {
    auto b1 = torch::tensor({{0.0, 0.0, 1.0, 1.0}});
    auto b2 = torch::tensor({{0.5, 0.0, 1.5, 1.0}});
    auto iou = computeIoU(b1, b2);
    // intersection: x [0.5,1.0], y [0,1] => area 0.5
    // union: 1.0 + 1.0 - 0.5 = 1.5
    // iou: 0.5/1.5 = 0.3333
    EXPECT_NEAR(iou.item<float>(), 0.5f / 1.5f, 1e-6);
}

// IoU 25%: inner box area 0.16 / outer box area 0.64
TEST(YOLOLossTest, IoUContains) {
    auto b1 = torch::tensor({{0.1, 0.1, 0.9, 0.9}});
    auto b2 = torch::tensor({{0.3, 0.3, 0.7, 0.7}});
    auto iou = computeIoU(b1, b2);
    // intersection = area of b2 = 0.4^2 = 0.16
    // union = area of b1 = 0.8^2 = 0.64
    // iou = 0.16/0.64 = 0.25
    EXPECT_NEAR(iou.item<float>(), 0.25f, 1e-6);
}

// ─── ciouLoss ──────────────────────────────────────────────────

TEST(YOLOLossTest, CIoUIdenticalZero) {
    auto b = torch::tensor({{0.2, 0.2, 0.8, 0.8}});
    auto loss = ciouLoss(b, b);
    // CIoU = 1 - IoU + 0 + 0 = 0 when IoU = 1, same box
    EXPECT_NEAR(loss.item<float>(), 0.0f, 1e-6);
}

TEST(YOLOLossTest, CIoUNonOverlap) {
    auto b1 = torch::tensor({{0.0, 0.0, 0.1, 0.1}});
    auto b2 = torch::tensor({{0.9, 0.9, 1.0, 1.0}});
    auto loss = ciouLoss(b1, b2);
    EXPECT_GT(loss.item<float>(), 1.0f); // IoU=0 + center dist + aspect ratio
}

TEST(YOLOLossTest, CIoUDifferentAspect) {
    // same center, identical IoU=0.5 boxes, different aspect ratio
    // b1: w=0.5, h=0.3 ; b2: w=0.3, h=0.5
    // IoU = 0.25/0.35 = 0.714, additional v-term adds ~0.001
    auto b1 = torch::tensor({{0.25, 0.35, 0.75, 0.65}});
    auto b2 = torch::tensor({{0.35, 0.25, 0.65, 0.75}});
    auto sameIoU = ciouLoss(b1, b2);
    // same-center, diff-aspect → loss > 1 - IoU (v-term adds penalty)
    auto plainIncome = (1 - computeIoU(b1, b2).item<float>());
    EXPECT_GT(sameIoU.item<float>(), plainIncome);
}

TEST(YOLOLossTest, CIoUMultipleBoxes) {
    auto b1 = torch::tensor({{0.0, 0.0, 1.0, 1.0}, {0.2, 0.2, 0.8, 0.8}});
    auto b2 = torch::tensor({{0.5, 0.0, 1.5, 1.0}, {0.2, 0.2, 0.8, 0.8}});
    auto loss = ciouLoss(b1, b2);
    EXPECT_EQ(loss.size(0), 2);
    EXPECT_NEAR(loss[1].item<float>(), 0.0f, 1e-6); // identical → 0
    EXPECT_GT(loss[0].item<float>(), 0.0f);           // different → >0
}

TEST(YOLOLossTest, CIoUGradientFlows) {
    auto b1 = torch::tensor({{0.0, 0.0, 1.0, 1.0}}, torch::requires_grad());
    auto b2 = torch::tensor({{0.2, 0.2, 0.8, 0.8}});
    auto loss = ciouLoss(b1, b2);
    loss.sum().backward();
    EXPECT_TRUE(b1.grad().defined());
    EXPECT_GT(b1.grad().abs().sum().item<float>(), 0.0f);
}

// ─── dflLoss ──────────────────────────────────────────────────

TEST(YOLOLossTest, DFLExactBin) {
    // target at exact bin center → only that bin has weight
    auto pred = torch::tensor({{-1.0, 0.0, 2.0, 1.0, -0.5}});
    auto target = torch::tensor({2.0}); // exact bin 2
    auto loss = dflLoss(pred, target, 5);
    EXPECT_GT(loss.item<float>(), 0.0f);
    EXPECT_LT(loss.item<float>(), 5.0f);
}

TEST(YOLOLossTest, DFLBetweenBins) {
    // target between bins 2 and 3 → both bins have weight
    auto pred = torch::tensor({{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0}});
    auto target = torch::tensor({2.5f}); // between bin 2 and 3
    auto loss = dflLoss(pred, target, 6);
    EXPECT_GT(loss.item<float>(), 0.0f);
}

TEST(YOLOLossTest, DFLMultipleSamples) {
    auto pred = torch::tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto target = torch::tensor({1.0f, 2.0f});
    auto loss = dflLoss(pred, target, 3);
    EXPECT_EQ(loss.size(0), 2);
    EXPECT_GT(loss[0].item<float>(), 0.0f);
    EXPECT_GT(loss[1].item<float>(), 0.0f);
}

TEST(YOLOLossTest, DFLPerfectPrediction) {
    // pred exactly matches target distribution (regMax=1 → trivial)
    auto pred = torch::tensor({{0.0f, 10.0f, 0.0f}});
    auto target = torch::tensor({1.0f});
    auto loss = dflLoss(pred, target, 3);
    EXPECT_NEAR(loss.item<float>(), 0.0f, 0.1f); // near-zero when confident
}

TEST(YOLOLossTest, DFLGradientFlows) {
    auto pred = torch::tensor({{1.0, 2.0, 3.0, 4.0}}, torch::requires_grad());
    auto target = torch::tensor({1.5f});
    auto loss = dflLoss(pred, target, 4);
    loss.sum().backward();
    EXPECT_TRUE(pred.grad().defined());
    EXPECT_GT(pred.grad().abs().sum().item<float>(), 0.0f);
}

// ─── bceLoss ──────────────────────────────────────────────────

TEST(YOLOLossTest, BCEPerfectPrediction) {
    auto pred = torch::tensor({{100.0f, -100.0f}}); // sigmoid ≈ (1, 0)
    auto target = torch::tensor({{1.0f, 0.0f}});
    auto loss = bceLoss(pred, target);
    EXPECT_NEAR(loss.item<float>(), 0.0f, 0.1f);
}

TEST(YOLOLossTest, BCEEqualProb) {
    // pred=0 → sigmoid=0.5 → BCE = -ln(0.5) ≈ 0.693
    auto pred = torch::tensor({{0.0f, 0.0f}});
    auto target = torch::tensor({{1.0f, 0.0f}});
    auto loss = bceLoss(pred, target);
    EXPECT_NEAR(loss.item<float>(), 0.693f, 0.01f);
}

TEST(YOLOLossTest, BCEWorstCase) {
    // confident wrong → high loss
    auto pred = torch::tensor({{-100.0f, 100.0f}});
    auto target = torch::tensor({{1.0f, 0.0f}});
    auto loss = bceLoss(pred, target);
    EXPECT_GT(loss.item<float>(), 10.0f);
}

TEST(YOLOLossTest, BCEMultipleLabels) {
    auto pred = torch::tensor({{0.5f, -0.5f}, {-1.0f, 2.0f}});
    auto target = torch::tensor({{1.0f, 0.0f}, {0.0f, 1.0f}});
    auto loss = bceLoss(pred, target);
    EXPECT_EQ(loss.sizes(), torch::IntArrayRef{});
    EXPECT_GT(loss.item<float>(), 0.0f);
}

TEST(YOLOLossTest, BCEGradientFlows) {
    auto pred = torch::tensor({{0.5f, -0.5f}}, torch::requires_grad());
    auto target = torch::tensor({{1.0f, 0.0f}});
    auto loss = bceLoss(pred, target);
    loss.backward();
    EXPECT_TRUE(pred.grad().defined());
    EXPECT_GT(pred.grad().abs().sum().item<float>(), 0.0f);
}

// ─── YOLOLoss (integrated) ────────────────────────────────────

TEST(YOLOLossTest, YOLOLossWithTargets) {
    YOLOLoss loss;
    // mock 3-scale predictions
    auto p1 = torch::zeros({1, 128, 8, 8});   // P3 scale
    auto p2 = torch::zeros({1, 128, 4, 4});   // P4 scale
    auto p3 = torch::zeros({1, 128, 2, 2});   // P5 scale
    // single target (batch=0, cls=1, cx=0.5, cy=0.5, w=0.2, h=0.4)
    auto targets = torch::tensor({{0.0f, 1.0f, 0.5f, 0.5f, 0.2f, 0.4f}});
    auto out = loss({p1, p2, p3}, targets);
    EXPECT_TRUE(out.totalLoss.defined());
    EXPECT_GT(out.totalLoss.item<float>(), 0.0f);
}

TEST(YOLOLossTest, YOLOLossEmptyTargets) {
    YOLOLoss loss;
    auto p1 = torch::zeros({1, 128, 8, 8});
    auto p2 = torch::zeros({1, 128, 4, 4});
    auto p3 = torch::zeros({1, 128, 2, 2});
    auto targets = torch::empty({0, 6});
    auto out = loss({p1, p2, p3}, targets);
    EXPECT_TRUE(out.totalLoss.defined());
    EXPECT_NEAR(out.totalLoss.item<float>(), 0.0f, 1e-6);
}

TEST(YOLOLossTest, YOLOLossEmptyPredictions) {
    YOLOLoss loss;
    auto targets = torch::tensor({{0.0f, 1.0f, 0.5f, 0.5f, 0.2f, 0.4f}});
    auto out = loss({}, targets);
    EXPECT_TRUE(out.totalLoss.defined());
    EXPECT_NEAR(out.totalLoss.item<float>(), 0.0f, 1e-6);
}

TEST(YOLOLossTest, YOLOLossGradientFlows) {
    YOLOLoss loss;
    auto p1 = torch::zeros({1, 128, 8, 8}, torch::requires_grad());
    auto p2 = torch::zeros({1, 128, 4, 4}, torch::requires_grad());
    auto p3 = torch::zeros({1, 128, 2, 2}, torch::requires_grad());
    auto targets = torch::tensor({{0.0f, 1.0f, 0.5f, 0.5f, 0.2f, 0.4f}});
    auto out = loss({p1, p2, p3}, targets);
    out.totalLoss.backward();
    EXPECT_TRUE(p1.grad().defined());
    EXPECT_GT(p1.grad().abs().sum().item<float>(), 0.0f);
}
