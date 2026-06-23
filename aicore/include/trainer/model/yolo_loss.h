#pragma once
#include <torch/nn.h>
#include <torch/serialize.h>
#include <vector>

namespace aicore {

// CIoU: 1 - IoU + rho^2/c^2 + alpha*v, v = (4/pi^2)*(atan(w_gt/h_gt)-atan(w/h))^2
torch::Tensor ciouLoss(
    const torch::Tensor& boxes,
    const torch::Tensor& targets);

// DFL: cross-entropy on softmax(reg_max distribution → continuous position)
torch::Tensor dflLoss(
    const torch::Tensor& pred,    // [N, regMax] 原始 logits
    const torch::Tensor& target,  // [N] 连续值 0..regMax-1
    int regMax = 16);

// BCE with sigmoid (multi-label)
torch::Tensor bceLoss(
    const torch::Tensor& pred,    // [N, C] logits
    const torch::Tensor& target); // [N, C] {0,1}

struct YOLOLossOutput {
    torch::Tensor totalLoss;
    torch::Tensor boxLoss;
    torch::Tensor clsLoss;
    torch::Tensor dflLoss;
};

struct YOLOLossConfig {
    float boxWeight = 7.5f;
    float clsWeight = 0.5f;
    float dflWeight = 1.5f;
    int regMax = 16;
    std::vector<int> strides = { 8, 16, 32 };
};

// YOLOv8 综合 Loss: assigner + CIoU + DFL + BCE 加权聚合
class YOLOLoss {
public:
    explicit YOLOLoss(const YOLOLossConfig& cfg = YOLOLossConfig{});

    YOLOLossOutput operator()(
        const std::vector<torch::Tensor>& preds,  // [3][B, C, H, W]
        const torch::Tensor& targets);             // [N, 6] (bi,cls,cx,cy,w,h)

private:
    struct AssignResult {
        torch::Tensor targetBox;    // [M, 4] x1y1x2y2
        torch::Tensor targetCls;    // [M]
        torch::Tensor targetScore;  // [M] task-aligned score
        std::vector<int> batchIdx;  // [M]
    };

    AssignResult assignTargets(
        const std::vector<torch::Tensor>& preds,
        const torch::Tensor& targets);

    YOLOLossConfig cfg_;
};

torch::Tensor boxCxwhToXyxy(const torch::Tensor& x);
torch::Tensor computeIoU(const torch::Tensor& b1, const torch::Tensor& b2);

} // namespace aicore
