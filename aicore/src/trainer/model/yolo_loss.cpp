#include "trainer/model/yolo_loss.h"
#include <cmath>

namespace aicore {

// ============================================================
// boxCxwhToXyxy: [N,4] cxcywh → x1y1x2y2 (归一化坐标)
// ============================================================
torch::Tensor boxCxwhToXyxy(const torch::Tensor& x) {
    auto cx = x.index({"...", 0});
    auto cy = x.index({"...", 1});
    auto w  = x.index({"...", 2});
    auto h  = x.index({"...", 3});
    auto halfW = w / 2;
    auto halfH = h / 2;
    return torch::stack({ cx - halfW, cy - halfH, cx + halfW, cy + halfH }, -1);
}

// ============================================================
// computeIoU: [N,4] vs [N,4] → [N] IoU
// ============================================================
torch::Tensor computeIoU(const torch::Tensor& b1, const torch::Tensor& b2) {
    auto interX1 = torch::max(b1.index({"...", 0}), b2.index({"...", 0}));
    auto interY1 = torch::max(b1.index({"...", 1}), b2.index({"...", 1}));
    auto interX2 = torch::min(b1.index({"...", 2}), b2.index({"...", 2}));
    auto interY2 = torch::min(b1.index({"...", 3}), b2.index({"...", 3}));
    auto interW = torch::clamp(interX2 - interX1, 0);
    auto interH = torch::clamp(interY2 - interY1, 0);
    auto interArea = interW * interH;

    auto area1 = (b1.index({"...", 2}) - b1.index({"...", 0})) *
                 (b1.index({"...", 3}) - b1.index({"...", 1}));
    auto area2 = (b2.index({"...", 2}) - b2.index({"...", 0})) *
                 (b2.index({"...", 3}) - b2.index({"...", 1}));
    auto unionArea = area1 + area2 - interArea;
    return interArea / (unionArea + 1e-7);
}

// ============================================================
// ciouLoss: 1 - IoU + rho^2/c^2 + alpha*v
// ============================================================
torch::Tensor ciouLoss(
    const torch::Tensor& boxes,
    const torch::Tensor& targets) {

    auto iou = computeIoU(boxes, targets);

    auto b1X1 = boxes.index({"...", 0});
    auto b1Y1 = boxes.index({"...", 1});
    auto b1X2 = boxes.index({"...", 2});
    auto b1Y2 = boxes.index({"...", 3});
    auto b2X1 = targets.index({"...", 0});
    auto b2Y1 = targets.index({"...", 1});
    auto b2X2 = targets.index({"...", 2});
    auto b2Y2 = targets.index({"...", 3});

    // center distance rho^2
    auto cxb1 = (b1X1 + b1X2) / 2;
    auto cyb1 = (b1Y1 + b1Y2) / 2;
    auto cxb2 = (b2X1 + b2X2) / 2;
    auto cyb2 = (b2Y1 + b2Y2) / 2;
    auto rho2 = (cxb1 - cxb2).pow(2) + (cyb1 - cyb2).pow(2);

    // diagonal of smallest enclosing box c^2
    auto encloseX1 = torch::min(b1X1, b2X1);
    auto encloseY1 = torch::min(b1Y1, b2Y1);
    auto encloseX2 = torch::max(b1X2, b2X2);
    auto encloseY2 = torch::max(b1Y2, b2Y2);
    auto c2 = (encloseX2 - encloseX1).pow(2) + (encloseY2 - encloseY1).pow(2) + 1e-7;

    // v = (4/pi^2) * (atan(w_gt/h_gt) - atan(w/h))^2
    auto wPred = b1X2 - b1X1;
    auto hPred = b1Y2 - b1Y1;
    auto wGt = b2X2 - b2X1;
    auto hGt = b2Y2 - b2Y1;
    auto v = (4 / (M_PI * M_PI)) * (torch::atan(wGt / (hGt + 1e-7)) - torch::atan(wPred / (hPred + 1e-7))).pow(2);

    auto alpha = v / (1 - iou + v + 1e-7);

    return 1 - iou + rho2 / c2 + alpha * v;
}

// ============================================================
// dflLoss: cross-entropy on softmax(reg_max distribution)
// ============================================================
torch::Tensor dflLoss(
    const torch::Tensor& pred,
    const torch::Tensor& target,
    int regMax) {

    // pred: [N, regMax] → softmax → [N, regMax]
    auto prob = torch::softmax(pred, -1);

    // 对每个 target 位置，找到左右临近的 bin
    auto targetClamped = torch::clamp(target, 0, regMax - 1);
    auto left = targetClamped.to(torch::kLong);
    auto right = torch::clamp(left + 1, 0, regMax - 1);

    // 线性插值权重: weight_right = target - left, weight_left = 1 - weight_right
    auto weightRight = targetClamped - left.to(torch::kFloat);
    auto weightLeft = 1 - weightRight;

    // Cross-entropy: sum(-weight * log(prob))
    auto lossLeft = -weightLeft * torch::log(prob.gather(1, left.unsqueeze(-1)).squeeze(-1) + 1e-7);
    auto lossRight = -weightRight * torch::log(prob.gather(1, right.unsqueeze(-1)).squeeze(-1) + 1e-7);

    return lossLeft + lossRight;
}

// ============================================================
// bceLoss: binary cross-entropy with sigmoid
// ============================================================
torch::Tensor bceLoss(
    const torch::Tensor& pred,
    const torch::Tensor& target) {

    // BCE with sigmoid: -target * log(sigmoid(pred)) - (1-target) * log(1-sigmoid(pred))
    // Numerically stable: max(pred, 0) - pred * target + log(1 + exp(-abs(pred)))
    return torch::mean(
        torch::max(pred, torch::zeros_like(pred))
        - pred * target
        + torch::log(1 + torch::exp(-torch::abs(pred))));
}

// ============================================================
// YOLOLoss
// ============================================================
YOLOLoss::YOLOLoss(const YOLOLossConfig& cfg) : cfg_(cfg) {}

YOLOLoss::AssignResult YOLOLoss::assignTargets(
    const std::vector<torch::Tensor>& preds,
    const torch::Tensor& targets) {

    AssignResult result;
    if (targets.size(0) == 0) {
        result.targetBox = torch::empty({0, 4});
        result.targetCls = torch::empty({0});
        result.targetScore = torch::empty({0});
        return result;
    }

    // 简单分配: 每个 GT 分配到 stride 最接近的尺度
    // 对于 YOLOv8 的简化实现，分配所有 GT 到所有尺度的所有 grid
    // 实际 YOLOv8 使用 TaskAlignedAssigner
    // 这里做简化: 按比例分配 targets 到各尺度

    auto bi = targets.index({"...", 0}).to(torch::kLong);
    auto cls = targets.index({"...", 1}).to(torch::kLong);
    auto boxes = targets.index({"...", torch::indexing::Slice(2, 6)}); // cx,cy,w,h
    auto xyxy = boxCxwhToXyxy(boxes);

    result.targetBox = xyxy;
    result.targetCls = cls;
    result.targetScore = torch::ones_like(cls.to(torch::kFloat));
    {
        auto biCpu = bi.cpu();
        auto* ptr = biCpu.data_ptr<int64_t>();
        result.batchIdx.assign(ptr, ptr + biCpu.size(0));
    }

    return result;
}

YOLOLossOutput YOLOLoss::operator()(
    const std::vector<torch::Tensor>& preds,
    const torch::Tensor& targets) {

    auto assign = assignTargets(preds, targets);
    auto numTargets = assign.targetBox.size(0);
    auto device = targets.device();

    if (numTargets == 0 || preds.empty()) {
        auto zero = torch::full({}, 0.0,
            torch::TensorOptions().device(device).requires_grad(true));
        return { zero, zero, zero, zero };
    }

    std::vector<torch::Tensor> allPreds;
    for (size_t i = 0; i < preds.size(); ++i) {
        allPreds.push_back(preds[i].flatten(2).transpose(1, 2));
    }
    auto merged = torch::cat(allPreds, 1);
    auto mergedFlat = merged.flatten(0, 1);
    auto takeN = std::min(numTargets, mergedFlat.size(0));
    auto boxCoords = mergedFlat.index({torch::indexing::Slice(0, takeN), torch::indexing::Slice(0, 4)});
    auto targetBox = assign.targetBox.index({torch::indexing::Slice(0, takeN)});
    auto lossBox = ciouLoss(boxCoords, targetBox).mean();

    auto lossDfl = torch::full({}, 0.0,
        torch::TensorOptions().device(device));
    auto lossCls = torch::full({}, 0.0,
        torch::TensorOptions().device(device));

    auto total = cfg_.boxWeight * lossBox + cfg_.clsWeight * lossCls + cfg_.dflWeight * lossDfl;

    return { total, lossBox, lossCls, lossDfl };
}

} // namespace aicore
