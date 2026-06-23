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
    int64_t numTargets = targets.size(0);
    int numScales = (int)preds.size();

    if (numTargets == 0 || numScales == 0) {
        result.targetBox = torch::empty({0, 4});
        result.targetCls = torch::empty({0});
        result.targetScore = torch::empty({0});
        result.flatIdx = torch::empty({0}, torch::kLong);
        return result;
    }

    auto device = targets.device();

    int64_t totalGrids = 0;
    std::vector<int64_t> gridsPerScale(numScales);
    std::vector<int64_t> cumPrev(numScales + 1, 0);
    for (int i = 0; i < numScales; ++i) {
        gridsPerScale[i] = preds[i].size(2) * preds[i].size(3);
        cumPrev[i + 1] = cumPrev[i] + gridsPerScale[i];
    }
    totalGrids = cumPrev[numScales];
    int64_t batchSize = preds[0].size(0);

    std::vector<float> boxBuf, clsBuf, scoreBuf, gcxBuf, gcyBuf;
    std::vector<int64_t> idxBuf, strideBuf;
    std::vector<int> batchBuf;

    for (int64_t ti = 0; ti < numTargets; ++ti) {
        auto t = targets[ti];
        int64_t bi = (int64_t)t[0].item<float>();
        int64_t clsid = (int64_t)t[1].item<float>();
        float cx = t[2].item<float>();
        float cy = t[3].item<float>();
        float w  = t[4].item<float>();
        float h  = t[5].item<float>();

        for (int si = 0; si < numScales; ++si) {
            int64_t H = preds[si].size(2);
            int64_t W = preds[si].size(3);
            int64_t gx = (int64_t)(cx * W);
            int64_t gy = (int64_t)(cy * H);
            if (gx < 0 || gx >= W || gy < 0 || gy >= H) continue;

            int64_t idx = bi * totalGrids + cumPrev[si] + gy * W + gx;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            float gridNormCx = (gx + 0.5f) / W;
            float gridNormCy = (gy + 0.5f) / H;

            idxBuf.push_back(idx);
            boxBuf.insert(boxBuf.end(), { x1, y1, x2, y2 });
            clsBuf.push_back((float)clsid);
            scoreBuf.push_back(1.0f);
            batchBuf.push_back((int)bi);
            gcxBuf.push_back(gridNormCx);
            gcyBuf.push_back(gridNormCy);
            strideBuf.push_back(cfg_.strides[si]);
        }
    }

    if (idxBuf.empty()) {
        result.targetBox = torch::empty({0, 4}, device).to(device);
        result.targetCls = torch::empty({0}, torch::kLong).to(device);
        result.targetScore = torch::empty({0}, device);
        result.flatIdx = torch::empty({0}, torch::kLong).to(device);
        result.gridCx = torch::empty({0}, device);
        result.gridCy = torch::empty({0}, device);
        result.strides = torch::empty({0}, device);
        return result;
    }

    result.flatIdx = torch::from_blob(idxBuf.data(), { (int64_t)idxBuf.size() },
        torch::TensorOptions(torch::kLong)).clone().to(device);
    {
        auto rawBox = torch::from_blob(boxBuf.data(), { (int64_t)(boxBuf.size() / 4), 4 });
        auto rawCls = torch::from_blob(clsBuf.data(), { (int64_t)clsBuf.size() });
        auto rawScore = torch::from_blob(scoreBuf.data(), { (int64_t)scoreBuf.size() });
        auto rawGcx = torch::from_blob(gcxBuf.data(), { (int64_t)gcxBuf.size() });
        auto rawGcy = torch::from_blob(gcyBuf.data(), { (int64_t)gcyBuf.size() });
        auto rawStride = torch::from_blob(strideBuf.data(), { (int64_t)strideBuf.size() },
            torch::TensorOptions(torch::kLong));
        result.targetBox = rawBox.clone().to(device);
        result.targetCls = rawCls.clone().to(device).to(torch::kLong);
        result.targetScore = rawScore.clone().to(device);
        result.gridCx = rawGcx.clone().to(device);
        result.gridCy = rawGcy.clone().to(device);
        result.strides = rawStride.clone().to(device);
    }
    result.batchIdx = std::move(batchBuf);

    return result;
}

YOLOLossOutput YOLOLoss::operator()(
    const std::vector<torch::Tensor>& preds,
    const torch::Tensor& targets) {

    auto assign = assignTargets(preds, targets);
    int64_t numAssign = assign.targetBox.size(0);
    auto device = targets.device();

    if (numAssign == 0 || preds.empty()) {
        auto zero = torch::full({}, 0.0,
            torch::TensorOptions().device(device).requires_grad(true));
        return { zero, zero, zero, zero };
    }

    int regMax = cfg_.regMax;
    int no = 4 * regMax + cfg_.numClasses;

    // [B*totalGrids, no]
    std::vector<torch::Tensor> allPreds;
    for (size_t i = 0; i < preds.size(); ++i) {
        allPreds.push_back(preds[i].flatten(2).transpose(1, 2));
    }
    auto merged = torch::cat(allPreds, 1);
    auto mergedFlat = merged.flatten(0, 1);

    auto flatIdx = assign.flatIdx;

    // 1. 提取回归分布 [M, 4*regMax] 和分类预测 [M, nc]
    auto predReg = mergedFlat.index({flatIdx, torch::indexing::Slice(0, 4 * regMax)});
    auto predCls = mergedFlat.index({flatIdx,
        torch::indexing::Slice(4 * regMax, no)});

    // 2. DFL 解码: [M, 4*regMax] → [M, 4] (l, r, t, b 网格单位偏移)
    auto regDist = predReg.reshape({numAssign, 4, regMax});
    auto dflProb = torch::softmax(regDist, 2);
    auto proj = torch::arange(0, regMax, torch::kFloat).to(device);
    // dflVals: [M, 4] 连续值 0..regMax-1, 网格单位
    auto dflVals = (dflProb * proj).sum(2);

    // 3. DFL 网格单位偏移 → 归一化 x1y1x2y2
    // dflVals 是网格单位 (0..regMax-1), 一个网格 = stride 像素
    // 归一化: 一个网格 = 1/W (归一化宽度), 所以: 归一化偏移 = dflVals / W
    // W = imgSize / stride, 可从前两个尺度间的比例推算: W0 * stride[0] = imgSize
    float imgSize = (float)(preds[0].size(3) * cfg_.strides[0]);
    auto gridW = torch::full_like(assign.strides.to(torch::kFloat),
        imgSize) / assign.strides.to(torch::kFloat);
    auto gridH = gridW.clone();

    auto lOff = dflVals.index({"...", 0}) / gridW;
    auto tOff = dflVals.index({"...", 1}) / gridH;
    auto rOff = dflVals.index({"...", 2}) / gridW;
    auto bOff = dflVals.index({"...", 3}) / gridH;

    auto predX1 = assign.gridCx - lOff;
    auto predY1 = assign.gridCy - tOff;
    auto predX2 = assign.gridCx + rOff;
    auto predY2 = assign.gridCy + bOff;

    auto predBoxes = torch::stack({predX1, predY1, predX2, predY2}, 1);

    // 4. CIoU 损失
    auto lossBox = ciouLoss(predBoxes, assign.targetBox).mean();

    // 5. DFL 损失: 目标 box → ltrb 网格单位偏移
    auto tgX1 = assign.targetBox.index({"...", 0});
    auto tgY1 = assign.targetBox.index({"...", 1});
    auto tgX2 = assign.targetBox.index({"...", 2});
    auto tgY2 = assign.targetBox.index({"...", 3});
    auto tgL = (assign.gridCx - tgX1) * gridW;
    auto tgT = (assign.gridCy - tgY1) * gridH;
    auto tgR = (tgX2 - assign.gridCx) * gridW;
    auto tgB = (tgY2 - assign.gridCy) * gridH;
    auto targetLtrb = torch::stack({tgL, tgT, tgR, tgB}, 1);
    targetLtrb = torch::clamp(targetLtrb, 0, (float)regMax - 0.01f);

    // DFL 损失: 对 4 个边各算一次 CE + 平均
    auto dflLossTotal = torch::full({}, 0.0, torch::TensorOptions().device(device));
    for (int si = 0; si < 4; ++si) {
        dflLossTotal = dflLossTotal + dflLoss(
            regDist.index({"...", si}),  // [M, regMax]
            targetLtrb.index({"...", si}), // [M]
            regMax).mean();
    }
    auto lossDfl = dflLossTotal / 4;

    // 6. 分类损失: BCE
    auto targetOnehot = torch::zeros({numAssign, cfg_.numClasses},
        torch::TensorOptions().device(device));
    targetOnehot.scatter_(1, assign.targetCls.unsqueeze(-1), 1.0);
    auto lossCls = bceLoss(predCls, targetOnehot);

    auto total = cfg_.boxWeight * lossBox + cfg_.clsWeight * lossCls + cfg_.dflWeight * lossDfl;

    return { total, lossBox, lossCls, lossDfl };
}

} // namespace aicore
