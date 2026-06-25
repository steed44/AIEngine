#include "postprocess/yolo_decode_node.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace aicore {

Status YoloDecodeNode::Init(const NodeConfig& config) {
    auto it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("num_classes");
    if (it != config.end()) numClasses_ = std::stoi(it->second);
    it = config.find("version");
    if (it != config.end()) versionStr_ = it->second;
    it = config.find("model_input_size");
    if (it != config.end()) modelInputSize_ = std::stoi(it->second);
    it = config.find("name");
    if (it != config.end()) name_ = it->second;
    return Status{};
}

void YoloDecodeNode::DecodeScale(
    const float* data, int numBoxes, int stride,
    int gridW, int gridH, int numClasses,
    float scale, int padX, int padY,
    std::vector<NodeResult>& candidates) const {

    int regMax = 16;
    bool isV8 = (versionStr_ == "v8");

    for (int i = 0; i < numBoxes; i++) {
        const float* cellData = data + i * stride;
        float cx, cy, w, h;

        if (isV8) {
            auto dfl = [&](const float* dist) -> float {
                float maxVal = dist[0];
                for (int j = 1; j < regMax; j++)
                    if (dist[j] > maxVal) maxVal = dist[j];
                float sum = 0, weighted = 0;
                for (int j = 0; j < regMax; j++) {
                    float e = std::exp(dist[j] - maxVal);
                    sum += e;
                    weighted += e * j;
                }
                return weighted / (sum + 1e-9f);
            };

            float l = dfl(cellData);
            float t = dfl(cellData + regMax);
            float r = dfl(cellData + 2 * regMax);
            float b = dfl(cellData + 3 * regMax);

            int gridIdx = i;
            int gx = gridIdx % gridW;
            int gy = gridIdx / gridW;

            float x1 = gx + 0.5f - l;
            float y1 = gy + 0.5f - t;
            float x2 = gx + 0.5f + r;
            float y2 = gy + 0.5f + b;
            cx = (x1 + x2) / 2.0f;
            cy = (y1 + y2) / 2.0f;
            w = x2 - x1;
            h = y2 - y1;

            const float* clsLogits = cellData + 4 * regMax;
            float maxScore = 0;
            int bestLabel = 0;
            for (int c = 0; c < numClasses; c++) {
                float score = 1.0f / (1.0f + std::exp(-clsLogits[c]));
                if (score > maxScore) {
                    maxScore = score;
                    bestLabel = c;
                }
            }

            if (maxScore < confidenceThreshold_)
                continue;

            float gridStride = (float)modelInputSize_ / gridW;
            cx *= gridStride;
            cy *= gridStride;
            w *= gridStride;
            h *= gridStride;

            cx = (cx - padX) / scale;
            cy = (cy - padY) / scale;
            w = w / scale;
            h = h / scale;

            NodeResult det;
            det.label = std::to_string(bestLabel);
            det.confidence = maxScore;
            det.bbox.x = cx;
            det.bbox.y = cy;
            det.bbox.w = w;
            det.bbox.h = h;
            candidates.push_back(std::move(det));

        } else {
            cx = cellData[0];
            cy = cellData[1];
            w = cellData[2];
            h = cellData[3];

            float maxScore = 0;
            int bestLabel = 0;
            for (int c = 0; c < numClasses; c++) {
                float score = 1.0f / (1.0f + std::exp(-cellData[4 + c]));
                if (score > maxScore) {
                    maxScore = score;
                    bestLabel = c;
                }
            }

            if (maxScore < confidenceThreshold_)
                continue;

            cx = (cx - padX) / scale;
            cy = (cy - padY) / scale;
            w = w / scale;
            h = h / scale;

            NodeResult det;
            det.label = std::to_string(bestLabel);
            det.confidence = maxScore;
            det.bbox.x = cx;
            det.bbox.y = cy;
            det.bbox.w = w;
            det.bbox.h = h;
            candidates.push_back(std::move(det));
        }
    }
}

void YoloDecodeNode::NMS(std::vector<NodeResult>& candidates,
                          float iouThreshold) {
    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.confidence > b.confidence; });

    std::vector<bool> keep(candidates.size(), true);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (!keep[i]) continue;
        const auto& ai = candidates[i].bbox;
        float ax1 = ai.x - ai.w / 2, ay1 = ai.y - ai.h / 2;
        float ax2 = ai.x + ai.w / 2, ay2 = ai.y + ai.h / 2;

        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (!keep[j]) continue;
            if (candidates[i].label != candidates[j].label) continue;

            const auto& bj = candidates[j].bbox;
            float bx1 = bj.x - bj.w / 2, by1 = bj.y - bj.h / 2;
            float bx2 = bj.x + bj.w / 2, by2 = bj.y + bj.h / 2;

            float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
            float iy = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
            float inter = ix * iy;
            float areaA = ai.w * ai.h, areaB = bj.w * bj.h;
            float iou = inter / (areaA + areaB - inter + 1e-6f);

            if (iou > iouThreshold)
                keep[j] = false;
        }
    }

    size_t writeIdx = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (keep[i])
            candidates[writeIdx++] = std::move(candidates[i]);
    }
    candidates.resize(writeIdx);
}

Status YoloDecodeNode::Process(const std::vector<Frame>& inputs,
                                std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "yolo_decode: no input"};

    for (const auto& frame : inputs) {
        float scale = 1.0f;
        float padX = 0, padY = 0;
        auto itScale = frame.roiMap.find("letterbox_scale");
        if (itScale != frame.roiMap.end()) scale = itScale->second;
        auto itPadX = frame.roiMap.find("letterbox_pad_x");
        if (itPadX != frame.roiMap.end()) padX = itPadX->second;
        auto itPadY = frame.roiMap.find("letterbox_pad_y");
        if (itPadY != frame.roiMap.end()) padY = itPadY->second;

        std::vector<NodeResult> allCandidates;

        for (const auto& tensor : frame.rawOutputs) {
            if (tensor.shape.size() != 4 || tensor.shape[0] != 1)
                continue;
            if (!tensor.data || tensor.bytes == 0)
                continue;

            int C = (int)tensor.shape[1];
            int H = (int)tensor.shape[2];
            int W = (int)tensor.shape[3];
            int numBoxes = H * W;
            int stride = C;

            const float* data = static_cast<const float*>(tensor.data);
            DecodeScale(data, numBoxes, stride, W, H,
                        numClasses_, scale, (int)padX, (int)padY,
                        allCandidates);
        }

        NMS(allCandidates, iouThreshold_);

        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;
        out.sourceId = frame.sourceId;
        out.detections = std::move(allCandidates);
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string YoloDecodeNode::GetName() const { return name_; }
std::string YoloDecodeNode::GetType() const { return "yolo_decode"; }

} // namespace aicore
