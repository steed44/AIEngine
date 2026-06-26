#include "pipeline/fusion_node.h"
#include "core/processor.h"
#include "engine/thread_pool.h"
#include <iostream>
#include <algorithm>

namespace aicore {

FusionNode::FusionNode() {}

Status FusionNode::Init(const NodeConfig& config) {
    nodeId_ = "fusion";
    auto it = config.find("node_id");
    if (it != config.end()) nodeId_ = it->second;

    std::string backboneType = "opencv_dnn";
    it = config.find("backbone_type");
    if (it != config.end()) backboneType = it->second;

    std::string modelPath;
    it = config.find("model_path");
    if (it != config.end()) modelPath = it->second;

    std::string layers = "layer2,layer3";
    it = config.find("backbone_layers");
    if (it != config.end()) layers = it->second;

    it = config.find("input_size");
    if (it != config.end()) inputSize_ = std::stoi(it->second);

    it = config.find("anomaly_threshold");
    if (it != config.end()) anomalyThreshold_ = std::stof(it->second);

    // 创建 backbone
    NodeConfig backboneCfg;
    backboneCfg["model_path"] = modelPath;
    backboneCfg["backbone_layers"] = layers;
    backboneCfg["input_size"] = std::to_string(inputSize_);
    backbone_ = CreateBackbone(backboneType, backboneCfg);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "fusion: unknown backbone type: " + backboneType};
    }
    Status s;
    try {
        s = backbone_->Init(backboneCfg);
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorModelLoad,
            "fusion: backbone init failed: " + std::string(e.what())};
    }
    if (!s) return s;

    // 加载 MemoryBank
    it = config.find("memory_bank_path");
    if (it != config.end()) {
        s = memoryBank_.Load(it->second);
        if (!s) return s;
    } else {
        return Status{StatusCode::ErrorConfigParse,
            "fusion: missing memory_bank_path"};
    }

    initialized_ = true;
    return Status{};
}

Status FusionNode::Process(const std::vector<Frame>& inputs,
                            std::vector<Frame>& outputs) {
    if (!initialized_) {
        return Status{StatusCode::ErrorInternal,
            "fusion: not initialized"};
    }
    if (inputs.size() < 2) {
        return Status{StatusCode::ErrorInvalidInput,
            "fusion: need 2 inputs (detections + image)"};
    }

    const Frame& detFrame = inputs[0];
    const Frame& imgFrame = inputs[1];
    cv::Mat image = imgFrame.image.clone();
    if (image.empty()) {
        return Status{StatusCode::ErrorInvalidInput,
            "fusion: empty image in input[1]"};
    }

    Frame out(image.clone());
    out.detections = detFrame.detections;

    if (out.detections.empty()) {
        outputs.push_back(std::move(out));
        return Status{};
    }

    size_t n = out.detections.size();
    std::vector<float> scores(n, 0);

    if (n > 1 && threadPool_) {
        std::vector<std::future<void>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; i++) {
            futures.push_back(threadPool_->Enqueue([&, i]() {
                float score = 0;
                cv::Mat heatmap;
                ProcessOneRoi(image, out.detections[i].bbox, score, heatmap);
                scores[i] = score;
            }));
        }
        for (auto& f : futures) f.get();
    } else {
        for (size_t i = 0; i < n; i++) {
            float score = 0;
            cv::Mat heatmap;
            ProcessOneRoi(image, out.detections[i].bbox, score, heatmap);
            scores[i] = score;
        }
    }

    for (size_t i = 0; i < n; i++) {
        out.detections[i].measurements["anomaly_score"] = scores[i];
        out.detections[i].measurements["is_anomaly"] =
            (scores[i] > anomalyThreshold_) ? 1.0 : 0.0;
    }

    outputs.push_back(std::move(out));
    return Status{};
}

Status FusionNode::ProcessOneRoi(const cv::Mat& fullImage, const BBox& bbox,
                                  float& outScore, cv::Mat& outHeatmap) {
    int x1 = std::max(0, (int)(bbox.x - bbox.w / 2));
    int y1 = std::max(0, (int)(bbox.y - bbox.h / 2));
    int x2 = std::min(fullImage.cols - 1, (int)(bbox.x + bbox.w / 2));
    int y2 = std::min(fullImage.rows - 1, (int)(bbox.y + bbox.h / 2));

    if (x2 <= x1 || y2 <= y1) {
        outScore = 0;
        return Status{};
    }

    cv::Rect roiRect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat roiImg;
    try {
        roiImg = fullImage(roiRect).clone();
    } catch (const cv::Exception&) {
        outScore = 0;
        return Status{};
    }

    auto patchFeatures = backbone_->Extract(roiImg);
    if (patchFeatures.empty()) {
        outScore = 0;
        return Status{};
    }

    auto anomalyMap = memoryBank_.ComputeAnomalyMap(
        patchFeatures, roiImg.rows, roiImg.cols);
    if (anomalyMap.empty()) {
        outScore = 0;
        return Status{};
    }

    float maxScore = 0;
    for (float v : anomalyMap) {
        if (v > maxScore) maxScore = v;
    }
    outScore = maxScore;

    return Status{};
}

std::string FusionNode::GetName() const {
    return nodeId_;
}

std::string FusionNode::GetType() const {
    return "fusion";
}

void FusionNode::SetThreadPool(ThreadPool* pool) {
    threadPool_ = pool;
}

} // namespace aicore
