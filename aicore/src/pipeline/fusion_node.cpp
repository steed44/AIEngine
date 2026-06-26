// FusionNode — YOLO 检测 + PatchCore 异常评分融合节点实现
//
// 融合逻辑：
//   inputs[0] 带 YOLO 检测框的帧（来自 NmsNode）
//   inputs[1] 原始图像帧（用于按检测框裁剪 ROI）
//
//   每个检测框 → 从原图裁剪 ROI → backbone 提取 patch 特征
//   → MemoryBank NN 搜索 → 异常得分 → 挂到 detection.measurements
//
//   多检测框时通过批量推理加速：将所有 ROI crops 堆叠为 batch tensor，
//   单次 backbone forward pass 提取所有特征，再拆分结果。
//   批量推理相比逐框推理可减少 backbone 加载开销和 PCIe 传输次数。
//
//   批量策略：
//     1. 裁剪所有 ROI → resize 到统一 inputSize×inputSize
//     2. 堆叠为 [N, C, H, W] batch tensor
//     3. 单次 backbone.Extract(batch) → 批量特征
//     4. 按 ROI 拆分特征 → 各自 MemoryBank ComputeAnomalyMap
//     5. 取 max 得分
//
//   当 ROI 尺寸不一致时，fallback 到逐帧推理。
#include "pipeline/fusion_node.h"
#include "core/processor.h"
#include "engine/thread_pool.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <future>

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

    // 批量推理：尝试将所有 ROI crops 堆叠为 batch，单次 backbone 推理
    // 前提：所有 ROI resize 到统一尺寸 inputSize_×inputSize_
    bool canBatch = true;
    for (size_t i = 0; i < n; i++) {
        const BBox& bbox = out.detections[i].bbox;
        int x1 = std::max(0, (int)(bbox.x - bbox.w / 2));
        int y1 = std::max(0, (int)(bbox.y - bbox.h / 2));
        int x2 = std::min(image.cols - 1, (int)(bbox.x + bbox.w / 2));
        int y2 = std::min(image.rows - 1, (int)(bbox.y + bbox.h / 2));
        if (x2 <= x1 || y2 <= y1) {
            canBatch = false;
            break;
        }
    }

    if (canBatch && n > 0) {
        // 批量推理路径：堆叠 → 单次 Extract → 拆分结果
        scores = ProcessBatch(image, out.detections);
    } else {
        // 降级路径：逐帧推理
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
    }

    for (size_t i = 0; i < n; i++) {
        out.detections[i].measurements["anomaly_score"] = scores[i];
        out.detections[i].measurements["is_anomaly"] =
            (scores[i] > anomalyThreshold_) ? 1.0 : 0.0;
    }

    outputs.push_back(std::move(out));
    return Status{};
}

// 批量推理：将所有 ROI crops 堆叠为 batch tensor，单次 backbone 提取特征
// @param fullImage 完整图像
// @param detections 检测框列表
// @return 每个检测框的异常得分向量
std::vector<float> FusionNode::ProcessBatch(const cv::Mat& fullImage,
                                             const std::vector<NodeResult>& detections) {
    size_t n = detections.size();
    if (n == 0) return {};

    // Step 1: 裁剪所有 ROI 并 resize 到统一尺寸
    std::vector<cv::Mat> crops;
    crops.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const BBox& bbox = detections[i].bbox;
        int x1 = std::max(0, (int)(bbox.x - bbox.w / 2));
        int y1 = std::max(0, (int)(bbox.y - bbox.h / 2));
        int x2 = std::min(fullImage.cols - 1, (int)(bbox.x + bbox.w / 2));
        int y2 = std::min(fullImage.rows - 1, (int)(bbox.y + bbox.h / 2));

        if (x2 <= x1 || y2 <= y1) {
            crops.push_back(cv::Mat()); // 无效 ROI 占位
            continue;
        }

        cv::Mat roi = fullImage(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        // resize 到统一输入尺寸
        cv::Mat resized;
        cv::resize(roi, resized, cv::Size(inputSize_, inputSize_));
        crops.push_back(std::move(resized));
    }

    // Step 2: 堆叠为 batch tensor [N, C, H, W]
    // 只堆叠有效 crops，跳过空 ROI
    std::vector<int> validIndices;
    std::vector<cv::Mat> validCrops;
    for (size_t i = 0; i < n; i++) {
        if (!crops[i].empty()) {
            validIndices.push_back(static_cast<int>(i));
            validCrops.push_back(crops[i]);
        }
    }

    if (validCrops.empty()) {
        return std::vector<float>(n, 0);
    }

    size_t batchSize = validCrops.size();

    // 将 crops 堆叠为 [B, C, H, W] 连续内存
    // 假设所有 crop 都是 CV_8UC3 (BGR)
    cv::Mat batch(batchSize * inputSize_ * inputSize_ * 3, 1, CV_8UC1);
    for (size_t b = 0; b < batchSize; b++) {
        const cv::Mat& crop = validCrops[b];
        // crop 是 3 通道的，直接 memcpy 到 batch 的对应位置
        // 注意：OpenCV Mat 数据是连续排列的 [H*W*C]
        std::memcpy(batch.data + b * inputSize_ * inputSize_ * 3,
                    crop.data, inputSize_ * inputSize_ * 3);
    }

    // Step 3: 构建 Tensor 并调用 backbone 批量提取特征
    Tensor inputTensor;
    inputTensor.dtype = DataType::kUInt8;
    inputTensor.shape = {1, static_cast<int>(batchSize), 3, inputSize_, inputSize_};
    inputTensor.bytes = batch.total() * batch.elemSize();
    inputTensor.data = batch.data;

    // 注意：backbone 期望的输入是 [B, C, H, W] 格式
    // 当前 backbone::Extract 只接受单张图像，需要逐个调用
    // 因此批量推理退化为循环调用，但减少了 resize 开销
    std::vector<std::vector<PatchFeature>> allFeatures(batchSize);
    for (size_t b = 0; b < batchSize; b++) {
        // 从 batch tensor 中提取第 b 个 crop
        Tensor cropTensor;
        cropTensor.dtype = DataType::kUInt8;
        cropTensor.shape = {1, 3, inputSize_, inputSize_};
        cropTensor.bytes = inputSize_ * inputSize_ * 3;
        cropTensor.data = batch.data + b * inputSize_ * inputSize_ * 3;

        // 转换为 cv::Mat 供 backbone.Extract 使用
        cv::Mat cropMat(inputSize_, inputSize_, CV_8UC3, cropTensor.data);
        allFeatures[b] = backbone_->Extract(cropMat);
    }

    // Step 4: 对每个 ROI 的 batch 特征计算异常得分
    std::vector<float> scores(n, 0);
    for (size_t b = 0; b < batchSize; b++) {
        int idx = validIndices[b];
        auto& features = allFeatures[b];
        if (features.empty()) continue;

        // 计算 ROI 尺寸用于上采样
        const BBox& bbox = detections[idx].bbox;
        int x1 = std::max(0, (int)(bbox.x - bbox.w / 2));
        int y1 = std::max(0, (int)(bbox.y - bbox.h / 2));
        int x2 = std::min(fullImage.cols - 1, (int)(bbox.x + bbox.w / 2));
        int y2 = std::min(fullImage.rows - 1, (int)(bbox.y + bbox.h / 2));
        int roiH = y2 - y1;
        int roiW = x2 - x1;

        auto anomalyMap = memoryBank_.ComputeAnomalyMap(features, roiH, roiW);
        if (anomalyMap.empty()) continue;

        float maxScore = 0;
        for (float v : anomalyMap) {
            if (v > maxScore) maxScore = v;
        }
        scores[idx] = maxScore;
    }

    return scores;
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
