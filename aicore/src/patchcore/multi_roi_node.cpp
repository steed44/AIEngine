// ============================================================
// multi_roi_node.cpp — 多 ROI PatchCore 推理节点实现
// 功能：加载配置 → 加载各 ROI 模型 → 对输入大图执行
//       多区域推理 → 在原图上叠加检测结果
//
// 支持两种模式：
//   1. 固定坐标模式：从 config 加载固定 ROI 坐标
//   2. 每图模式：通过 LoadPerImageRois() 动态加载 ROI 坐标
// ============================================================
#include "patchcore/multi_roi_node.h"
#include "utils/draw_utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <future>
#include <filesystem>
#include <string>

namespace aicore {

// -------------------------------------------------------
// 初始化多 ROI 推理节点
// 配置参数：
//   - config_path: MultiRoiConfig JSON 文件路径（必需）
//   - draw_overlay: 是否在原图绘制 ROI 框（可选，"true"/"false"）
//
// 初始化流程：
//   1. 从 JSON 加载所有 ROI 定义和公共参数
//   2. 加载共享 backbone
//   3. 对每个 ROI，加载其 MemoryBank 文件
// -------------------------------------------------------
Status MultiRoiNode::Init(const NodeConfig& config) {
    auto cp = config.find("config_path");
    if (cp == config.end()) {
        return Status{StatusCode::ErrorConfigParse,
            "multi_roi: config_path required"};
    }

    // 加载 JSON 配置
    auto s = config_.FromJson(cp->second);
    if (!s) return s;

    // 可选参数：是否绘制叠加层
    auto drawIt = config.find("draw_overlay");
    if (drawIt != config.end()) {
        drawOverlay_ = (drawIt->second == "true");
    }

    // ---- 加载共享 backbone ----
    NodeConfig backboneConfig;
    backboneConfig["model_path"] = config_.backboneModelPath;
    backboneConfig["backbone_layers"] = config_.backboneLayers;
    backboneConfig["input_size"] = std::to_string(config_.inputSize);
    if (config_.backboneType == "model_backend") {
        backboneConfig["backend_type"] = config_.backendType;
    }

    backbone_ = CreateBackbone(config_.backboneType, backboneConfig);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "multi_roi: unknown backbone type: " + config_.backboneType};
    }
    s = backbone_->Init(backboneConfig);
    if (!s) return s;

    // ---- 加载各 ROI 的 MemoryBank（mmap，CPU 层） ----
    for (const auto& roi : config_.rois) {
        RoiModelSlot slot;
        slot.roiId = roi.id;
        slot.rect = cv::Rect(roi.x, roi.y, roi.w, roi.h);
        slot.anomalyThreshold = config_.anomalyThreshold;

        std::string bankPath = config_.modelDir + "/" + roi.id + ".bin";
        auto loadStatus = slot.memoryBank.Load(bankPath);
        if (!loadStatus) {
            return Status{StatusCode::ErrorModelLoad,
                "multi_roi: cannot load memory bank: " + bankPath +
                " (" + loadStatus.message + ")"};
        }

        slots_.push_back(std::move(slot));
    }

    // ---- 尝试将各 ROI 的 MemoryBank 提升至 GPU ----
    // 按 LRU 顺序依次提升；若某 Bank 超出预算则保持 CPU 层
    for (auto& slot : slots_) {
        auto promoteStatus = slot.memoryBank.PromoteToGPU();
        if (!promoteStatus) {
            std::cout << "multi_roi: ROI " << slot.roiId
                      << " stays on CPU tier (" << promoteStatus.message << ")"
                      << std::endl;
        }
    }

    return Status{};
}

// -------------------------------------------------------
// 动态加载 per-image ROI 坐标
// 根据图片文件名找到对应的 JSON，加载 ROI 坐标但不加载模型
// 模型仍从 config_.modelDir/{id}.bin 加载
// -------------------------------------------------------
Status MultiRoiNode::LoadPerImageRois(const std::string& imageFilename,
                                      const std::string& roisDir) {
    namespace fs = std::filesystem;
    std::string stem = fs::path(imageFilename).stem().string();
    std::string roiJsonPath = roisDir + "/" + stem + ".json";

    PerImageRoiConfig picRois;
    auto s = PerImageRoiConfig::FromJson(roiJsonPath, picRois);
    if (!s) {
        return Status{StatusCode::ErrorConfigParse,
            "cannot load per-image ROI config: " + roiJsonPath +
            " (" + s.message + ")"};
    }

    // 清除旧的 slots
    slots_.clear();

    // 为每个 ROI 创建 slot，加载对应的 MemoryBank
    for (const auto& roi : picRois.rois) {
        RoiModelSlot slot;
        slot.roiId = roi.id;
        slot.rect = cv::Rect(roi.x, roi.y, roi.w, roi.h);
        slot.anomalyThreshold = config_.anomalyThreshold;

        std::string bankPath = config_.modelDir + "/" + roi.id + ".bin";
        auto loadStatus = slot.memoryBank.Load(bankPath);
        if (!loadStatus) {
            return Status{StatusCode::ErrorModelLoad,
                "multi_roi: cannot load memory bank: " + bankPath +
                " (" + loadStatus.message + ")"};
        }

        slots_.push_back(std::move(slot));
    }

    if (slots_.empty()) {
        return Status{StatusCode::ErrorConfigParse,
            "no ROIs found for image: " + imageFilename};
    }

    return Status{};
}

// -------------------------------------------------------
// 推理一帧大图：并行推理所有 ROI，再串行收集结果→叠加
//
// 输出帧包含：
//   - roiMap["roi_{id}_score"] = float 异常得分
//   - roiMap["roi_{id}_anomaly"] = 1.0f（异常）或 0.0f（正常）
//   - image 上绘制了 ROI 框（绿色=正常 红色=异常）
//
// 并行策略：
//   若有 threadPool_ 且 ROI 数 > 1，每个 ROI 的
//   裁剪→Extract→MemoryBank 比对提交到线程池并发执行。
//   全部完成后在主线程串行收集结果并叠加绘制。
// -------------------------------------------------------
struct RoiInferResult {
    bool ok = false;
    std::string errorMsg;
    float score = 0;
};

Status MultiRoiNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    if (inputs.empty()) {
        return Status{StatusCode::ErrorInvalidInput, "multi_roi: no input"};
    }

    cv::Mat fullImage = inputs[0].image.clone();
    if (fullImage.empty()) {
        return Status{StatusCode::ErrorPreprocess, "multi_roi: empty image"};
    }

    Frame out(fullImage.clone());
    bool anyAnomaly = false;
    size_t n = slots_.size();

    std::vector<RoiInferResult> results(n);

    if (n > 1 && threadPool_) {
        // ---- 并行路径：通过 ThreadPool 并发推理所有 ROI ----
        std::vector<std::future<void>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; i++) {
            futures.push_back(threadPool_->Enqueue([&, i]() {
                float score = 0;
                cv::Mat heatmap;
                auto s = ProcessOneRoi(fullImage, slots_[i], score, heatmap);
                results[i].ok = (s || s.code == StatusCode::Skip);
                results[i].errorMsg = s.message;
                results[i].score = score;
            }));
        }
        for (auto& f : futures) f.get();
    } else {
        // ---- 串行退化路径（无线程池或单 ROI） ----
        for (size_t i = 0; i < n; i++) {
            float score = 0;
            cv::Mat heatmap;
            auto s = ProcessOneRoi(fullImage, slots_[i], score, heatmap);
            results[i].ok = (s || s.code == StatusCode::Skip);
            results[i].errorMsg = s.message;
            results[i].score = score;
        }
    }

    // ---- 串行收集结果 + 叠加绘制 ----
    for (size_t i = 0; i < n; i++) {
        auto& slot = slots_[i];
        if (!results[i].ok) {
            std::cerr << "multi_roi: ROI " << slot.roiId
                      << " inference failed: " << results[i].errorMsg << std::endl;
            continue;
        }
        float score = results[i].score;
        bool isAnomaly = score > slot.anomalyThreshold;
        anyAnomaly = anyAnomaly || isAnomaly;
        out.roiMap["roi_" + slot.roiId + "_score"] = score;
        out.roiMap["roi_" + slot.roiId + "_anomaly"] = isAnomaly ? 1.0f : 0.0f;
        if (drawOverlay_) {
            DrawRoiAnomaly(out.image, slot.rect, slot.roiId, score, isAnomaly);
        }
    }

    out.roiMap["multi_roi_any_anomaly"] = anyAnomaly ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));
    return Status{};
}

// -------------------------------------------------------
// 单 ROI 推理：裁剪 → 特征提取 → MemoryBank 比对
//
// 流程：
//   1. 用 cv::Rect 从大图裁剪 ROI 子图（零拷贝 Mat header）
//      → .clone() 深拷贝为独立内存
//   2. backbone->Extract() 提取该 ROI 子图的 PatchFeature
//   3. MemoryBank.ComputeAnomalyMap() 生成热力图
//   4. 取热力图最大值作为 ROI 级异常得分
// -------------------------------------------------------
Status MultiRoiNode::ProcessOneRoi(const cv::Mat& fullImage,
                                    const RoiModelSlot& slot,
                                    float& outScore,
                                    cv::Mat& outHeatmap) {
    // ---- 裁剪 ROI 子图 ----
    cv::Mat roiImg;
    try {
        roiImg = fullImage(slot.rect).clone();
    } catch (const cv::Exception& e) {
        return Status{StatusCode::ErrorPreprocess,
            "ROI " + slot.roiId + " out of bounds: " + e.what()};
    }

    // ---- backbone 提取特征 ----
    auto patchFeatures = backbone_->Extract(roiImg);
    if (patchFeatures.empty()) {
        return Status{StatusCode::ErrorModelInfer,
            "ROI " + slot.roiId + ": backbone returned no features"};
    }

    // ---- MemoryBank 生成异常热力图 ----
    auto anomalyMap = slot.memoryBank.ComputeAnomalyMap(
        patchFeatures, roiImg.rows, roiImg.cols);
    if (anomalyMap.empty()) {
        return Status{StatusCode::ErrorInternal,
            "ROI " + slot.roiId + ": anomaly map empty"};
    }

    // ---- 取最大异常得分 ----
    // 使用 max_element 而非 minMaxLoc，避免 cv::Mat 包装开销
    outScore = *std::max_element(anomalyMap.begin(), anomalyMap.end());

    // ---- 构造 ROI 尺寸的热力图（用于叠加显示） ----
    outHeatmap = cv::Mat(roiImg.rows, roiImg.cols, CV_32F, anomalyMap.data()).clone();

    return Status{};
}

} // namespace aicore
