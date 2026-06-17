// ============================================================
// multi_roi_node.cpp — 多 ROI PatchCore 推理节点实现
// 功能：加载配置 → 加载各 ROI 模型 → 对输入大图执行
//       多区域推理 → 在原图上叠加检测结果
// ============================================================
#include "patchcore/multi_roi_node.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

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

    // ---- 加载各 ROI 的 MemoryBank ----
    for (const auto& roi : config_.rois) {
        RoiModelSlot slot;
        slot.roiId = roi.id;
        slot.rect = cv::Rect(roi.x, roi.y, roi.w, roi.h);
        slot.anomalyThreshold = config_.anomalyThreshold;

        std::string bankPath = config_.modelDir + "/" + roi.id + ".bin";
        if (!slot.memoryBank.Load(bankPath)) {
            return Status{StatusCode::ErrorModelLoad,
                "multi_roi: cannot load memory bank: " + bankPath};
        }

        slots_.push_back(slot);
    }

    return Status{};
}

// -------------------------------------------------------
// 推理一帧大图：遍历所有 ROI，执行裁剪→提取→比对→叠加
//
// 输出帧包含：
//   - roiMap["roi_{id}_score"] = float 异常得分
//   - roiMap["roi_{id}_anomaly"] = 1.0f（异常）或 0.0f（正常）
//   - image 上绘制了 ROI 框（绿色=正常 红色=异常）
//
// 全 ROI 推理的总时间 = Σ(每 ROI 裁剪 + 特征提取 + NN 搜索)
// 共享 backbone 避免了重复加载模型，但每 ROI 仍需独立 Extract
// -------------------------------------------------------
Status MultiRoiNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    if (inputs.empty()) {
        return Status{StatusCode::ErrorInvalidInput, "multi_roi: no input"};
    }

    cv::Mat fullImage = inputs[0].image.clone();
    if (fullImage.empty()) {
        return Status{StatusCode::ErrorPreprocess, "multi_roi: empty image"};
    }

    // 储存推理结果
    Frame out(fullImage.clone());
    bool anyAnomaly = false;

    // ---- 遍历每个 ROI 执行推理 ----
    for (size_t i = 0; i < slots_.size(); i++) {
        const auto& slot = slots_[i];
        float score = 0;
        cv::Mat heatmap;

        auto s = ProcessOneRoi(fullImage, slot, score, heatmap);
        if (!s) {
            std::cerr << "multi_roi: ROI " << slot.roiId
                      << " inference failed: " << s.message << std::endl;
            continue;
        }

        bool isAnomaly = score > slot.anomalyThreshold;
        anyAnomaly = anyAnomaly || isAnomaly;

        // 将结果写入输出帧的 roiMap
        out.roiMap["roi_" + slot.roiId + "_score"] = score;
        out.roiMap["roi_" + slot.roiId + "_anomaly"] = isAnomaly ? 1.0f : 0.0f;

        // 在输出图像上绘制 ROI 框和热力图
        if (drawOverlay_) {
            DrawRoiOverlay(out.image, slot, score);
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

// -------------------------------------------------------
// 在大图上绘制 ROI 检测结果
//
// 绘制内容：
//   1. ROI 矩形框：阀值线框，绿色（正常）或 红色（异常）
//   2. ROI ID + 异常得分标签
//   3. 半透明热力图叠加（更直观显示异常区域位置）
// -------------------------------------------------------
void MultiRoiNode::DrawRoiOverlay(cv::Mat& image,
                                   const RoiModelSlot& slot,
                                   float score) {
    bool isAnomaly = score > slot.anomalyThreshold;

    // ---- 1. 选择框颜色 ----
    // 正常 = 绿色 (0, 255, 0)，异常 = 红色 (0, 0, 255)
    cv::Scalar color = isAnomaly
        ? cv::Scalar(0, 0, 255)
        : cv::Scalar(0, 255, 0);

    // ---- 2. 绘制 ROI 矩形框 ----
    cv::rectangle(image, slot.rect, color, 2);

    // ---- 3. 绘制标签 ----
    std::string label = slot.roiId + ": " +
        (isAnomaly ? "NG " : "OK ") +
        std::to_string(static_cast<int>(score * 100)) + "%";

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseline);
    cv::Point labelPos(slot.rect.x, slot.rect.y - 5);
    if (labelPos.y < textSize.height) {
        labelPos.y = slot.rect.y + slot.rect.height + textSize.height + 5;
    }

    // 标签背景
    cv::rectangle(image,
                  cv::Point(labelPos.x, labelPos.y - textSize.height),
                  cv::Point(labelPos.x + textSize.width, labelPos.y + baseline),
                  color, cv::FILLED);
    // 标签文字（黑色）
    cv::putText(image, label, labelPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

} // namespace aicore
