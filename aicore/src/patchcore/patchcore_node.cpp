// ============================================================
// patchcore_node.cpp — PatchCoreNode 的处理节点实现
// 功能：实现 PatchCore 推理节点的初始化和 Process 逻辑
// ============================================================
#include "patchcore/patchcore_node.h"
#include <opencv2/imgproc.hpp>

namespace aicore {

// -------------------------------------------------------
// 初始化 PatchCore 推理节点
// 从 NodeConfig 中解析以下关键配置：
//   - name: 节点名称（可选，默认 "patchcore"）
//   - backbone: backbone 类型（可选，默认 "opencv_dnn"）
//   - memory_bank_path: 预训练记忆库文件路径（可选）
//   - input_size: 输入图像缩放尺寸（可选，默认 224）
//   - anomaly_threshold: 异常判定阈值（可选，默认 0.5）
// -------------------------------------------------------
Status PatchCoreNode::Init(const NodeConfig& config) {
    name_ = config.count("name") ? config.at("name") : "patchcore";

    auto bt = config.find("backbone");
    std::string backboneType = (bt != config.end()) ? bt->second : "opencv_dnn";

    // 通过工厂方法创建 backbone 实例
    backbone_ = CreateBackbone(backboneType, config);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone type: " + backboneType};
    }
    auto s = backbone_->Init(config);
    if (!s) return s;

    // 加载预训练的记忆库文件（若配置中指定）
    auto mn = config.find("memory_bank_path");
    if (mn != config.end()) {
        if (!memoryBank_.Load(mn->second)) {
            return Status{StatusCode::ErrorModelLoad, "patchcore: cannot load memory bank"};
        }
    }

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    auto at = config.find("anomaly_threshold");
    if (at != config.end()) anomalyThreshold_ = std::stof(at->second);

    return Status{};
}

// -------------------------------------------------------
// 处理一帧图像：提取特征 → 与记忆库比对 → 生成异常热力图和评分
// 输出帧包含：
//   - anomaly_score: 图像级别的最大异常得分
//   - is_anomaly: 是否异常（二值判定，基于 anomalyThreshold_）
// -------------------------------------------------------
Status PatchCoreNode::Process(const std::vector<Frame>& inputs,
                               std::vector<Frame>& outputs) {
    if (inputs.empty()) {
        return Status{StatusCode::ErrorInvalidInput, "patchcore: no input"};
    }

    cv::Mat img = inputs[0].image;
    if (img.empty()) {
        return Status{StatusCode::ErrorPreprocess, "patchcore: empty image"};
    }

    // Step 1: backbone 提取局部 Patch 特征
    // backbone（如 WideResNet50）将输入图像映射到多层次特征图。
    // 每一层特征图的一个空间位置对应输入图像的一个"感受野"区域（Patch）。
    // 例如 layer3 输出 28×28×512 的特征图，则每个 Patch 对应约 16×16 像素的输入区域。
    // 这些 Patch 特征编码了该区域的纹理、边缘、形状等视觉信息。
    auto patchFeatures = backbone_->Extract(img);
    if (patchFeatures.empty()) {
        return Status{StatusCode::ErrorModelInfer, "patchcore: backbone returned no features"};
    }

    // Step 2: 逐 Patch 与 MemoryBank 最近邻比对，生成异常热力图
    // 对每个 Patch 特征向量 f_p，在记忆库中找最近邻 f_n，
    // 异常得分 = ||f_p - f_n||₂（L2 距离）
    // 得分越高 = 该 Patch 越"异常"（偏离正常特征分布）
    auto anomalyMap = memoryBank_.ComputeAnomalyMap(patchFeatures, img.rows, img.cols);
    if (anomalyMap.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: anomaly map empty"};
    }

    // Step 3: 将浮点热力图包装为 cv::Mat 并取最大值作为帧级异常得分
    cv::Mat scoreMap(img.rows, img.cols, CV_32F, anomalyMap.data());

    double maxVal = 0;
    cv::minMaxLoc(scoreMap, nullptr, &maxVal);

    // Step 4: 异常判定
    // is_anomaly = max(anomaly_map) > threshold
    // 使用最大得分而非平均分的原因是：
    //   如果图像中只有一小块区域异常（如微小划痕），平均分会淹没该信号，
    //   而最大分能捕捉到局部异常。这也是 PatchCore 的论文设计。
    // 构造输出帧，携带异常得分和判定结果
    Frame out(scoreMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));

    return Status{};
}

} // namespace aicore
