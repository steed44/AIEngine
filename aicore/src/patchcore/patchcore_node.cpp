// ============================================================
// patchcore_node.cpp — PatchCoreNode 的处理节点实现
// 功能：实现 PatchCore 推理节点的初始化和 Process 逻辑
// ============================================================
#include "patchcore/patchcore_node.h"
#include "patchcore/scheduler.h"
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
//
// 调度器支持：
//   除主 backbone 外，额外创建 GPU backbone（LibTorch）和
//   CPU backbone（OpenCV DNN），运行时根据 Scheduler 策略
//   自动选择 GPU/CPU 后端，GPU OOM 时自动降级到 CPU。
// -------------------------------------------------------
Status PatchCoreNode::Init(const NodeConfig& config) {
    name_ = config.count("name") ? config.at("name") : "patchcore";

    auto bt = config.find("backbone");
    std::string backboneType = (bt != config.end()) ? bt->second : "opencv_dnn";

    // 创建主 backbone
    backbone_ = CreateBackbone(backboneType, config);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone type: " + backboneType};
    }
    auto s = backbone_->Init(config);
    if (!s) return s;

    // 额外创建 CPU backbone（OpenCV DNN）和 GPU backbone（LibTorch）
    // 供 Scheduler 动态切换使用
    auto cpuCfg = config;
    cpuCfg["backbone"] = "opencv_dnn";
    cpuBackbone_ = CreateBackbone("opencv_dnn", cpuCfg);
    if (cpuBackbone_) cpuBackbone_->Init(cpuCfg);

    auto gpuCfg = config;
    gpuCfg["backbone"] = "libtorch";
    gpuBackbone_ = CreateBackbone("libtorch", gpuCfg);
    if (gpuBackbone_) gpuBackbone_->Init(gpuCfg);

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
    // 根据 Scheduler 决策选择 GPU 或 CPU backbone
    // GPU OOM 时自动降级到 CPU
    std::vector<PatchFeature> patchFeatures;
    bool useGpu = Scheduler::Instance().InferenceUseGPU() && gpuBackbone_;

    if (useGpu) {
        try {
            patchFeatures = gpuBackbone_->Extract(img);
        } catch (const std::runtime_error&) {
            // GPU OOM → 自动降级到 CPU
            if (cpuBackbone_) {
                patchFeatures = cpuBackbone_->Extract(img);
            }
        }
    } else if (cpuBackbone_) {
        patchFeatures = cpuBackbone_->Extract(img);
    } else {
        patchFeatures = backbone_->Extract(img);
    }

    if (patchFeatures.empty()) {
        return Status{StatusCode::ErrorModelInfer, "patchcore: backbone returned no features"};
    }

    // Step 2: 逐 Patch 与 MemoryBank 最近邻比对，生成异常热力图
    auto anomalyMap = memoryBank_.ComputeAnomalyMap(patchFeatures, img.rows, img.cols);
    if (anomalyMap.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: anomaly map empty"};
    }

    // Step 3: 将浮点热力图包装为 cv::Mat 并取最大值作为帧级异常得分
    cv::Mat scoreMap(img.rows, img.cols, CV_32F, anomalyMap.data());

    double maxVal = 0;
    cv::minMaxLoc(scoreMap, nullptr, &maxVal);

    // Step 4: 异常判定，构造输出帧
    Frame out(scoreMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));

    return Status{};
}

} // namespace aicore
