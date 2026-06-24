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

    // ---- 校验必填配置 ----
    if (config.find("model_path") == config.end()) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: model_path is required"};
    }
    if (config.find("memory_bank_path") == config.end()) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: memory_bank_path is required"};
    }

    auto bt = config.find("backbone");
    std::string backboneType = (bt != config.end()) ? bt->second : "opencv_dnn";

    backbone_ = CreateBackbone(backboneType, config);
    if (!backbone_) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone type: " + backboneType};
    }
    auto s = backbone_->Init(config);
    if (!s) return s;

    // 额外创建 CPU backbone 和 GPU backbone，供 Scheduler 动态切换
    auto cpuCfg = config;
    cpuCfg["backbone"] = "opencv_dnn";
    cpuBackbone_ = CreateBackbone("opencv_dnn", cpuCfg);
    if (cpuBackbone_) cpuBackbone_->Init(cpuCfg);

    auto gpuCfg = config;
    gpuCfg["backbone"] = "libtorch";
    gpuBackbone_ = CreateBackbone("libtorch", gpuCfg);
    if (gpuBackbone_) gpuBackbone_->Init(gpuCfg);

    // 加载记忆库
    auto mn = config.find("memory_bank_path");
    auto loadStatus = memoryBank_.Load(mn->second);
    if (!loadStatus) {
        return Status{StatusCode::ErrorModelLoad,
            "patchcore: cannot load memory bank: " + loadStatus.message};
    }
    memoryBank_.PromoteToGPU();

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    auto at = config.find("anomaly_threshold");
    if (at != config.end()) anomalyThreshold_ = std::stof(at->second);

    auto ts = config.find("max_tile_size");
    if (ts != config.end()) maxTileSize_ = std::stoi(ts->second);

    return Status{};
}

// -------------------------------------------------------
// 处理一帧图像：提取特征 → 与记忆库比对 → 生成异常热力图和评分
// 支持大图自动分片处理: 当图像宽或高超过 maxTileSize_ 时,
//   将图像拆分为非重叠分片, 每片独立推理, 结果融合后输出
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

    auto pickBackbone = [&]() -> IBackbone* {
        if (Scheduler::Instance().InferenceUseGPU() && gpuBackbone_) {
            try {
                gpuBackbone_->Extract(img);
                return gpuBackbone_.get();
            } catch (const std::runtime_error&) {
            }
        }
        if (cpuBackbone_) return cpuBackbone_.get();
        return backbone_.get();
    };

    cv::Mat fullMap;
    bool needsTiling = (maxTileSize_ > 0) &&
        (img.cols > maxTileSize_ || img.rows > maxTileSize_);

    if (!needsTiling) {
        // ── 小图: 整图一次性推理 ──
        auto backbone = pickBackbone();
        auto features = backbone->Extract(img);
        if (features.empty()) {
            return Status{StatusCode::ErrorModelInfer,
                "patchcore: backbone returned no features"};
        }
        auto anomalyData = memoryBank_.ComputeAnomalyMap(features, img.rows, img.cols);
        if (anomalyData.empty()) {
            return Status{StatusCode::ErrorInternal,
                "patchcore: anomaly map empty"};
        }
        fullMap = cv::Mat(img.rows, img.cols, CV_32F, anomalyData.data()).clone();
    } else {
        // ── 大图: 分片处理, max-fusion 拼接 ──
        fullMap = cv::Mat::zeros(img.rows, img.cols, CV_32F);

        int tileH = std::min(maxTileSize_, img.rows);
        int tileW = std::min(maxTileSize_, img.cols);
        int overlap = 32; // 融合重叠宽度

        for (int y0 = 0; y0 < img.rows; y0 += tileH) {
            for (int x0 = 0; x0 < img.cols; x0 += tileW) {
                int y1 = std::min(y0 + tileH + overlap, img.rows);
                int x1 = std::min(x0 + tileW + overlap, img.cols);
                int cropY0 = std::max(0, y0 - overlap);
                int cropX0 = std::max(0, x0 - overlap);
                // 限制尺寸不超过 tile
                y1 = std::min(y1, cropY0 + tileH + overlap);
                x1 = std::min(x1, cropX0 + tileW + overlap);

                cv::Rect roi(cropX0, cropY0, x1 - cropX0, y1 - cropY0);
                cv::Mat tile = img(roi);

                auto backbone = pickBackbone();
                auto features = backbone->Extract(tile);
                if (features.empty()) continue;

                auto tileAnomaly = memoryBank_.ComputeAnomalyMap(
                    features, tile.rows, tile.cols);
                if (tileAnomaly.empty()) continue;

                cv::Mat tileMap(tile.rows, tile.cols, CV_32F, tileAnomaly.data());
                // max-fusion 拼入 fullMap
                cv::Mat dst = fullMap(roi);
                cv::max(dst, tileMap, dst);
            }
        }
    }

    double maxVal = 0;
    cv::minMaxLoc(fullMap, nullptr, &maxVal);

    Frame out(fullMap.clone());
    out.roiMap["anomaly_score"] = static_cast<float>(maxVal);
    out.roiMap["is_anomaly"] = maxVal > anomalyThreshold_ ? 1.0f : 0.0f;
    outputs.push_back(std::move(out));

    return Status{};
}

} // namespace aicore
