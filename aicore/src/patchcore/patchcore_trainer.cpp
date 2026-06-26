// ============================================================
// patchcore_trainer.cpp — PatchCore 训练器实现
// 功能：编排完整训练流水线：数据集遍历 → 特征提取 → 随机降采样
//       → Coreset 核心集采样 → 构建 MemoryBank → 序列化保存
// ============================================================
#include "patchcore/patchcore_trainer.h"
#include "patchcore/backbone.h"
#include "patchcore/coreset_sampler.h"
#include "patchcore/faiss_index_bridge.h"
#include "patchcore/folder_dataset.h"
#include "patchcore/scheduler.h"
#include "json.hpp"
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace aicore {

// -------------------------------------------------------
// 通用训练接口：遍历数据集提取特征，Coreset 采样后保存 MemoryBank
//
// 训练流水线：
//   1. 根据 Scheduler 决策选择 GPU 或 CPU backbone
//   2. 创建并初始化 backbone
//   3. 遍历数据集中每张图像，提取所有 PatchFeature
//   4. 若特征数超过 maxFeatures，随机打乱后截断
//   5. 按 coresetFraction 比例用 CoresetSampler 采样
//   6. 用采样后的特征子集构建 MemoryBank
//   7. 序列化保存到 outputPath
//
// GPU OOM 降级：
//   训练循环中 catch runtime_error → 触发 RecheckGPU →
//   kBalanced/kInference 模式自动降级为 CPU backbone 重试
// -------------------------------------------------------
Status PatchCoreTrainer::Train(IDataset& dataset, const std::string& modelPath,
                                const std::string& outputPath,
                                const PatchCoreTrainConfig& cfg) {
    if (dataset.Size() == 0) {
        return Status{StatusCode::ErrorInvalidInput, "patchcore: dataset is empty"};
    }

    // 构建 backbone 的配置参数
    NodeConfig backboneConfig;
    backboneConfig["model_path"] = modelPath;
    backboneConfig["backbone_layers"] = cfg.backboneLayers;
    backboneConfig["input_size"] = std::to_string(cfg.inputSize);
    if (cfg.backboneType == "model_backend") {
        backboneConfig["backend_type"] = cfg.backendType;
    }

    // 根据 Scheduler 决策选择 GPU 或 CPU backbone 类型
    std::string effectiveType = cfg.backboneType;
    if (Scheduler::Instance().TrainingUseGPU()) {
        backboneConfig["backbone"] = "libtorch";
        effectiveType = "libtorch";
    } else {
        backboneConfig["backbone"] = "opencv_dnn";
        effectiveType = "opencv_dnn";
    }

    // 创建并初始化 backbone
    bool isBackboneGpu = (effectiveType == "libtorch");
    auto backbone = CreateBackbone(effectiveType, backboneConfig);
    if (!backbone) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone: " + effectiveType};
    }
    auto s = backbone->Init(backboneConfig);
    if (!s) return s;

    // ---- 特征提取阶段 ----
    std::vector<PatchFeature> allFeatures;
    size_t total = dataset.Size();

    for (size_t i = 0; i < total; ) {
        if (cfg.onProgress) {
            cfg.onProgress(static_cast<int>(i), static_cast<int>(total), "extracting features");
        }

        auto sample = dataset.Get(i);
        std::vector<PatchFeature> feats;
        if (isBackboneGpu) {
            try {
                feats = backbone->Extract(sample.image);
            } catch (const std::runtime_error&) {
                Scheduler::Instance().RecheckGPU();
                if (!Scheduler::Instance().TrainingUseGPU()) {
                    backboneConfig["backbone"] = "opencv_dnn";
                    backbone = CreateBackbone("opencv_dnn", backboneConfig);
                    if (backbone) backbone->Init(backboneConfig);
                    isBackboneGpu = false;
                    continue;
                }
                return Status{StatusCode::ErrorGpuDevice,
                    "OOM in kTraining mode, cannot continue"};
            }
        } else {
            feats = backbone->Extract(sample.image);
        }
        allFeatures.insert(allFeatures.end(), feats.begin(), feats.end());
        i++;
    }

    if (allFeatures.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: no features extracted"};
    }

    size_t nFeats = allFeatures.size();

    // ---- 特征上限截断 ----
    if (nFeats > cfg.maxFeatures) {
        if (cfg.onProgress) {
            cfg.onProgress(static_cast<int>(total), static_cast<int>(total), "random downsampling");
        }
        std::shuffle(allFeatures.begin(), allFeatures.end(),
                     std::mt19937(std::random_device()()));
        allFeatures.resize(cfg.maxFeatures);
        nFeats = cfg.maxFeatures;
    }

    size_t targetSize = static_cast<size_t>(nFeats * cfg.coresetFraction);
    if (targetSize == 0) targetSize = 1;

    if (cfg.onProgress) {
        cfg.onProgress(static_cast<int>(total), static_cast<int>(total), "coreset sampling");
    }
    CoresetSampler sampler;
    auto indices = sampler.Sample(allFeatures, targetSize);

    MemoryBank bank;
    std::vector<PatchFeature> coreFeatures;
    for (auto idx : indices) {
        coreFeatures.push_back(allFeatures[idx]);
    }
    bank.Build(coreFeatures);

    if (cfg.computeNormParams && nFeats > 0) {
        if (cfg.onProgress) {
            cfg.onProgress(static_cast<int>(total), static_cast<int>(total), "computing norm params");
        }
        int dim = static_cast<int>(allFeatures[0].features.size());
        bank.hasNormParams_ = true;
        bank.normMean_.assign(dim, 0.0f);
        bank.normStd_.assign(dim, 0.0f);
        for (auto& pf : allFeatures) {
            for (int d = 0; d < dim; d++) {
                bank.normMean_[d] += pf.features[d];
            }
        }
        float invN = 1.0f / nFeats;
        for (int d = 0; d < dim; d++) {
            bank.normMean_[d] *= invN;
        }
        for (auto& pf : allFeatures) {
            for (int d = 0; d < dim; d++) {
                float diff = pf.features[d] - bank.normMean_[d];
                bank.normStd_[d] += diff * diff;
            }
        }
        float invNm1 = 1.0f / (nFeats > 1 ? (nFeats - 1) : 1);
        for (int d = 0; d < dim; d++) {
            bank.normStd_[d] = std::sqrt(bank.normStd_[d] * invNm1 + 1e-8f);
        }
    }

    if (cfg.onProgress) {
        cfg.onProgress(static_cast<int>(total), static_cast<int>(total), "saving memory bank");
    }
    auto saveStatus = bank.Save(outputPath);
    if (!saveStatus) {
        lastError_ = saveStatus.message;
        return saveStatus;
    }

    // 可选：构建 FAISS 索引
    if (cfg.buildFaissIndex) {
        if (cfg.onProgress) {
            cfg.onProgress(static_cast<int>(total), static_cast<int>(total),
                           "building faiss index");
        }

        FaissIndexConfig faissCfg;
        faissCfg.algorithm = cfg.faissAlgorithm;
        faissCfg.nlist = cfg.faissNlist;
        faissCfg.nprobe = cfg.faissNprobe;
        faissCfg.M = cfg.faissM;
        faissCfg.efConstruction = cfg.faissEfConstruction;
        faissCfg.efSearch = cfg.faissEfSearch;

        FaissIndexBridge bridge;
        Status faissSt = bridge.TrainFromMemoryBank(outputPath, faissCfg);
        if (!faissSt) {
            // FAISS 构建失败非致命，不影响 MemoryBank 可用性
            lastError_ = faissSt.message;
        }
    }

    // 可选：自动计算阈值（用训练集正样本的 anomaly score 分布）
    if (cfg.computeThresholdFromTrainData) {
        if (cfg.onProgress) {
            cfg.onProgress(static_cast<int>(total), static_cast<int>(total),
                           "computing threshold from training data");
        }

        // 按采样比例随机选择图片
        std::vector<size_t> indices(total);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
        size_t sampleCount = std::max<size_t>(1,
            static_cast<size_t>(total * cfg.thresholdSampleRatio));
        if (sampleCount > total) sampleCount = total;
        indices.resize(sampleCount);

        // 遍历采样图片，收集 anomaly score
        std::vector<float> scores;
        scores.reserve(sampleCount);

        for (size_t idx : indices) {
            auto sample = dataset.Get(idx);
            auto feats = backbone->Extract(sample.image);
            if (feats.empty()) continue;

            auto anomalyMap = bank.ComputeAnomalyMap(feats,
                sample.image.rows, sample.image.cols);
            if (anomalyMap.empty()) continue;

            float maxVal = *std::max_element(anomalyMap.begin(), anomalyMap.end());
            scores.push_back(maxVal);
        }

        if (!scores.empty()) {
            float scoreMin = *std::min_element(scores.begin(), scores.end());
            float scoreMax = *std::max_element(scores.begin(), scores.end());
            double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
            double mean = sum / scores.size();
            double sqSum = 0.0;
            for (float s : scores) sqSum += (s - mean) * (s - mean);
            double stddev = std::sqrt(sqSum / scores.size());

            float threshold = 0.0f;
            switch (cfg.thresholdMethod) {
            case ThresholdMethod::MaxScore:
                threshold = scoreMax;
                break;
            case ThresholdMethod::MeanKSigma:
                threshold = static_cast<float>(mean + cfg.thresholdSigma * stddev);
                break;
            case ThresholdMethod::Percentile:
                std::sort(scores.begin(), scores.end());
                {
                    size_t pIdx = static_cast<size_t>(
                        std::ceil(cfg.thresholdPercentile / 100.0 * scores.size())) - 1;
                    if (pIdx >= scores.size()) pIdx = scores.size() - 1;
                    threshold = scores[pIdx];
                }
                break;
            }

            // 写出 .threshold.json 文件
            nlohmann::json j;
            j["anomaly_threshold"] = threshold;
            switch (cfg.thresholdMethod) {
            case ThresholdMethod::MaxScore:   j["method"] = "max_score"; break;
            case ThresholdMethod::MeanKSigma: j["method"] = "mean_ksigma"; break;
            case ThresholdMethod::Percentile: j["method"] = "percentile"; break;
            }
            j["num_samples"] = scores.size();
            j["score_stats"]["min"] = scoreMin;
            j["score_stats"]["max"] = scoreMax;
            j["score_stats"]["mean"] = mean;
            j["score_stats"]["std"] = stddev;

            if (cfg.thresholdMethod == ThresholdMethod::MeanKSigma) {
                j["threshold_params"]["sigma"] = cfg.thresholdSigma;
            } else if (cfg.thresholdMethod == ThresholdMethod::Percentile) {
                j["threshold_params"]["percentile"] = cfg.thresholdPercentile;
            }

            std::string jsonPath = outputPath + ".threshold.json";
            std::ofstream of(jsonPath);
            if (of) of << j.dump(2);
        } else {
            lastError_ = "threshold computation: no valid scores collected";
        }
    }

    return Status{};
}

// -------------------------------------------------------
// 便捷接口：从文件夹路径加载数据集后调用 Train
// -------------------------------------------------------
Status PatchCoreTrainer::TrainFromFolder(const std::string& folderPath,
                                          const std::string& modelPath,
                                          const std::string& outputPath,
                                          const PatchCoreTrainConfig& cfg) {
    FolderDataset dataset;
    auto s = dataset.Load(folderPath);
    if (!s) {
        lastError_ = s.message;
        return s;
    }
    return Train(dataset, modelPath, outputPath, cfg);
}

} // namespace aicore
