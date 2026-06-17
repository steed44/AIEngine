// ============================================================
// patchcore_trainer.cpp — PatchCore 训练器实现
// 功能：编排完整训练流水线：数据集遍历 → 特征提取 → 随机降采样
//       → Coreset 核心集采样 → 构建 MemoryBank → 序列化保存
// ============================================================
#include "patchcore/patchcore_trainer.h"
#include "patchcore/backbone.h"
#include "patchcore/coreset_sampler.h"
#include "patchcore/folder_dataset.h"
#include <random>

namespace aicore {

// -------------------------------------------------------
// 通用训练接口：遍历数据集提取特征，Coreset 采样后保存 MemoryBank
//
// 训练流水线：
//   1. 创建并初始化 backbone（根据 cfg.backboneType）
//   2. 遍历数据集中每张图像，提取所有 PatchFeature
//   3. 若特征数超过 maxFeatures，随机打乱后截断
//   4. 按 coresetFraction 比例用 CoresetSampler 采样
//   5. 用采样后的特征子集构建 MemoryBank
//   6. 序列化保存到 outputPath
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

    // 创建并初始化 backbone
    auto backbone = CreateBackbone(cfg.backboneType, backboneConfig);
    if (!backbone) {
        return Status{StatusCode::ErrorConfigParse,
            "patchcore: unknown backbone: " + cfg.backboneType};
    }
    auto s = backbone->Init(backboneConfig);
    if (!s) return s;

    // ---- 特征提取阶段 ----
    // 遍历数据集中每张正常图像，用 backbone 提取所有 Patch 特征。
    // 所有特征被拼接到 allFeatures 向量中，
    // 特征总量 = 图像数 × 每图 Patch 数（如 100 张图 × 28×28 = 78400 个特征）
    std::vector<PatchFeature> allFeatures;

    for (size_t i = 0; i < dataset.Size(); i++) {
        auto sample = dataset.Get(i);
        auto feats = backbone->Extract(sample.image);
        allFeatures.insert(allFeatures.end(), feats.begin(), feats.end());
    }

    if (allFeatures.empty()) {
        return Status{StatusCode::ErrorInternal, "patchcore: no features extracted"};
    }

    // ---- 特征上限截断 ----
    // maxFeatures 防止记忆库过大导致：
    // 1. 磁盘文件过大（序列化慢）
    // 2. 推理时暴力 NN 搜索变慢（O(N·D)）
    // 3. Coreset 采样时间开销 O(n·k)
    // 随机打乱后截断比简单截断更公平——保证不同图像的 Patch 有均等机会被保留
    if (allFeatures.size() > cfg.maxFeatures) {
        std::shuffle(allFeatures.begin(), allFeatures.end(),
                     std::mt19937(std::random_device()()));
        allFeatures.resize(cfg.maxFeatures);
    }

    // 计算 Coreset 目标采样数量，至少保留一个
    size_t targetSize = static_cast<size_t>(allFeatures.size() * cfg.coresetFraction);
    if (targetSize == 0) targetSize = 1;

    // ---- Coreset 核心集采样 ----
    // 用最远点采样从 allFeatures 中选取最具多样性的子集。
    // 例如 100000 个特征以 10% 比例采样到 10000 个。
    // 相比随机采样，Coreset 能更好的保持特征空间的覆盖范围，
    // 减少因采样导致的"漏检"风险。
    // 这是 PatchCore 论文的核心贡献之一。
    CoresetSampler sampler;
    auto indices = sampler.Sample(allFeatures, targetSize);

    // 用采样结果构建 MemoryBank 并保存
    MemoryBank bank;
    std::vector<PatchFeature> coreFeatures;
    for (auto idx : indices) {
        coreFeatures.push_back(allFeatures[idx]);
    }
    bank.Build(coreFeatures);

    if (!bank.Save(outputPath)) {
        lastError_ = "failed to save memory bank to " + outputPath;
        return Status{StatusCode::ErrorInternal, lastError_};
    }

    return Status{};
}

// -------------------------------------------------------
// 便捷接口：从文件夹路径加载数据集后调用 Train
// 自动创建 FolderDataset 并加载指定文件夹中的图像
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
