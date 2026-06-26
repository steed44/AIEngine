// ============================================================
// patchcore_trainer.h — PatchCore 模型训练器
// 功能：管理 PatchCore 的完整训练流程，从正常样本中提取
//       特征、Coreset 采样、构建 MemoryBank 并保存到磁盘
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include "patchcore/faiss_index.h"
#include "trainer/data/dataset.h"
#include <string>
#include <functional>

namespace aicore {

using ProgressCallback = std::function<void(int current, int total, const std::string& status)>;

// -------------------------------------------------------
// ThresholdMethod — 自动阈值计算方法
// -------------------------------------------------------
enum class ThresholdMethod {
    MaxScore,     // 使用训练集 anomaly score 最大值
    MeanKSigma,   // 使用均值 + k * 标准差
    Percentile    // 使用指定分位数
};

// -------------------------------------------------------
// PatchCoreTrainConfig — 训练配置参数
// -------------------------------------------------------
struct AICORE_API PatchCoreTrainConfig {
    int inputSize = 224;                      // 输入图像缩放尺寸
    std::string backboneLayers = "layer2,layer3";  // backbone 待提取的中间层名（逗号分隔）
    std::string backboneType = "opencv_dnn";  // backbone 类型：opencv_dnn / model_backend / libtorch
    std::string backendType = "onnxruntime";  // model_backend 时指定具体后端：onnxruntime / tensorrt / libtorch
    double coresetFraction = 0.1;             // Coreset 采样比例（默认保留 10%）
    size_t maxFeatures = 100000;              // 最大特征数上限（超出时随机降采样）
    bool computeNormParams = true;            // 是否计算并保存特征归一化参数
    ProgressCallback onProgress = nullptr;    // 可选: 训练进度回调

    // 自动阈值计算（训练完后用训练集正样本的 anomaly score 分布生成推荐阈值）
    bool computeThresholdFromTrainData = false;            // 是否自动计算阈值
    ThresholdMethod thresholdMethod = ThresholdMethod::MeanKSigma;  // 计算方法
    float thresholdSigma = 3.0f;                           // MeanKSigma 法的 k 值
    float thresholdPercentile = 99.0f;                     // Percentile 法的 p 值
    float thresholdSampleRatio = 1.0f;                     // 训练集采样比例 (0~1)

    // FAISS 索引构建（可选）
    bool buildFaissIndex = false;                         // 训练结束时是否构建 FAISS 索引
    FaissSearchAlgorithm faissAlgorithm = FaissSearchAlgorithm::HNSW;
    int faissNlist = 100;
    int faissNprobe = 16;
    int faissM = 16;
    int faissEfConstruction = 200;
    int faissEfSearch = 64;
};

// -------------------------------------------------------
// PatchCoreTrainer — PatchCore 训练器
// 职责：编排完整训练流程：读取数据集 → 遍历每张图提取 Patch 特征
//       → 随机降采样（若超限）→ Coreset 核心集采样 → 构建
//       MemoryBank → 序列化保存到磁盘
// 典型使用场景：用户提供正常样本数据集和 backbone 模型文件，
//       训练器输出 MemoryBank .bin 文件，供 PatchCoreNode 推理使用
// -------------------------------------------------------
class AICORE_API PatchCoreTrainer {
public:
    // 通用训练接口：直接传入已加载的 IDataset
    // @param dataset    训练数据集（正常样本）
    // @param modelPath  backbone 模型文件路径
    // @param outputPath MemoryBank 输出路径（.bin 文件）
    // @param cfg        训练配置参数
    // 前置条件：dataset 非空，modelPath 有效，outputPath 所在目录可写
    // 后置条件：成功时 outputPath 文件包含序列化的 MemoryBank
    // 训练流程：遍历 dataset → Extract 每张图 → 收集 PatchFeature →
    //   随机降采样（超限时）→ Coreset 采样 → Build MemoryBank → Save
    // 进度回调：cfg.onProgress(current, total, status) 每秒触发一次
    Status Train(IDataset& dataset, const std::string& modelPath,
                 const std::string& outputPath,
                 const PatchCoreTrainConfig& cfg);
    // 便捷接口：从文件夹路径直接训练
    // @param folderPath 正常样本图像文件夹路径
    // @param modelPath  backbone 模型文件路径
    // @param outputPath MemoryBank 输出路径（.bin 文件）
    // @param cfg        训练配置参数
    // 内部创建 FolderDataset 后调用 Train()
    Status TrainFromFolder(const std::string& folderPath,
                           const std::string& modelPath,
                           const std::string& outputPath,
                           const PatchCoreTrainConfig& cfg);
    // 获取最后出错的详细信息
    std::string GetLastError() const { return lastError_; }

private:
    std::string lastError_;  // 最后一次训练的出错信息
};

} // namespace aicore
