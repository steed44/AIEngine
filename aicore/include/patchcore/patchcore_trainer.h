// ============================================================
// patchcore_trainer.h — PatchCore 模型训练器
// 功能：管理 PatchCore 的完整训练流程，从正常样本中提取
//       特征、Coreset 采样、构建 MemoryBank 并保存到磁盘
// ============================================================
#pragma once
#include "core/types.h"
#include "patchcore/memory_bank.h"
#include "trainer/data/dataset.h"
#include <string>
#include <functional>

namespace aicore {

using ProgressCallback = std::function<void(int current, int total, const std::string& status)>;

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
