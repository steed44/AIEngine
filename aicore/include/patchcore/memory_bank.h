// ============================================================
// memory_bank.h — PatchCore 的记忆库（Memory Bank）数据结构
// 功能：存储正常样本的 Patch 级特征，提供最近邻检索和异常热力图生成
//       支持序列化保存/加载，用于训练→推理的模型传递
// ============================================================
#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace aicore {

// -------------------------------------------------------
// PatchFeature — 单个图像块的局部特征表示
// 职责：保存 backbone 在某层(layer)某位置(row,col)处的
//       特征向量，用于后续最近邻比对
// -------------------------------------------------------
struct PatchFeature {
    std::vector<float> features;    // 特征向量（浮点数组，维度由 backbone 输出层决定）
    int layerIdx = 0;               // 来自 backbone 的哪一层输出
    int patchRow = 0, patchCol = 0; // 该 Patch 在特征图上的行/列坐标
};

// -------------------------------------------------------
// MemoryBank — 正常样本特征记忆库
// 职责：训练阶段收集正常图像的所有 PatchFeature，通过 Coreset
//       采样缩减后存储；推理阶段对输入 PatchFeature 做逐块
//       最近邻检索，输出像素级异常热力图
// 典型使用场景：patchcore 训练完成后保存 .bin 文件，推理节点
//       加载后对每帧图像计算异常得分
// -------------------------------------------------------
class MemoryBank {
public:
    MemoryBank() = default;

    // 从二进制文件加载记忆库（含魔数校验）
    // 支持新旧两种格式:
    //   新格式: magic(4B) + version(4B) + num(4B) + dim(4B) + features
    //   旧格式: magic(4B) + num(4B) + dim(4B) + features
    // @param path .bin 文件路径
    // @return 加载成功返回 Status
    // 前置条件：path 非空，文件可读
    // 后置条件：成功时 bank_ 填充特征数据，featureDim_ 记录维度
    Status Load(const std::string& path);
    // 将记忆库序列化保存到二进制文件（新格式, version=1）
    // @param path 输出文件路径
    // @return 保存成功返回 Status
    // 前置条件：Build() 已调用或 Load() 成功
    // 后置条件：文件写入成功，格式为魔数 + version + num + dim + features
    Status Save(const std::string& path) const;

    // 构建记忆库：从特征列表构建，记录特征维度
    // @param features 特征向量列表（所有特征维度必须一致）
    // 前置条件：features 非空，所有 PatchFeature.features.size() 相同
    // 后置条件：bank_ 被替换为 features 的副本，featureDim_ 已设置
    void Build(const std::vector<PatchFeature>& features);
    // 清空记忆库
    // 后置条件：bank_ 为空，featureDim_ = 0
    void Clear();

    // 查找与 query 特征最接近的库内特征（暴力 L2 距离）
    // @param query    查询特征向量（维度需等于 featureDim_）
    // @param distOut  [out] 最近邻的 L2 距离
    // @return 最近邻元素在 bank_ 中的索引
    // 前置条件：bank_ 非空，query.size() == featureDim_
    // 复杂度：O(bank_.size() * featureDim_)
    size_t NearestNeighbor(const std::vector<float>& query, float& distOut) const;
    // 计算整张图像的异常热力图
    // 对每个查询 PatchFeature，在 MemoryBank 中找最近邻，
    // 将 L2 距离映射到对应像素位置，最后上采样到原图尺寸
    // @param queries 查询图像的 PatchFeature 列表
    // @param imgH    原始图像高度（用于上采样到原图尺寸）
    // @param imgW    原始图像宽度
    // @return 上采样到原始尺寸的浮点热力图（逐像素异常得分，范围 [0, +∞)）
    // 前置条件：bank_ 非空，queries 非空
    // 后置条件：返回向量长度为 imgH * imgW
    std::vector<float> ComputeAnomalyMap(const std::vector<PatchFeature>& queries,
                                          int imgH, int imgW) const;

    size_t Size() const { return bank_.size(); }
    int FeatureDim() const { return featureDim_; }

    static constexpr uint32_t kMagic = 0x50434F52;  // 旧格式魔数 "PCOR"
    static constexpr uint32_t kMagicNew = 0x50434F56; // 新格式魔数 "PCORv"
    static constexpr uint32_t kCurrentVersion = 1;    // 当前版本号

    // 归一化参数: 训练时特征各维度的 mean/std
    // 推理时用相同参数归一化查询特征后再做 NN 搜索
    bool hasNormParams_ = false;
    std::vector<float> normMean_;
    std::vector<float> normStd_;

private:
    std::vector<PatchFeature> bank_;
    int featureDim_ = 0;
};

} // namespace aicore
