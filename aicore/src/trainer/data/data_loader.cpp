// ============================================================================
// 文件：data_loader.cpp
// 用途：数据加载器实现，负责按批次（Batch）从数据集读取样本
// 功能：支持批次读取、随机打乱、索引重置
// ============================================================================

#include "trainer/data/data_loader.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 构造函数：初始化数据加载器
// 参数 dataset   - 数据集接口指针
// 参数 batchSize - 每个批次包含的样本数
// 参数 shuffle   - 是否在每个 epoch 开始时打乱样本顺序
DataLoader::DataLoader(std::shared_ptr<IDataset> dataset, int batchSize, bool shuffle)
    : dataset_(dataset), batchSize_(batchSize), shuffle_(shuffle), rng_(std::random_device{}()) {
    // 构造时立即初始化索引并打乱
    Reset();
}

// 重置加载器状态：重新生成索引序列，必要时打乱顺序
// 每次开始新 epoch 前调用，确保数据遍历的随机性
void DataLoader::Reset() {
    current_ = 0;
    indices_.resize(dataset_->Size());
    for (size_t i = 0; i < indices_.size(); ++i) indices_[i] = i;
    if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
}

// 取出下一个批次的数据
// 遍历当前批次索引，从数据集中逐个读取样本并组装成 Batch
// 返回值 - Batch 结构体，包含 images、labels 和 bboxes 等字段
Batch DataLoader::Next() {
    Batch batch;
    for (int i = 0; i < batchSize_ && HasNext(); ++i, ++current_) {
        auto sample = dataset_->Get(indices_[current_]);
        batch.images.push_back(sample.image);
        batch.labels.push_back(sample.label);
        batch.bboxes.push_back(sample.bbox);
    }
    return batch;
}

// 判断是否还有未读取的样本
bool DataLoader::HasNext() const { return current_ < indices_.size(); }

// 计算完整遍历一次数据集所需的批次数（向上取整）
size_t DataLoader::NumBatches() const { return (indices_.size() + batchSize_ - 1) / batchSize_; }

} // namespace aicore
