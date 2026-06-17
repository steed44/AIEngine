// 数据加载器，负责将数据集分批加载并提供随机打乱功能
// 支持迭代式访问，每次调用 Next() 获取一个批次
#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <vector>
#include <random>

namespace aicore {

// 批次数据结构
// 将一批样本的图片、标签和边界框分别聚合为数组
struct Batch {
    std::vector<cv::Mat> images;  // 批次中的图像列表
    std::vector<int> labels;      // 批次中的标签列表
    std::vector<BBox> bboxes;     // 批次中的边界框列表
};

// 数据加载器类
// 包装 IDataset，提供批次大小控制、随机打乱和迭代访问
// 典型使用：for (loader.Reset(); loader.HasNext(); ) { auto batch = loader.Next(); }
class DataLoader {
public:
    // 构造数据加载器
    // @param dataset    数据集对象
    // @param batchSize  每批样本数量（默认 16）
    // @param shuffle    是否在每次 Reset() 时打乱数据顺序（默认 true）
    DataLoader(std::shared_ptr<IDataset> dataset, int batchSize = 16, bool shuffle = true);

    // 获取下一批数据
    // @return Batch 结构，包含 images/labels/bboxes
    Batch Next();

    // 检查是否还有未读取的批次
    // @return true 表示还有数据可读
    bool HasNext() const;

    // 重置迭代器位置（重新从头开始读取）
    // 若 shuffle 为 true，会重新打乱索引
    void Reset();

    // 计算总批次数量（基于数据集大小和 batchSize）
    // @return 总批次数
    size_t NumBatches() const;

private:
    std::shared_ptr<IDataset> dataset_;  // 关联的数据集
    int batchSize_;                       // 每批样本数
    bool shuffle_;                        // 是否打乱
    size_t current_ = 0;                  // 当前读取位置（索引偏移）
    std::vector<size_t> indices_;         // 打乱后的样本索引序列
    std::mt19937 rng_;                    // 随机数生成器（用于打乱）
};

} // namespace aicore
