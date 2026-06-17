// 数据增强接口和增强流水线
// 支持链式组合多种数据增强操作，提升模型泛化能力
#pragma once
#include "core/types.h"
#include "trainer/data/dataset.h"
#include <memory>
#include <vector>

namespace aicore {

// 数据增强抽象接口
// 实现此接口以创建自定义增强操作（随机裁剪、翻转、颜色抖动等）
class IAugmentation {
public:
    virtual ~IAugmentation() = default;

    // 对单条样本应用增强操作
    // @param input  原始样本
    // @return 增强后的样本（可能修改 image、bbox 等字段）
    virtual Sample Apply(const Sample& input) = 0;
};

// 增强流水线类
// 将多个 IAugmentation 按添加顺序串联执行，形成完整的预处理流程
class AugmentationPipeline {
public:
    // 添加一个增强操作到流水线末尾
    // @param aug  增强操作实例
    void Add(std::shared_ptr<IAugmentation> aug);

    // 按顺序执行所有已注册的增强操作
    // @param input  原始样本
    // @return 经过全部增强后的样本
    Sample Apply(const Sample& input);

    // 清空所有已注册的增强操作
    void Clear();

private:
    std::vector<std::shared_ptr<IAugmentation>> augs_;  // 增强操作列表
};

} // namespace aicore
