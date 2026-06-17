// ============================================================================
// 文件：augmentation.cpp
// 用途：数据增强流水线实现，通过链式调用多个增强操作对样本进行变换
// 功能：批量添加增强操作、依次执行增强、清空增强链
// ============================================================================

#include "trainer/data/augmentation.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 向增强流水线末尾添加一个增强操作
// 参数 aug - 增强操作的智能指针，支持随机翻转、颜色抖动、缩放裁剪等
void AugmentationPipeline::Add(std::shared_ptr<IAugmentation> aug) {
    augs_.push_back(std::move(aug));
}

// 对输入样本执行完整的增强流水线：按添加顺序依次调用每个增强操作
// 参数 input - 原始输入样本
// 返回值    - 经过所有增强操作处理后的样本
Sample AugmentationPipeline::Apply(const Sample& input) {
    Sample out = input;
    for (auto& aug : augs_)
        out = aug->Apply(out);
    return out;
}

// 清空增强流水线中的所有操作
void AugmentationPipeline::Clear() { augs_.clear(); }

} // namespace aicore
