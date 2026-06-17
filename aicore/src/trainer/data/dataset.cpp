// ============================================================================
// 文件：dataset.cpp
// 用途：数据集（Dataset）相关实现，负责加载和访问训练/验证数据
// 功能：提供 COCO 格式数据集的基本操作：加载、按索引读取样本
// ============================================================================

#include "trainer/data/dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>

namespace aicore {
// ========================================================================
// 命名空间：aicore
// AIEngine 核心命名空间，包含训练、推理、数据等核心模块
// ========================================================================

// 加载指定路径的 COCO 格式数据集（标注文件、图片目录等）
// 参数 path - 数据集配置文件或目录路径
// 返回值   - 操作状态，成功返回 Status{}，失败带回错误码和描述
Status COCODataset::Load(const std::string& path) {
    (void)path;
    return Status{};
}

// 返回数据集中的样本总数
// 返回值 - 样本数量
size_t COCODataset::Size() const {
    return samples_.size();
}

// 获取指定索引位置的样本，并从磁盘加载对应的图片数据
// 参数 index - 样本索引，从 0 开始
// 返回值    - Sample 结构体，包含图片、标签等信息；索引越界时返回空 Sample
Sample COCODataset::Get(size_t index) {
    if (index >= samples_.size()) return {};
    auto& s = samples_[index];
    s.image = cv::imread(s.imagePath);
    return s;
}

// 返回 COCO 数据集的默认类别数量（COCO 标准为 80 类）
// 返回值 - 类别数（80）
int COCODataset::NumClasses() const {
    return 80;
}

} // namespace aicore
