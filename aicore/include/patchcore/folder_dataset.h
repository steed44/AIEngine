// ============================================================
// folder_dataset.h — 文件夹图像数据集
// 功能：将指定文件夹中的所有图像文件加载为 IDataset，
//       支持常见图像格式（jpg/png/bmp/tiff），用于 PatchCore 训练
// ============================================================
#pragma once
#include "trainer/data/dataset.h"
#include <vector>
#include <string>

namespace aicore {

// -------------------------------------------------------
// FolderDataset — 文件夹数据集实现
// 职责：遍历文件夹，自动识别图像文件并读取为 cv::Mat，
//       以统一接口提供给 PatchCoreTrainer 使用
// 典型使用场景：训练 PatchCore 时，将正常样本图片放入文件夹，
//       通过该数据集类批量加载并提取特征
// -------------------------------------------------------
class FolderDataset : public IDataset {
public:
    // 加载指定文件夹中的所有图像
    // @param folderPath 图像文件夹路径
    // @return 加载成功返回 Status{}，否则返回错误状态
    // 前置条件：folderPath 非空，路径存在且为可读目录
    // 后置条件：成功时 samples_ 包含所有图像文件的 Sample 数据
    // 注意：大目录（>1000 文件）时 Load 可能较慢，建议在后台线程调用
    Status Load(const std::string& folderPath) override;
    size_t Size() const override;
    Sample Get(size_t index) override;
    int NumClasses() const override { return 1; }

    // 列出文件夹中所有图像文件路径（不加载图像）
    // @param folderPath 图像文件夹路径
    // @return 图像文件路径列表（按字母序排序）
    static std::vector<std::string> ListImageFiles(const std::string& folderPath);

private:
    std::vector<Sample> samples_;  // 加载后的样本列表
};

} // namespace aicore
