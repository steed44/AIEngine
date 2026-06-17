// 数据集抽象接口与常用数据集实现
// 提供统一的样本表示和数据集加载规范，支持 COCO 等标准格式
#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace aicore {

// 单条样本的数据结构
// 包含图像数据、类别标签、文件路径和边界框信息
struct Sample {
    cv::Mat image;          // 图像数据（OpenCV Mat 格式）
    int label = 0;          // 类别标签（0 ~ numClasses-1）
    std::string imagePath;  // 图像的原始文件路径
    BBox bbox;              // 目标检测的边界框（坐标格式与核心定义一致）
};

// 数据集抽象接口
// 所有数据集实现必须继承此类，提供加载、访问和元信息查询功能
class IDataset {
public:
    virtual ~IDataset() = default;

    // 从指定路径加载数据集
    // @param path  数据集路径（支持目录或具体文件，由子类决定）
    // @return 成功返回 Success，失败返回对应错误码
    virtual Status Load(const std::string& path) = 0;

    // 获取数据集总样本数
    // @return 样本数量
    virtual size_t Size() const = 0;

    // 获取指定索引的样本
    // @param index  样本索引（范围 0 ~ Size()-1）
    // @return 样本数据结构
    virtual Sample Get(size_t index) = 0;

    // 获取数据集类别总数
    // @return 类别数量
    virtual int NumClasses() const = 0;
};

// COCO 格式数据集实现
// 支持 COCO JSON 标注格式的数据集加载，适用于目标检测任务
class COCODataset : public IDataset {
public:
    Status Load(const std::string& path) override;
    size_t Size() const override;
    Sample Get(size_t index) override;
    int NumClasses() const override;

private:
    std::vector<Sample> samples_;  // 存储所有样本的内存缓存
};

} // namespace aicore
