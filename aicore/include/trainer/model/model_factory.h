// 模型工厂，用于创建和管理不同架构的神经网络模型
// 提供统一的模型抽象接口，支持 CNN、ResNet 等常见架构
#pragma once
#include "core/types.h"
#include <memory>
#include <string>

namespace aicore {

// 支持的模型架构枚举
enum class ModelArch {
    kSimpleCNN,  // 简单卷积神经网络（适合快速实验）
    kResNet18,   // ResNet-18 残差网络（适合中等规模任务）
    kCustom      // 用户自定义架构
};

// 模型配置结构体
// 描述模型构建所需的所有参数
struct ModelConfig {
    ModelArch arch = ModelArch::kSimpleCNN;  // 模型架构类型
    int numClasses = 10;                      // 分类任务类别数
    int inputChannels = 3;                    // 输入图像通道数（RGB=3，灰度=1）
    int inputSize = 224;                      // 输入图像尺寸（宽高相同）
    std::string pretrainedPath;              // 预训练权重文件路径（为空则不加载）
};

// 模型抽象接口
// 所有具体模型必须实现此接口，提供构建、保存、加载等核心能力
class IModel {
public:
    virtual ~IModel() = default;

    // 根据配置构建模型结构并初始化参数
    // @param config  模型配置
    // @return 成功返回 Success
    virtual Status Build(const ModelConfig& config) = 0;

    // 将模型权重保存到文件
    // @param path  保存路径
    // @return 成功返回 Success
    virtual Status Save(const std::string& path) = 0;

    // 从文件加载模型权重
    // @param path  权重文件路径
    // @return 成功返回 Success
    virtual Status Load(const std::string& path) = 0;

    // 获取模型架构名称
    // @return 架构名称字符串（如 "resnet18"）
    virtual std::string GetArchName() const = 0;
};

// 模型工厂类
// 根据模型架构枚举创建对应的 IModel 实例
// 典型使用：auto model = ModelFactory::Create(ModelArch::kResNet18);
class ModelFactory {
public:
    // 创建指定架构的模型实例
    // @param arch  模型架构枚举值
    // @return 创建的模型实例（unique_ptr 所有权）
    static std::unique_ptr<IModel> Create(ModelArch arch);
};

} // namespace aicore
