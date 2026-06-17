// ============================================================================
// 文件：model_factory.cpp
// 用途：模型工厂实现，根据架构类型创建对应的模型实例
// 功能：提供 SimpleCNN 等内置模型的构建、保存和加载
// ============================================================================

#include "trainer/model/model_factory.h"

namespace aicore {
// ========================================================================
// 命名空间：aicore - AIEngine 核心命名空间
// ========================================================================

// 简易 CNN 模型实现，作为内置的默认模型架构示例
// 适用于快速原型验证和小规模训练任务
class SimpleCNN : public IModel {
public:
    // 根据配置构建模型结构（此处为桩实现，真实场景会搭建卷积层、池化层等）
    Status Build(const ModelConfig& config) override {
        (void)config;
        built_ = true;
        return Status{};
    }
    // 将模型权重保存到指定路径
    Status Save(const std::string& path) override {
        (void)path;
        return Status{};
    }
    // 从指定路径加载模型权重
    Status Load(const std::string& path) override {
        (void)path;
        built_ = true;
        return Status{};
    }
    // 返回模型架构名称标识
    std::string GetArchName() const override { return "simple_cnn"; }
private:
    bool built_ = false; // 标记模型是否已构建
};

// 工厂方法：根据枚举类型创建对应的模型实例
// 参数 arch - 模型架构枚举（kSimpleCNN 等）
// 返回值   - 模型实例的 unique_ptr，不识别的架构返回 nullptr
std::unique_ptr<IModel> ModelFactory::Create(ModelArch arch) {
    switch (arch) {
    case ModelArch::kSimpleCNN:
        return std::make_unique<SimpleCNN>();
    default:
        return nullptr;
    }
}

} // namespace aicore
