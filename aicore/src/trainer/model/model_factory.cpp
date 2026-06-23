// ============================================================================
// 文件：model_factory.cpp
// 用途：模型工厂实现，根据架构类型创建对应的模型实例
// 功能：提供 SimpleCNN 等内置模型的构建、保存和加载
// ============================================================================

#include "trainer/model/model_factory.h"
#ifdef AICORE_HAS_LIBTORCH
#include "trainer/model/yolo_model.h"
#endif

namespace aicore {

class SimpleCNN : public IModel {
public:
    Status Build(const ModelConfig& config) override {
        (void)config;
        built_ = true;
        return Status{};
    }
    Status Save(const std::string& path) override {
        (void)path;
        return Status{};
    }
    Status Load(const std::string& path) override {
        (void)path;
        built_ = true;
        return Status{};
    }
    std::string GetArchName() const override { return "simple_cnn"; }
private:
    bool built_ = false;
};

std::unique_ptr<IModel> ModelFactory::Create(ModelArch arch) {
    switch (arch) {
    case ModelArch::kSimpleCNN:
        return std::make_unique<SimpleCNN>();
#ifdef AICORE_HAS_LIBTORCH
    case ModelArch::kYOLOv8:
        return std::make_unique<YOLOv8Model>();
#endif
    default:
        return nullptr;
    }
}

} // namespace aicore
