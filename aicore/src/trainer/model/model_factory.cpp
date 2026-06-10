#include "trainer/model/model_factory.h"

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
    default:
        return nullptr;
    }
}

} // namespace aicore
