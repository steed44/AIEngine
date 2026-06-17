// 推理后端工厂类
// 根据后端类型创建对应的 IModelBackend 实例
#pragma once
#include "core/model_backend.h"
#include <memory>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 后端工厂类（静态工具类）
// 封装了 IModelBackend 的创建逻辑，根据 BackendType 选择具体的后端实现
class BackendFactory {
public:
    // 创建指定类型的模型后端实例
    // @param type 后端类型枚举值
    // @return 模型后端的唯一指针，不支持的类型返回 nullptr
    static std::unique_ptr<IModelBackend> Create(BackendType type);
};

} // namespace aicore
