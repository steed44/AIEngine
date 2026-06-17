// 模型引擎池
// 管理和复用 IModelBackend 实例，避免重复加载模型，支持按模型 ID 的获取/释放
#pragma once
#include "core/model_backend.h"
#include "backend/backend_factory.h"
#include <memory>
#include <unordered_map>
#include <mutex>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 模型引擎池
// 线程安全的对象池，按 modelId 缓存 IModelBackend 实例
// 当池中空闲实例不足时自动创建新实例（不超过 maxEngines 限制）
class EnginePool {
public:
    // 构造引擎池
    // @param maxEngines 每个模型的最大缓存引擎数，默认 4
    explicit EnginePool(size_t maxEngines = 4);

    // 从池中获取一个模型后端实例
    // @param modelId 模型唯一标识
    // @param info    模型信息（首次加载时使用）
    // @param engine  输出的后端实例共享指针（引用传出）
    // @return 成功返回 Status::kOk
    Status Acquire(const std::string& modelId, const ModelInfo& info,
                   std::shared_ptr<IModelBackend>& engine);
    // 将模型后端实例归还到池中
    // @param modelId 模型唯一标识
    // @param engine  待归还的后端实例
    // @return 成功返回 Status::kOk
    Status Release(const std::string& modelId,
                   std::shared_ptr<IModelBackend> engine);
    // 清空所有缓存的后端实例
    void Clear();

private:
    // 池条目：包含某个模型的所有空闲实例和对应的 ModelInfo
    struct PoolEntry {
        std::vector<std::shared_ptr<IModelBackend>> free;   // 空闲实例列表
        ModelInfo info;                                      // 模型元信息
    };

    size_t maxEngines_;                                     // 每模型最大缓存数
    std::unordered_map<std::string, PoolEntry> pools_;      // modelId -> 池条目
    std::mutex mutex_;                                      // 线程安全互斥锁
};

} // namespace aicore
